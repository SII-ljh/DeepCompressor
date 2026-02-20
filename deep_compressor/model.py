"""DeepCompressor: main model orchestrating all components."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.loss import DistillationLoss, compute_total_loss
from deep_compressor.modules.anchor_align import AnchorAlign
from deep_compressor.modules.down_proj import DownProj
from deep_compressor.modules.fact_decode_head import FactDecodeHead
from deep_compressor.modules.ner_head import NERHead
from deep_compressor.modules.perceiver import GuidedPerceiver
from deep_compressor.modules.query_init import QueryInit
from deep_compressor.modules.up_mlp import UpMLP


class DeepCompressor(nn.Module):
    def __init__(self, config: DeepCompressorConfig, qwen_model=None):
        super().__init__()
        self.config = config
        qcfg = config.qwen
        pcfg = config.perceiver
        pjcfg = config.projection

        # Frozen Qwen model
        if qwen_model is not None:
            self.qwen = qwen_model
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                qcfg.model_name_or_path, torch_dtype=torch.float32,
            )
        for p in self.qwen.parameters():
            p.requires_grad = False

        # Trainable modules
        self.down_proj = DownProj(qcfg.hidden_size, pcfg.perceiver_dim, pjcfg.down_hidden, pjcfg.dropout)
        self.query_init = QueryInit(pcfg.num_queries, pcfg.perceiver_dim, qcfg.hidden_size)
        self.perceiver = GuidedPerceiver(
            dim=pcfg.perceiver_dim, num_heads=pcfg.num_heads, head_dim=pcfg.head_dim,
            ff_mult=pcfg.ff_mult, dropout=pcfg.dropout,
            stage_a_cross_layers=pcfg.stage_a_cross_layers,
            stage_a_self_layers=pcfg.stage_a_self_layers,
            stage_b_layers=pcfg.stage_b_layers,
            stage_c_cross_layers=pcfg.stage_c_cross_layers,
            stage_c_self_layers=pcfg.stage_c_self_layers,
            finbert_enabled=config.finbert.enabled,
            anchor_score_scale_init=pcfg.anchor_score_scale_init,
        )
        self.up_mlp = UpMLP(pcfg.perceiver_dim, qcfg.hidden_size, pjcfg.up_hidden, pjcfg.dropout)

        # FinBERT optional modules
        if config.finbert.enabled:
            self.anchor_align = AnchorAlign(
                config.finbert.hidden_size, pcfg.perceiver_dim, config.finbert.anchor_align_layers,
            )
            self.ner_head = NERHead(config.finbert.hidden_size, config.finbert.num_ner_labels)
            self.fact_decode_head = FactDecodeHead(pcfg.perceiver_dim, config.finbert.hidden_size)

        # Distillation loss helper
        self.distill_loss = DistillationLoss(
            temperature=config.loss.kl_temperature,
            hidden_distill_layers=config.loss.hidden_distill_layers,
            hidden_distill_ramp_steps=config.loss.hidden_distill_ramp_steps,
        )

    def forward(self, *, mode: str = "ntp", **kwargs) -> Dict[str, torch.Tensor]:
        """Unified entry point for DDP compatibility.

        Use mode="ntp" for Stage 1 or mode="qa" for Stage 2.
        All other kwargs are forwarded to the corresponding method.
        """
        if mode == "ntp":
            return self.forward_ntp(**kwargs)
        elif mode == "qa":
            return self.forward_qa(**kwargs)
        raise ValueError(f"Unknown mode: {mode}")

    def _get_qwen_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer from the Qwen model."""
        return self.qwen.model.embed_tokens

    @torch.no_grad()
    def encode_document(self, doc_input_ids: torch.Tensor,
                        doc_attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode document with frozen Qwen, project to Perceiver dim.

        Args:
            doc_input_ids: (batch, doc_len)
            doc_attention_mask: (batch, doc_len)
        Returns:
            byte_array: (batch, doc_len, perceiver_dim)
        """
        outputs = self.qwen(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        # Use last hidden state
        hidden = outputs.hidden_states[-1]  # (B, doc_len, qwen_dim)
        # Re-enable grad for DownProj
        hidden = hidden.detach()
        return self.down_proj(hidden)

    @torch.no_grad()
    def encode_question(self, q_input_ids: torch.Tensor,
                        q_attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode question with frozen Qwen, mean-pool, then QueryInit.

        Args:
            q_input_ids: (batch, q_len)
            q_attention_mask: (batch, q_len)
        Returns:
            queries: (batch, num_queries, perceiver_dim)
        """
        outputs = self.qwen(
            input_ids=q_input_ids, attention_mask=q_attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # (B, q_len, qwen_dim)
        # Mean pool over valid positions
        mask_expanded = q_attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        pooled = pooled.detach()  # (B, qwen_dim)
        return self.query_init(pooled)

    def compress(self, queries: torch.Tensor, byte_array: torch.Tensor,
                 byte_mask: Optional[torch.Tensor] = None,
                 anchor_scores: Optional[torch.Tensor] = None,
                 anchor_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run Perceiver compression.

        Returns:
            latent_array: (batch, num_queries, perceiver_dim)
        """
        return self.perceiver(
            queries, byte_array, byte_mask=byte_mask,
            anchor_scores=anchor_scores, anchor_embs=anchor_embs,
        )

    def decode(self, prefix_embeds: torch.Tensor, suffix_ids: torch.Tensor,
               suffix_attention_mask: torch.Tensor,
               labels: Optional[torch.Tensor] = None,
               output_hidden_states: bool = False):
        """Decode by concatenating prefix embeddings with suffix token embeddings.

        Args:
            prefix_embeds: (batch, num_queries, qwen_dim) — from UpMLP
            suffix_ids: (batch, suffix_len) — question + answer token ids
            suffix_attention_mask: (batch, suffix_len)
            labels: (batch, suffix_len) — target ids for loss, or None
            output_hidden_states: whether to return hidden states
        Returns:
            model output with loss if labels provided
        """
        embed_layer = self._get_qwen_embeddings()
        suffix_embeds = embed_layer(suffix_ids)  # (B, suffix_len, qwen_dim)

        # Concatenate prefix and suffix embeddings
        inputs_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)

        # Build attention mask: all ones for prefix, then suffix mask
        B = prefix_embeds.shape[0]
        prefix_len = prefix_embeds.shape[1]
        prefix_mask = torch.ones(B, prefix_len, device=prefix_embeds.device, dtype=suffix_attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, suffix_attention_mask], dim=1)

        # Build labels: -100 for prefix positions, then actual labels
        if labels is not None:
            prefix_labels = torch.full((B, prefix_len), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

    @torch.no_grad()
    def generate_answer(self, prefix_embeds: torch.Tensor,
                        question_ids: torch.Tensor,
                        question_mask: torch.Tensor,
                        tokenizer, max_new_tokens: int = 64) -> torch.Tensor:
        """Generate answer tokens given compressed prefix and question.

        Uses greedy decoding (do_sample=False) for reproducible evaluation.

        Args:
            prefix_embeds: (batch, num_queries, qwen_dim) — from UpMLP
            question_ids: (batch, q_len)
            question_mask: (batch, q_len)
            tokenizer: tokenizer with pad_token_id and eos_token_id
            max_new_tokens: maximum number of tokens to generate
        Returns:
            generated token ids: (batch, num_generated)
        """
        embed_layer = self._get_qwen_embeddings()
        q_embeds = embed_layer(question_ids)  # (B, q_len, D)
        inputs_embeds = torch.cat([prefix_embeds, q_embeds], dim=1)

        B = prefix_embeds.shape[0]
        prefix_len = prefix_embeds.shape[1]
        prefix_mask = torch.ones(B, prefix_len, device=prefix_embeds.device,
                                 dtype=question_mask.dtype)
        attention_mask = torch.cat([prefix_mask, question_mask], dim=1)

        out = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # generate() returns full sequence including input positions;
        # strip the input-length prefix to get only generated tokens
        input_len = inputs_embeds.shape[1]
        return out[:, input_len:]

    def forward_ntp(self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor,
                    segment_ids: torch.Tensor, segment_attention_mask: torch.Tensor,
                    segment_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """NTP pretraining forward: compress doc → prefix → next-token prediction on segment.

        Args:
            doc_input_ids, doc_attention_mask: document tokens
            segment_ids, segment_attention_mask: continuation segment tokens
            segment_labels: target ids for NTP
        Returns:
            dict with 'total' loss and 'ntp' loss
        """
        byte_array = self.encode_document(doc_input_ids, doc_attention_mask)

        # For NTP, use zero question vector → pure base queries
        B = doc_input_ids.shape[0]
        zero_pooled = torch.zeros(B, self.config.qwen.hidden_size, device=doc_input_ids.device)
        queries = self.query_init(zero_pooled)

        latent_array = self.compress(queries, byte_array, byte_mask=doc_attention_mask)
        prefix_embeds = self.up_mlp(latent_array)

        outputs = self.decode(prefix_embeds, segment_ids, segment_attention_mask, labels=segment_labels)

        return {"total": outputs.loss, "ntp": outputs.loss}

    def forward_qa(self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor,
                   q_input_ids: torch.Tensor, q_attention_mask: torch.Tensor,
                   answer_ids: torch.Tensor, answer_attention_mask: torch.Tensor,
                   answer_labels: torch.Tensor,
                   teacher_logits: Optional[torch.Tensor] = None,
                   teacher_hidden: Optional[list] = None,
                   global_step: int = 0,
                   anchor_scores: Optional[torch.Tensor] = None,
                   anchor_embs: Optional[torch.Tensor] = None,
                   original_anchors: Optional[torch.Tensor] = None,
                   ) -> Dict[str, torch.Tensor]:
        """QA + distillation forward pass.

        Args:
            doc_input_ids, doc_attention_mask: document tokens
            q_input_ids, q_attention_mask: question tokens
            answer_ids, answer_attention_mask: answer tokens (input side)
            answer_labels: target answer ids
            teacher_logits: (batch, qa_len, vocab) — pre-computed teacher logits
            teacher_hidden: list of (batch, qa_len, dim) — pre-computed teacher hidden states
            global_step: current training step
            anchor_scores: (batch, doc_len) — FinBERT entity scores (optional)
            anchor_embs: (batch, top_k, perceiver_dim) — aligned anchor embeddings (optional)
            original_anchors: (batch, top_k, finbert_dim) — original FinBERT anchors (optional)
        Returns:
            dict with 'total' and all loss components
        """
        byte_array = self.encode_document(doc_input_ids, doc_attention_mask)
        queries = self.encode_question(q_input_ids, q_attention_mask)

        latent_array = self.compress(
            queries, byte_array, byte_mask=doc_attention_mask,
            anchor_scores=anchor_scores, anchor_embs=anchor_embs,
        )
        prefix_embeds = self.up_mlp(latent_array)

        # Concatenate question + answer as suffix
        suffix_ids = torch.cat([q_input_ids, answer_ids], dim=1)
        suffix_mask = torch.cat([q_attention_mask, answer_attention_mask], dim=1)

        # Labels: -100 for question positions, answer_labels for answer positions
        q_len = q_input_ids.shape[1]
        q_labels = torch.full_like(q_input_ids, -100)
        full_labels = torch.cat([q_labels, answer_labels], dim=1)

        need_hidden = teacher_hidden is not None
        outputs = self.decode(
            prefix_embeds, suffix_ids, suffix_mask,
            labels=full_labels, output_hidden_states=need_hidden,
        )

        qa_ce_loss = outputs.loss
        lcfg = self.config.loss

        # Distillation losses
        kl_loss = None
        hidden_mse_loss = None
        if teacher_logits is not None:
            # Student logits for question+answer region (skip prefix in model output logits)
            prefix_len = prefix_embeds.shape[1]
            student_logits = outputs.logits[:, prefix_len:, :]  # (B, qa_len, V)

            # Build answer mask for the qa region
            answer_mask = torch.zeros_like(suffix_mask, dtype=torch.float)
            answer_mask[:, q_len:] = answer_attention_mask.float()

            kl_loss = self.distill_loss.compute_kl_loss(student_logits, teacher_logits, answer_mask)

        if teacher_hidden is not None:
            prefix_len = prefix_embeds.shape[1]
            student_all_hidden = outputs.hidden_states  # tuple of (B, total_len, D)
            # Extract question+answer region from student
            student_hidden = [h[:, prefix_len:, :] for h in student_all_hidden]

            shared_mask = suffix_mask.float()
            hidden_mse_loss = self.distill_loss.compute_hidden_mse_loss(
                student_hidden, teacher_hidden, shared_mask, global_step,
            )

        # Anchor reconstruction loss
        anchor_recon_loss = None
        if self.config.finbert.enabled and original_anchors is not None:
            anchor_recon_loss = self.fact_decode_head(latent_array, original_anchors)

        return compute_total_loss(
            qa_ce_loss,
            kl_loss=kl_loss,
            hidden_mse_loss=hidden_mse_loss,
            anchor_recon_loss=anchor_recon_loss,
            qa_ce_weight=lcfg.qa_ce_weight,
            kl_weight=lcfg.kl_weight,
            hidden_mse_weight=lcfg.hidden_mse_weight,
            anchor_recon_weight=lcfg.anchor_recon_weight,
        )

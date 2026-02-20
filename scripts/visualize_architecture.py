#!/usr/bin/env python3
"""
Deep Compressor 模型架构可视化
==============================

四种工具, 各司其职, 不重复:

  torchinfo    — 文本摘要: 完整模型层次 + 参数统计 (一份即可)
  torchviz     — 计算图: 仅对小模块 (DownProj/QueryInit/UpMLP) 生成可读的梯度流图
  Netron       — ONNX 交互查看: 导出各模块供浏览器查看算子级结构
  HiddenLayer  — 高层架构图: torch.fx 追踪后渲染模块级拓扑 (含 Perceiver)

安装:
    pip install torchinfo torchviz onnx onnxscript netron graphviz
    brew install graphviz

用法:
    python scripts/visualize_architecture.py                # 全部生成
    python scripts/visualize_architecture.py --torchinfo    # 仅文本摘要
    python scripts/visualize_architecture.py --torchviz     # 仅计算图
    python scripts/visualize_architecture.py --netron       # 仅导出 ONNX
    python scripts/visualize_architecture.py --hiddenlayer  # 仅高层架构图
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
    DeepCompressorConfig, FinBERTConfig, LossConfig,
    PerceiverConfig, ProjectionConfig, QwenConfig, TrainingConfig,
)
from deep_compressor.model import DeepCompressor


# ── Dummy Qwen (不下载权重, 结构兼容) ──────────────────────────

class _DummyOutput:
    def __init__(self, hidden_states, loss=None, logits=None):
        self.hidden_states = hidden_states
        self.loss = loss
        self.logits = logits


class _DummyQwen(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._vocab_size = vocab_size

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, output_hidden_states=False, use_cache=False):
        h = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
        states = [h]
        for _ in range(self._num_layers):
            h = h + 0
            states.append(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self._vocab_size), labels.view(-1), ignore_index=-100)
        return _DummyOutput(
            hidden_states=tuple(states) if output_hidden_states else None,
            loss=loss, logits=logits)


def _build():
    """构建 FinBERT ON 的模型 (包含所有模块, 一次性)。"""
    cfg = DeepCompressorConfig(
        qwen=QwenConfig(model_name_or_path="dummy", hidden_size=1024,
                        num_hidden_layers=28, vocab_size=1000),
        finbert=FinBERTConfig(enabled=True, hidden_size=768),
        perceiver=PerceiverConfig(),
        projection=ProjectionConfig(),
        loss=LossConfig(),
        training=TrainingConfig(),
    )
    qwen = _DummyQwen(cfg.qwen.vocab_size, cfg.qwen.hidden_size, cfg.qwen.num_hidden_layers)
    model = DeepCompressor(cfg, qwen_model=qwen)
    return cfg, model


# ══════════════════════════════════════════════════════════════
# 1. torchinfo — 一份完整文本摘要
# ══════════════════════════════════════════════════════════════

def run_torchinfo(out: Path):
    try:
        from torchinfo import summary
    except ImportError:
        print("[SKIP] pip install torchinfo")
        return

    cfg, model = _build()
    model.eval()
    print("\n[torchinfo] 模型结构摘要")

    # (a) 完整层次
    result = summary(model, depth=4,
                     col_names=["num_params", "trainable"],
                     col_width=18, row_settings=["var_names"], verbose=2)
    fpath = out / "torchinfo_model.txt"
    with open(fpath, "w") as f:
        f.write(str(result))
    print(f"  完整模型 -> {fpath}")

    # (b) GuidedPerceiver 带形状
    r = summary(model.perceiver, depth=3,
                input_size=[(1, 64, 1024), (1, 128, 1024)],
                col_names=["input_size", "output_size", "num_params"],
                col_width=18, verbose=1)
    fpath = out / "torchinfo_perceiver.txt"
    with open(fpath, "w") as f:
        f.write(str(r))
    print(f"  Perceiver -> {fpath}")

    # (c) 参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    lines = [
        f"{'='*45}",
        f"  可训练:  {trainable:>12,}  ({100*trainable/total:.1f}%)",
        f"  冻结:    {frozen:>12,}  ({100*frozen/total:.1f}%)",
        f"  总计:    {total:>12,}",
        f"{'='*45}",
        "  模块明细:",
    ]
    for name in ["down_proj", "query_init", "perceiver", "up_mlp",
                 "anchor_align", "ner_head", "fact_decode_head"]:
        if hasattr(model, name):
            cnt = sum(p.numel() for p in getattr(model, name).parameters())
            lines.append(f"    {name:<25s} {cnt:>12,}")
    lines.append(f"    {'qwen (frozen)':<25s} {frozen:>12,}")
    text = "\n".join(lines)
    print(text)
    fpath = out / "torchinfo_stats.txt"
    with open(fpath, "w") as f:
        f.write(text)


# ══════════════════════════════════════════════════════════════
# 2. torchviz — 仅对小模块生成计算图 (大模块不可读)
# ══════════════════════════════════════════════════════════════

def run_torchviz(out: Path):
    try:
        from torchviz import make_dot
    except ImportError:
        print("[SKIP] pip install torchviz")
        return

    cfg, model = _build()
    model.train()
    print("\n[torchviz] 子模块计算图 (仅小模块, 大模块跳过)")

    targets = {
        "DownProj": (model.down_proj,
                     lambda: torch.randn(1, 16, 1024, requires_grad=True)),
        "QueryInit": (model.query_init,
                      lambda: torch.randn(1, 1024, requires_grad=True)),
        "UpMLP": (model.up_mlp,
                  lambda: torch.randn(1, 64, 1024, requires_grad=True)),
        "NERHead": (model.ner_head,
                    lambda: torch.randn(1, 16, 768, requires_grad=True)),
        "AnchorAlign": (model.anchor_align,
                        lambda: torch.randn(1, 32, 768, requires_grad=True)),
    }

    for name, (mod, make_input) in targets.items():
        try:
            x = make_input()
            y = mod(x)
            loss = y.sum() if y.dim() > 0 else y
            dot = make_dot(loss, params=dict(mod.named_parameters()))
            fpath = out / f"torchviz_{name}"
            dot.render(str(fpath), format="png", cleanup=True)
            print(f"  {name} -> {fpath}.png")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")

    print("  (GuidedPerceiver / 完整模型: 算子太多, 图不可读, 已跳过)")


# ══════════════════════════════════════════════════════════════
# 3. Netron — ONNX 导出 (不含权重数据, 只看结构)
# ══════════════════════════════════════════════════════════════

def run_netron(out: Path, interactive: bool = False):
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("[SKIP] pip install onnx onnxscript")
        return

    cfg, model = _build()
    model.eval()
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    print("\n[Netron] ONNX 导出")

    exports = {
        "DownProj": (model.down_proj,
                     torch.randn(1, 128, 1024),
                     ["hidden_states"], ["byte_array"],
                     {"hidden_states": {1: "seq"}}),
        "QueryInit": (model.query_init,
                      torch.randn(1, 1024),
                      ["question_pooled"], ["queries"],
                      {"question_pooled": {0: "batch"}}),
        "GuidedPerceiver": (model.perceiver,
                            (torch.randn(1, 64, 1024), torch.randn(1, 128, 1024)),
                            ["queries", "byte_array"], ["latent_array"],
                            {"byte_array": {1: "doc_len"}}),
        "UpMLP": (model.up_mlp,
                  torch.randn(1, 64, 1024),
                  ["latent_array"], ["prefix_embeds"],
                  {}),
        "AnchorAlign": (model.anchor_align,
                        torch.randn(1, 32, 768),
                        ["finbert_anchors"], ["aligned_anchors"],
                        {}),
        "NERHead": (model.ner_head,
                    torch.randn(1, 128, 768),
                    ["finbert_hidden"], ["entity_scores"],
                    {"finbert_hidden": {1: "seq"}}),
    }

    for name, (mod, inp, in_names, out_names, dyn) in exports.items():
        fpath = onnx_dir / f"{name}.onnx"
        try:
            torch.onnx.export(mod, inp, str(fpath),
                              input_names=in_names, output_names=out_names,
                              dynamic_axes=dyn, opset_version=18)
            print(f"  {name} -> {fpath}")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")

    # 删除 .onnx.data 权重文件 (只保留结构)
    for f in onnx_dir.glob("*.onnx.data"):
        f.unlink()
        print(f"  (已删除权重文件 {f.name}, 只保留结构)")

    print(f"\n  查看方式: netron outputs/visualizations/onnx/GuidedPerceiver.onnx")
    print(f"  或拖入 https://netron.app")

    if interactive:
        try:
            import netron
            perc = onnx_dir / "GuidedPerceiver.onnx"
            if perc.exists():
                netron.start(str(perc), browse=True)
        except ImportError:
            pass


# ══════════════════════════════════════════════════════════════
# 4. HiddenLayer — 高层架构图 (torch.fx 追踪)
# ══════════════════════════════════════════════════════════════

def run_hiddenlayer(out: Path):
    """
    hiddenlayer 与 PyTorch >=2.0 不兼容 (torch.onnx._optimize_trace 已移除).
    这里用 torch.fx 符号追踪 + graphviz 渲染, 效果等价:
    模块级节点 + 数据流边, 而不是底层算子。
    """
    # 先尝试原生 hiddenlayer
    hl_ok = False
    try:
        import hiddenlayer as hl
        hl.build_graph(nn.Linear(2, 2), torch.randn(1, 2))
        hl_ok = True
    except Exception:
        pass

    if hl_ok:
        _hiddenlayer_native(hl, out)
    else:
        _hiddenlayer_fx(out)


def _hiddenlayer_native(hl, out: Path):
    """原生 hiddenlayer (PyTorch < 2.0)。"""
    cfg, model = _build()
    model.eval()
    print("\n[HiddenLayer] 高层架构图 (native)")

    for name, (mod, inp) in _trace_targets(model, cfg).items():
        try:
            g = hl.build_graph(mod, inp)
            g.theme = hl.graph.THEMES["blue"].copy()
            fpath = out / f"hiddenlayer_{name}.png"
            g.save(str(fpath), format="png")
            print(f"  {name} -> {fpath}")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")


def _hiddenlayer_fx(out: Path):
    """torch.fx 替代方案, 控制深度只展示模块级节点。"""
    try:
        import graphviz
    except ImportError:
        print("[SKIP] pip install graphviz")
        return

    from torch.fx import symbolic_trace

    cfg, model = _build()
    model.eval()
    print("\n[HiddenLayer] 高层架构图 (torch.fx + graphviz)")

    # max_depth: call_module target 允许的 '.' 分隔层数
    # GuidedPerceiver depth=1 => 只保留 stage_a_cross.0 级别的 PerceiverBlock
    depth_map = {"DownProj": 99, "QueryInit": 99, "GuidedPerceiver": 1, "UpMLP": 99}

    for name, (mod, _) in _trace_targets(model, cfg).items():
        max_depth = depth_map.get(name, 99)
        try:
            traced = symbolic_trace(mod)
            dot = graphviz.Digraph(
                name, format="png",
                graph_attr={"rankdir": "TB", "fontname": "Helvetica",
                            "bgcolor": "#FAFAFA", "dpi": "150",
                            "label": name, "labelloc": "t", "fontsize": "14"},
                node_attr={"fontname": "Helvetica", "fontsize": "10",
                           "style": "filled", "shape": "record"},
            )

            # 对大模块: 只保留 call_module + placeholder + output, 过滤掉函数/属性噪声
            modules_only = (max_depth < 99)

            # node.name -> 合并后的显示 ID
            redirect = {}
            visible = set()

            for node in traced.graph.nodes:
                if node.op == "call_module":
                    parts = node.target.split(".")
                    if len(parts) > max_depth + 1:
                        parent = ".".join(parts[:max_depth + 1])
                        redirect[node.name] = parent
                        visible.add(parent)
                    else:
                        visible.add(node.name)
                elif node.op in ("placeholder", "output"):
                    visible.add(node.name)
                elif not modules_only:
                    # 小模块: 保留所有节点
                    if node.op == "call_function":
                        visible.add(node.name)
                    elif node.op == "get_attr":
                        visible.add(node.name)
                    elif node.op == "call_method":
                        visible.add(node.name)
                else:
                    # 大模块: 非 call_module 节点做透传 (查找它连到的下一个可见节点)
                    pass

            # 对不可见节点, 把它们的输入透传给输出 (找最近的可见祖先)
            def resolve(n):
                """递归找到 n 对应的可见节点 ID。"""
                nid = redirect.get(n, n)
                if nid in visible:
                    return nid
                # 找这个节点的第一个输入, 递归上溯
                for node in traced.graph.nodes:
                    if node.name == n:
                        for arg in node.args:
                            if hasattr(arg, "name"):
                                return resolve(arg.name)
                return None

            # 创建节点
            added = set()
            for node in traced.graph.nodes:
                nid = redirect.get(node.name, node.name)
                if nid not in visible or nid in added:
                    continue
                added.add(nid)

                if node.op == "placeholder":
                    dot.node(nid, f"Input: {node.name}", fillcolor="#B3D9FF")
                elif node.op == "call_module":
                    target = nid if nid != node.name else node.target
                    submod = traced.get_submodule(target)
                    dot.node(nid, f"{target}\\n{type(submod).__name__}",
                             fillcolor="#B8E6B8")
                elif node.op == "call_function":
                    dot.node(nid, node.target.__name__, fillcolor="#FFD9B3")
                elif node.op == "call_method":
                    dot.node(nid, f".{node.target}()", fillcolor="#FFFFB3")
                elif node.op == "get_attr":
                    dot.node(nid, f"param: {node.target}", fillcolor="#E6B3FF")
                elif node.op == "output":
                    dot.node(nid, "Output", fillcolor="#D5F5E3")

            # 创建边
            edges_seen = set()
            for node in traced.graph.nodes:
                dst = resolve(node.name)
                if dst is None:
                    continue
                for arg in node.args:
                    if hasattr(arg, "name"):
                        src = resolve(arg.name)
                        if src and src != dst and (src, dst) not in edges_seen:
                            dot.edge(src, dst)
                            edges_seen.add((src, dst))
                if isinstance(node.kwargs, dict):
                    for v in node.kwargs.values():
                        if hasattr(v, "name"):
                            src = resolve(v.name)
                            if src and src != dst and (src, dst) not in edges_seen:
                                dot.edge(src, dst, style="dashed")
                                edges_seen.add((src, dst))

            fpath = out / f"hiddenlayer_{name}"
            dot.render(str(fpath), cleanup=True)
            print(f"  {name} -> {fpath}.png")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")


def _trace_targets(model, cfg):
    """只追踪值得看的模块, 不重复。"""
    return {
        "DownProj": (model.down_proj,
                     torch.randn(1, 128, cfg.qwen.hidden_size)),
        "QueryInit": (model.query_init,
                      torch.randn(1, cfg.qwen.hidden_size)),
        "GuidedPerceiver": (model.perceiver,
                            (torch.randn(1, 64, 1024), torch.randn(1, 128, 1024))),
        "UpMLP": (model.up_mlp,
                  torch.randn(1, 64, 1024)),
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep Compressor 模型架构可视化")
    parser.add_argument("--all", action="store_true", help="全部生成")
    parser.add_argument("--torchinfo", action="store_true", help="文本摘要")
    parser.add_argument("--torchviz", action="store_true", help="计算图")
    parser.add_argument("--netron", action="store_true", help="导出 ONNX")
    parser.add_argument("--hiddenlayer", action="store_true", help="高层架构图")
    parser.add_argument("--interactive", action="store_true", help="Netron 打开浏览器")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()

    if not any([args.all, args.torchinfo, args.torchviz, args.netron, args.hiddenlayer]):
        args.all = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {out.resolve()}\n")

    if args.all or args.torchinfo:
        run_torchinfo(out)
    if args.all or args.torchviz:
        run_torchviz(out)
    if args.all or args.netron:
        run_netron(out, interactive=args.interactive)
    if args.all or args.hiddenlayer:
        run_hiddenlayer(out)

    print(f"\n{'='*50}")
    print(f"  完成! -> {out.resolve()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

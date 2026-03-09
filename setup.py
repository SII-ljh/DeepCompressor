"""Setup script for Deep Compressor."""

from setuptools import setup, find_packages

setup(
    name="deep_compressor",
    version="0.1.0",
    description="Deep Compressor: Compress ultra-long financial texts for QA",
    author="Deep Compressor Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "scripts.*"]),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "datasets>=2.20.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "optuna>=3.5.0",
        "wandb>=0.16.0",
        "jieba>=0.42.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)

# Long-Context Needle-in-a-Haystack Benchmark

## Abstract
This repository provides a reproducible PyTorch benchmark for long-context retrieval using a synthetic needle-in-a-haystack task. We compare three sequence-modeling families under controlled context growth: a vanilla GRU baseline, a memory-augmented GRU with explicit hidden-state retrieval, and a transformer encoder with full attention. The benchmark is designed to expose how retrieval quality degrades as sequence length increases and to isolate the effect of explicit memory on long-range recall.

## Problem Statement
Sequence models often perform well on short contexts but degrade when a crucial token is placed far from the end of the sequence. This project asks a focused question: when a synthetic passkey is hidden in a long stream of distractor tokens, how do recurrent, memory-augmented recurrent, and attention-based models compare as the context grows from 1K to 8K tokens?

## Repository Layout
```text
niah_benchmark/
  README.md
  requirements.txt
  configs/
    default.yaml
    cpu_demo.yaml
    research.yaml
    h100_a100.yaml
  data/
  src/
    data/
      __init__.py
      dataset.py
      generate_dataset.py
    models/
      __init__.py
      build.py
      gru_baseline.py
      memory_gru.py
      transformer_encoder.py
    memory/
      __init__.py
      retrieval.py
    training/
      __init__.py
      trainer.py
    evaluation/
      __init__.py
      evaluator.py
      metrics.py
    utils/
      __init__.py
      config.py
      io.py
      runtime.py
      seed.py
  experiments/
    run_scaling.py
  outputs/
    logs/
    plots/
    checkpoints/
  scripts/
    train.py
    eval.py
    plot_results.py
  notebooks/
    colab_demo.ipynb
```

## Methodology
Each synthetic sample is a long token sequence filled with distractor words and one or more injected passkeys of the form `PASSKEY-XXXX-YYYY`. The model receives the sequence and must classify which passkey was present. Because the answer space is a fixed passkey vocabulary, retrieval can be measured with exact-match accuracy, top-1 retrieval accuracy, and failure rate.

### Synthetic Task
- Sequence lengths: `1000, 2000, 4000, 8000`
- Splits: train, validation, test
- Question: `"What is the passkey?"`
- Answer: a single passkey token
- Metadata includes sequence length, exact needle position, and coarse position bucket (`early`, `middle`, `late`)

### Models
- `GRU baseline`: embeds the sequence, encodes it with a GRU, and predicts from the final hidden state.
- `Memory-Augmented GRU`: encodes the sequence with a GRU, stores sparse hidden states in an explicit memory cache, retrieves the most similar memory slots via cosine similarity, and fuses retrieved memory with the final hidden state before prediction.
- `Transformer encoder`: prepends a learned `[CLS]` token, applies positional encoding and full self-attention, and predicts from the `[CLS]` representation.

The intended qualitative result is straightforward: vanilla recurrent models struggle as the passkey moves far from the end, explicit memory improves retrieval stability, and transformer attention provides the strongest global access pattern.

## Experimental Setup
Default research experiments train each model across the full sequence-length suite and repeat runs across three random seeds. All experiments are fully synthetic, use fixed seeds for reproducibility, and run in PyTorch with optional CUDA acceleration. The accelerated GPU configs also support mixed precision, curriculum staging, resumable scaling runs, and stronger model sizes for A100/H100-class hardware.

### Metrics
- Exact match accuracy
- Top-1 retrieval accuracy
- Failure rate
- Accuracy by sequence length
- Accuracy by needle position bucket

## How To Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Generate data and train a single model:

```bash
python scripts/train.py --config configs/default.yaml --model-type gru --experiment-name gru_default
```

Evaluate a checkpoint:

```bash
python scripts/eval.py --config configs/default.yaml --model-type gru --experiment-name gru_default --checkpoint outputs/checkpoints/gru_default_best.pt
```

Run the scaling benchmark with three seeds:

```bash
python experiments/run_scaling.py --config configs/research.yaml --models gru memory_gru transformer --seeds 123 456 789
```

For A100/H100 GPUs, use the stronger accelerated config:

```bash
python experiments/run_scaling.py --config configs/h100_a100.yaml --models gru memory_gru transformer --seeds 123 456 789
```

Generate plots:

```bash
python scripts/plot_results.py
```

## Google Colab
The notebook [notebooks/colab_demo.ipynb](notebooks/colab_demo.ipynb) provides a lightweight end-to-end path for CPU or GPU Colab sessions. For quick iteration, start with `configs/cpu_demo.yaml`. For stronger multi-seed runs, use `configs/research.yaml`. On A100/H100 GPUs, prefer `configs/h100_a100.yaml` to enable mixed precision, curriculum staging, and larger models.

## Results Interpretation
This benchmark is intentionally narrow and interpretable:
- If GRU accuracy drops sharply with length, the benchmark is capturing recurrent long-context failure.
- If memory-GRU degrades more slowly, the explicit memory cache is improving access to earlier evidence.
- If the transformer remains strongest, attention is providing a useful upper-bound baseline for global retrieval.

The plotting utilities produce publication-style figures for:
- Accuracy vs sequence length
- Failure rate vs sequence length
- Needle position sensitivity

## Example Outputs
Expected artifacts are written to:
- `outputs/checkpoints/`: best model checkpoints
- `outputs/logs/`: train histories, evaluation predictions, and aggregated scaling results
- `outputs/plots/`: saved figures in PNG format

## Key Findings This Benchmark Is Designed To Surface
1. Vanilla RNN/GRU models fail progressively as context length increases.
2. Explicit memory retrieval improves stability over the vanilla recurrent baseline.
3. Transformer encoders provide the strongest long-range retrieval under full attention.
4. Needle position matters: early-context needles are usually harder for recurrent models than late-context needles.

## Reproducibility Notes
- Seeds are fixed at dataset generation, training, and evaluation time.
- No external datasets are required.
- All configuration is centralized in YAML files for easy experiment control.

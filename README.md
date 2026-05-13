# MLLM_project

BEV road centerline reconstruction VLM based on a LLaVA-style multimodal stack.

Current working branch:

```text
qwen3vl_deepstack_checkpointing
```

This branch supports:

- Qwen2.5 / Qwen3-VL language backbones.
- DINOv2 / DINOv3 vision towers.
- Qwen3-VL-style DeepStack visual injection.
- Training with or without DeepStack.
- LoRA and full-parameter checkpoint loading.
- Inference that recovers DeepStack settings from checkpoint metadata.
- Rank-0-only training, eval, and checkpoint metric files with `DI_throughput: ... tokens/s/npu`.

## Documentation

- Main architecture, training, inference, logging, and validation notes:
  [docs/qwen3vl_dinov3_deepstack.md](docs/qwen3vl_dinov3_deepstack.md)
- Script naming and placement:
  [scripts/README.md](scripts/README.md)

## Main NPU Scripts

Full-parameter training:

```bash
bash scripts/train_full_dinov2_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov2_qwen3vl-8b_no-deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_deepstack_npu.sh
bash scripts/train_full_dinov3_qwen3vl-8b_no-deepstack_npu.sh
```

Full-checkpoint testing:

```bash
bash scripts/test_full_dinov2_qwen3vl-8b_npu.sh
bash scripts/test_full_dinov3_qwen3vl-8b_npu.sh
```

The test scripts do not need a manual DeepStack flag. They infer whether DeepStack is enabled from the checkpoint configuration.

## Validation

The latest lightweight GPU matrix passed on two GPUs:

```text
qwen2.5 + dinov2 + deepstack on/off
qwen2.5 + dinov3 + deepstack on/off
qwen3vl-2b + dinov2 + deepstack on/off
qwen3vl-2b + dinov3 + deepstack on/off
```

The key `qwen3vl + dinov3 + deepstack` path was also checked separately:

```text
Loaded 754/754 model tensors from full-finetune checkpoint
DeepStack enabled
shape alignment passed
2-GPU training passed
2-GPU inference passed
no UNEXPECTED vision-weight load warning
```

## Training Logs

The training console uses the Hugging Face Trainer default progress and metric output. `DI_throughput` is not injected into the console output.

Throughput and runtime metrics are written to log files:

```text
train_metrics.log
eval_metrics.log
checkpoint_events.log
```

Example `train_metrics.log` line:

```text
time: 2026-05-13 15:43:14  global_step: 1  epoch: 1  loss: 1.23  learning_rate: 2e-05  DI_throughput: 12716.48 tokens/s/npu
```

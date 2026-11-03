# Validation: LoRA-ISR ARC add-on

Date: 2026-11-02. Status: **18 CPU tests passed**, plus **5 existing baseline pipeline tests passed unchanged**.

Actual environment used: Python 3.12, torch 2.6.0+cpu, transformers 4.50.3, accelerate 1.6.0, safetensors 0.5.3, Modal SDK 1.6.3. PEFT 0.14.1.dev0 installed editable from the authors' repository at commit `7c9eeabe54cf94e2051632829019c86c4198b3df`. Modal image uses the original requirements and Python 3.10, with W&B upgraded to 0.22.3 as in the existing working pipeline.

Verified with actual PyTorch / Trainer / Accelerate, not an import-only mock:

- Gram expression equals explicit dense Frobenius norm.
- Float64 gradcheck and analytical gradient through both numerator and denominator.
- Scale invariance, rank-one penalty/gradient, known-spectrum power iteration and exact small-core SVD.
- Warm-up and lambda=0 bypass PI/Gram/private RNG; active PI does not consume global torch RNG.
- Fixed total M when another matrix is skipped; degenerate PI vector restarts.
- Serialization restores PI vectors, per-module RNG, successful/attempted optimizer counters; mismatched config and changed total updates are refused.
- Auxiliary gradient coefficient is unchanged with accumulation 1, 2, 4, 128; exactly one auxiliary backward per update.
- Partial final accumulation group with micro-batch size 1 executes one auxiliary backward at its final boundary.
- Actual CPU GradScaler scales both CE and auxiliary, then unscale/clipping matches the unscaled reference; an injected overflow skips optimizer weights and does not advance the ISR successful-update counter. CUDA FP16 execution remains untested.
- With lambda=0, full Trainer training with dropout produces bitwise-equal model weights to the original Trainer on CPU.
- Interrupted/checkpoint/resumed Trainer training matches uninterrupted training, including controller state.
- Tiny Llama with the upstream PEFT LoRA trains with all 10 selected modules, saves/reloads the standard adapter with matching logits, and resumes a real PEFT checkpoint to match uninterrupted weights.
- ISR plans contain only requested ARC tasks, use LoRA backend and LoRA learning rate, and distinguish changed ISR hyperparameters in run IDs.
- Checkpoint readiness requires ISR state; result collection refuses validation/test split mismatches.
- Comparison joins by task/rank/seed/test split, retains separate ISR hyperparameter variants, refuses ambiguous baselines and leaves absent references empty.
- Checked build-time source patch applies to the real pinned training script and compiles.
- Modal app imports, local plan prints two rank-8 H100 configurations, and Modal CLI `--help` accepts the documented flags.

CPU test command (inside an environment containing the versions above and upstream PEFT):

```bash
POLAR_UPSTREAM_DIR=/path/to/pinned/polar python -m unittest discover -s tests -p test_lora_isr.py -v
```

The pinned source is also used at Modal image build, with checked replacements that fail if expected source fragments change. No LoRA forward code or AdamW implementation was rewritten. ISR only changes the Trainer auxiliary-loss contribution and extends checkpoint state.

Not run here: authenticated Modal image build/deployment, remote H100 jobs, full Llama-2-7B training, real ARC evaluation, W&B authentication/sync, CUDA FP16 overflow/resume. The machine used for implementation has no CUDA GPU or the user's service credentials. Smoke and full-run commands in the README are to be executed by the user; no GPU jobs have been submitted while preparing this package.

Consequently, this is tested implementation code and a reproducible run configuration, not a claim of a measured accuracy improvement or a GPU benchmark. Lambda/warm-up/ramp are pilot choices and should be selected using validation data.

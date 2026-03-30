# [Round 1] Baseline: PSNR=39.57 dB — Synthetic benchmark scaffold

## What this establishes

A Python evaluation harness for Wiener blind deconvolution that:
1. Loads `kutty-1.jpg` as the ground-truth sharp image
2. Applies a known synthetic motion blur (`len=15, angle=45°`)
3. Runs blind kernel search (circular / Gaussian / motion families)
4. Deblurs with the winning kernel using Wiener deconvolution
5. Measures PSNR vs. the original sharp image

## Baseline Results

| Metric | Value |
|--------|-------|
| PSNR (blind deblur) | **39.57 dB** |
| PSNR (oracle — true kernel) | 13.65 dB |
| PSNR (blurred input) | 42.97 dB |
| Blind kernel found | `motion len=2, angle=5.0°` (incorrect — true: `len=15, angle=45°`) |
| KAPPA | 0.01 |
| Tests passing | 13/13 |

## Key Diagnostic

The oracle PSNR (13.65 dB) is far below the blurred input (42.97 dB).
This means `KAPPA=0.01` causes severe noise amplification when deconvolving
a 15-pixel motion blur — the algorithm is making the image *worse* when given
the exact correct kernel. Primary target for Round 2+: **tune KAPPA**.

## Files Created

- `projects/wiener_deblur_run.yaml` — EPOCH config
- `projects/wiener_deblur/evaluate.py` — evaluation harness
- `projects/wiener_deblur/pyproject.toml` — dependencies
- `projects/wiener_deblur/tests/test_deblur.py` — 13 sanity tests
- `projects/wiener_deblur/run-20260330-1710/baseline_metrics.json` — baseline

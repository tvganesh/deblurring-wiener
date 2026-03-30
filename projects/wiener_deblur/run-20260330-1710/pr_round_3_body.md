# [Round 3] (psnr_oracle: 29.82 → 30.65 dB) Add ringing-suppression blend

## Investigation Findings

At KAPPA=100, the Wiener output has residual ringing artifacts along strong edges.
Three post-processing approaches were evaluated:

| Approach | Best delta | Meets 0.5 dB? |
|----------|-----------|--------------|
| Gaussian blur (sigma=6) | +0.39 dB | ❌ |
| Bilateral filter (d=9) | +0.09 dB | ❌ |
| **Blend: α·Wiener + (1-α)·blurred** | **+0.83 dB (α=0.9)** | ✅ |

The blend approach (`0.9 * Wiener + 0.1 * blurred`) suppresses ringing by anchoring a small fraction of the signal to the blurred input. At α=0.9, the deblurring effect is 90% preserved.

## Change

Added constant `POST_BLEND_ALPHA = 0.9` to `evaluate.py`.
Applied in `main()` after Wiener deconvolution for both blind and oracle outputs:
```python
recovered = POST_BLEND_ALPHA * wiener(blurred, best_kernel, KAPPA) \
            + (1.0 - POST_BLEND_ALPHA) * blurred
```

## Results

| Metric | Round 2 | Round 3 | Delta |
|--------|---------|---------|-------|
| **psnr_oracle** (primary) | 29.82 dB | **30.65 dB** | **+0.83 dB** ✅ |
| psnr (blind) | 37.85 dB | 38.47 dB | +0.62 dB |
| Tests passing | 13/13 | 13/13 | 0 |
| KAPPA | 100.0 | 100.0 | — |
| POST_BLEND_ALPHA | — | 0.9 | new |

## Verdict: ✅ ACCEPT

psnr_oracle improved +0.83 dB (>0.5 dB threshold). Both primary and blind PSNR improved. Tests intact.

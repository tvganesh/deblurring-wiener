"""
Synthetic benchmark for Wiener blind deconvolution.

Evaluates KAPPA and LAMBDA by:
  1. Loading kutty-1.jpg as the ground-truth sharp image
  2. Applying a known motion blur to create a synthetic blurred input
  3. Running blind kernel search (circular / Gaussian / motion families)
  4. Applying Wiener deconvolution with the winning kernel and KAPPA
  5. Computing PSNR of the recovered image vs. the original sharp image
  6. Printing metrics as JSON to stdout

EPOCH tunes KAPPA and POST_BLEND_ALPHA to maximise psnr_oracle.
"""

import json
import math
import pathlib
import sys

import cv2
import numpy as np

# ── Tunable constants (EPOCH will optimise these) ────────────────────────
KAPPA            = 100.0  # Wiener regularisation (noise-to-signal ratio)
LAMBDA           = 0.005  # Sharpness reward weight in kernel-search loss
POST_BLEND_ALPHA = 0.74   # Blend weight: output = alpha*Wiener + (1-alpha)*blurred
                          # Reduces residual ringing by anchoring to the blurred input

# ── Synthetic blur parameters (fixed ground truth) ───────────────────────
SYNTH_LEN   = 15    # motion blur length (pixels)
SYNTH_ANGLE = 45.0  # motion blur angle  (degrees)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
IMG_PATH = ROOT / "kutty-1.jpg"


# ── Kernel builders ──────────────────────────────────────────────────────

def build_circular(radius: int) -> np.ndarray:
    sz = 2 * radius + 1
    K  = np.zeros((sz, sz), dtype=np.float32)
    c  = radius
    for y in range(sz):
        for x in range(sz):
            if math.sqrt((x - c) ** 2 + (y - c) ** 2) <= radius:
                K[y, x] = 1.0
    s = K.sum()
    return K / s if s > 0 else K


def build_gaussian(sigma: float) -> np.ndarray:
    half = int(math.ceil(3.0 * sigma))
    sz   = 2 * half + 1
    K    = np.zeros((sz, sz), dtype=np.float32)
    for y in range(sz):
        for x in range(sz):
            K[y, x] = math.exp(-((x - half) ** 2 + (y - half) ** 2)
                                / (2.0 * sigma ** 2))
    s = K.sum()
    return K / s if s > 0 else K


def build_motion(length: int, angle_deg: float) -> np.ndarray:
    sz    = length if length % 2 == 1 else length + 1
    K     = np.zeros((sz, sz), dtype=np.float32)
    angle = math.radians(angle_deg)
    cx = cy = (sz - 1) / 2.0
    for i in range(length):
        t = (i / (length - 1) - 0.5) if length > 1 else 0.0
        x = int(round(cx + t * (length - 1) * math.cos(angle)))
        y = int(round(cy + t * (length - 1) * math.sin(angle)))
        if 0 <= x < sz and 0 <= y < sz:
            K[y, x] = 1.0
    s = K.sum()
    return K / s if s > 0 else K


# ── Wiener deconvolution (frequency domain) ──────────────────────────────

def wiener(G: np.ndarray, K: np.ndarray, kappa: float) -> np.ndarray:
    """
    Frequency-domain Wiener filter.
    G     : input image, any numeric dtype
    K     : blur kernel, float32 or float64
    kappa : regularisation (noise/signal ratio)
    Returns deblurred image normalised to [0, 1], dtype float64.
    """
    G64  = G.astype(np.float64)
    rows, cols = G64.shape

    dft_r = cv2.getOptimalDFTSize(rows + K.shape[0] - 1)
    dft_c = cv2.getOptimalDFTSize(cols + K.shape[1] - 1)

    G_pad = np.zeros((dft_r, dft_c), dtype=np.float64)
    G_pad[:rows, :cols] = G64
    DFT_G = np.fft.rfft2(G_pad)

    K_pad = np.zeros((dft_r, dft_c), dtype=np.float64)
    K_pad[:K.shape[0], :K.shape[1]] = K.astype(np.float64)
    DFT_K = np.fft.rfft2(K_pad)

    denom = np.abs(DFT_K) ** 2 + kappa
    DFT_F = DFT_G * np.conj(DFT_K) / denom

    F = np.fft.irfft2(DFT_F)[:rows, :cols]

    mn, mx = F.min(), F.max()
    if mx > mn:
        F = (F - mn) / (mx - mn)
    return F


# ── Loss function (mirrors the C++ implementation) ───────────────────────

def compute_loss(G: np.ndarray, K: np.ndarray) -> float:
    F_hat = wiener(G, K, KAPPA)

    G_hat = cv2.filter2D(
        F_hat.astype(np.float32), -1,
        K.astype(np.float32),
        borderType=cv2.BORDER_REFLECT,
    )

    G32 = G.astype(np.float32)
    mn, mx = G32.min(), G32.max()
    if mx > mn:
        G32 = (G32 - mn) / (mx - mn)

    recon     = float(np.mean((G_hat - G32) ** 2))
    lap       = cv2.Laplacian(F_hat.astype(np.float32), cv2.CV_32F)
    sharpness = float(np.var(lap))

    return recon - LAMBDA * sharpness


# ── Per-family blind kernel searches ────────────────────────────────────

def search_circular(G: np.ndarray):
    best_loss, best_r = float("inf"), 1
    for r in range(1, 21):
        loss = compute_loss(G, build_circular(r))
        if loss < best_loss:
            best_loss, best_r = loss, r
    return build_circular(best_r), best_loss, f"circular r={best_r}"


def search_gaussian(G: np.ndarray):
    best_loss, best_sigma = float("inf"), 0.5
    sigma = 0.5
    while sigma <= 15.0 + 1e-9:
        loss = compute_loss(G, build_gaussian(sigma))
        if loss < best_loss:
            best_loss, best_sigma = loss, sigma
        sigma += 0.5
    return build_gaussian(best_sigma), best_loss, f"gaussian sigma={best_sigma:.2f}"


def search_motion(G: np.ndarray):
    best_loss  = float("inf")
    best_len   = 5
    best_angle = 0.0

    # Coarse pass
    for length in range(3, 26, 4):
        for ang in np.arange(0.0, 180.0, 30.0):
            loss = compute_loss(G, build_motion(length, float(ang)))
            if loss < best_loss:
                best_loss, best_len, best_angle = loss, length, float(ang)

    # Fine pass around coarse winner
    for length in range(max(2, best_len - 3), best_len + 4):
        for ang in np.arange(
            max(0.0, best_angle - 25.0),
            min(175.0, best_angle + 25.0) + 1e-9,
            5.0,
        ):
            loss = compute_loss(G, build_motion(length, float(ang)))
            if loss < best_loss:
                best_loss, best_len, best_angle = loss, length, float(ang)

    return (
        build_motion(best_len, best_angle),
        best_loss,
        f"motion len={best_len} angle={best_angle:.1f}",
    )


# ── PSNR ─────────────────────────────────────────────────────────────────

def psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    """PSNR in dB. Both inputs expected in [0, 1] float64."""
    mse = float(np.mean(
        (original.astype(np.float64) - recovered.astype(np.float64)) ** 2
    ))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ── Main evaluation ───────────────────────────────────────────────────────

def main():
    # Load ground-truth sharp image (grayscale, normalised to [0, 1])
    sharp_bgr = cv2.imread(str(IMG_PATH))
    if sharp_bgr is None:
        print(json.dumps({"error": f"Cannot load {IMG_PATH}"}))
        sys.exit(1)
    sharp = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    # Synthesise blurred input with a known motion blur kernel
    true_kernel = build_motion(SYNTH_LEN, SYNTH_ANGLE)
    blurred = cv2.filter2D(
        sharp.astype(np.float32), -1,
        true_kernel,
        borderType=cv2.BORDER_REFLECT,
    ).astype(np.float64)

    # Blind kernel search on the blurred image
    results = [
        search_circular(blurred),
        search_gaussian(blurred),
        search_motion(blurred),
    ]

    # Pick the kernel with the lowest loss
    best_kernel, best_loss, best_desc = min(results, key=lambda r: r[1])

    # Deblur with the winning (blind) kernel, then blend with blurred to reduce ringing
    recovered = (POST_BLEND_ALPHA * wiener(blurred, best_kernel, KAPPA)
                 + (1.0 - POST_BLEND_ALPHA) * blurred)

    # Oracle: deblur with the true kernel (upper-bound reference), same post-processing
    recovered_oracle = (POST_BLEND_ALPHA * wiener(blurred, true_kernel, KAPPA)
                        + (1.0 - POST_BLEND_ALPHA) * blurred)

    metrics = {
        "psnr":         round(psnr(sharp, recovered), 4),
        "psnr_oracle":  round(psnr(sharp, recovered_oracle), 4),
        "psnr_blurred": round(psnr(sharp, blurred), 4),
        "best_kernel":  best_desc,
        "best_loss":    round(best_loss, 6),
        "kappa":        KAPPA,
        "lambda":       LAMBDA,
        "synth_len":    SYNTH_LEN,
        "synth_angle":  SYNTH_ANGLE,
        "all_kernels": [
            {"desc": d, "loss": round(l, 6)}
            for _, l, d in results
        ],
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

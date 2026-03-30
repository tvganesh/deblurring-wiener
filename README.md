# Wiener Deblurring with Blind Kernel Estimation

A C++ implementation of blind image deconvolution using the **Wiener filter**.
Instead of assuming a fixed blur kernel, the program searches over three kernel
families — circular, Gaussian, and motion blur — and picks the one that best
explains the input image, using a self-supervised loss function that requires no
ground-truth sharp reference.

---

## Algorithm Overview

### The Problem

A blurry image can be modelled as:

```
G = H * F + N
```

where `F` is the unknown sharp image, `H` is the blur kernel (point spread
function), `N` is noise, and `*` denotes convolution.  The goal is to recover
`F` from `G` without knowing `H` in advance.

### Wiener Deconvolution (frequency domain)

```
F_hat(u,v) = [ H*(u,v) / ( |H(u,v)|² + κ ) ] · G(u,v)
```

- `H*` — complex conjugate of the kernel's DFT  
- `|H|²` — power spectrum of the kernel  
- `κ`  — regularisation constant (controls noise vs sharpness trade-off)

### Blind Kernel Estimation

Because the true kernel `H` is unknown, we iterate over candidate kernels and
minimise a **self-supervised composite loss**:

```
L(K) = MSE( K * F_hat(K), G )  −  λ · Var( Laplacian( F_hat ) )
```

| Term | Role |
|---|---|
| `MSE(K * F_hat, G)` | Reconstruction consistency — re-blurring the recovered image with `K` should reproduce `G` |
| `Var(Laplacian(F_hat))` | Sharpness reward — sharper recovered images score higher (subtracted to minimise) |
| `λ = 0.005` | Weight balancing the two terms |

No ground-truth sharp image is needed.

### Three Kernel Families

| Family | Parameters | Search strategy |
|---|---|---|
| **Circular** | radius `r` ∈ [1, 20] px | Full integer grid (20 evaluations) |
| **Gaussian** | sigma `σ` ∈ [0.5, 15.0] | Grid in 0.5 steps (29 evaluations) |
| **Motion blur** | length `l` ∈ [3, 25] px, angle `θ` ∈ [0°, 175°] | Coarse grid → fine refinement (~77 evaluations) |

The kernel with the lowest loss across all three families is selected as the
winner.

### Colour Restoration

Kernel search and Wiener deconvolution run on the **luminance channel only**
(YCrCb colour space).  The chrominance channels (Cr, Cb) are left untouched and
merged back before display, avoiding colour fringing that would result from
deblurring R, G, B independently.

---

## Requirements

- **C++14** or later  
- **OpenCV 4.x**  
- **CMake 3.10+**

### Install dependencies (macOS)

```bash
brew install cmake opencv
```

---

## Build

```bash
git clone https://github.com/<your-username>/deblurring-wiener.git
cd deblurring-wiener
mkdir build && cd build
cmake ..
make
```

---

## Usage

```bash
./deblurring <image_path>
```

**Example:**

```bash
./deblurring ~/Desktop/photo.jpg
```

The program prints the search progress and loss values for each kernel candidate,
then opens a single resizable montage window:

```
Row 1 (grayscale): Input | Circular * | Gaussian | Motion
Row 2 (colour):    Input | Circular * | Gaussian | Motion
```

`*` marks the winning kernel.  Press any key to close.

### Console output example

```
--- Circular kernel search ---
  radius=1    loss=+0.003421
  radius=2    loss=+0.002108
  ...
  => Best: Circular r=7  loss=-0.001234

--- Gaussian kernel search ---
  sigma=0.50  loss=+0.003190
  ...
  => Best: Gaussian sigma=4.50  loss=-0.001891

--- Motion blur kernel search ---
  [Coarse pass]
  len=3   angle=0.0    loss=+0.002341
  ...
  [Fine pass around len=11, angle=45.0]
  ...
  => Best: Motion len=11 angle=45.0  loss=-0.002104

========================================
RESULTS SUMMARY:
  Circular r=7                    loss=-0.001234
  Gaussian sigma=4.50             loss=-0.001891
  Motion len=11 angle=45.0        loss=-0.002104  <-- WINNER
========================================
```

---

## Tunable Constants

Edit these `#define` values at the top of `deblurring.cpp` to adjust behaviour:

| Constant | Default | Effect |
|---|---|---|
| `KAPPA` | `0.01` | Wiener regularisation — lower = sharper but noisier |
| `LAMBDA` | `0.005` | Sharpness reward weight in loss — higher biases toward sharper results |

---

## Project Structure

```
deblurring-wiener/
├── deblurring.cpp      # Main source file
├── CMakeLists.txt      # Build configuration
└── README.md           # This file
```

---

## Limitations

- Assumes the blur is **spatially uniform** across the entire image.  
- The three kernel families cover common real-world blurs; unusual or mixed
  blur types may not be well represented.  
- For state-of-the-art results on arbitrary real-world photos, deep learning
  models (e.g. NAFNet, MPRNet, DeblurGAN) trained on large datasets are more
  robust, but require a pre-trained model or training pipeline.

---

## Authors

- Tinniam V Ganesh  
- Egli Simon

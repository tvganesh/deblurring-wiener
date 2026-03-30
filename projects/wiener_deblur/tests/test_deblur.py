"""Sanity tests for the Wiener deblurring evaluation harness."""

import importlib.util
import math
import pathlib

import numpy as np
import pytest

# ── Import evaluate.py ────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "evaluate",
    pathlib.Path(__file__).resolve().parent.parent / "evaluate.py",
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


class TestKernelBuilders:
    def test_circular_sums_to_one(self):
        K = evaluate.build_circular(3)
        assert abs(K.sum() - 1.0) < 1e-5

    def test_gaussian_sums_to_one(self):
        K = evaluate.build_gaussian(2.0)
        assert abs(K.sum() - 1.0) < 1e-5

    def test_motion_sums_to_one(self):
        K = evaluate.build_motion(10, 45.0)
        assert abs(K.sum() - 1.0) < 1e-5

    def test_circular_shape(self):
        K = evaluate.build_circular(5)
        assert K.shape == (11, 11)

    def test_motion_odd_size(self):
        K = evaluate.build_motion(10, 0.0)
        assert K.shape[0] % 2 == 1 and K.shape[1] % 2 == 1

    def test_gaussian_shape_grows_with_sigma(self):
        K_small = evaluate.build_gaussian(1.0)
        K_large = evaluate.build_gaussian(3.0)
        assert K_large.shape[0] > K_small.shape[0]


class TestWiener:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        img = rng.random((64, 64))
        K   = evaluate.build_gaussian(2.0)
        out = evaluate.wiener(img, K, kappa=0.01)
        assert out.min() >= -1e-9
        assert out.max() <= 1.0 + 1e-9

    def test_output_shape_preserved(self):
        img = np.random.default_rng(0).random((80, 60))
        K   = evaluate.build_circular(2)
        out = evaluate.wiener(img, K, kappa=0.01)
        assert out.shape == img.shape

    def test_kappa_changes_output(self):
        """Different KAPPA values must produce different deblurred images."""
        rng = np.random.default_rng(7)
        img = rng.random((64, 64))
        K   = evaluate.build_motion(9, 30.0)
        out_low  = evaluate.wiener(img, K, kappa=0.0001)
        out_high = evaluate.wiener(img, K, kappa=0.5)
        assert not np.allclose(out_low, out_high)


class TestPSNR:
    def test_identical_images_returns_inf(self):
        img = np.ones((32, 32)) * 0.5
        assert evaluate.psnr(img, img) == float("inf")

    def test_zero_vs_one_returns_zero_db(self):
        img_a = np.zeros((32, 32))
        img_b = np.ones((32, 32))
        assert abs(evaluate.psnr(img_a, img_b) - 0.0) < 1e-9

    def test_psnr_increases_with_similarity(self):
        rng  = np.random.default_rng(1)
        orig = rng.random((64, 64))
        noisy_heavy = np.clip(orig + rng.normal(0, 0.2, orig.shape), 0, 1)
        noisy_light = np.clip(orig + rng.normal(0, 0.05, orig.shape), 0, 1)
        assert evaluate.psnr(orig, noisy_light) > evaluate.psnr(orig, noisy_heavy)


class TestImagePath:
    def test_image_exists(self):
        assert evaluate.IMG_PATH.exists(), f"Image not found: {evaluate.IMG_PATH}"

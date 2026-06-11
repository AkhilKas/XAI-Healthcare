import base64

import numpy as np
import torch

from backend.rnn_inference import (
    compute_compensation,
    compute_motion_metrics,
    compute_movement_quality,
    compute_rom,
    generate_plot,
    get_fallback_response,
    to_json_safe,
)


# ---------- compute_rom ----------
def test_compute_rom_static_signal_is_zero():
    seg = np.ones((100, 3)) * 0.5
    assert compute_rom(seg) == 0.0


def test_compute_rom_moving_signal_is_positive():
    t = np.linspace(0, 1, 100)
    seg = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.zeros_like(t)], axis=1)
    assert compute_rom(seg) > 0


def test_compute_rom_accepts_torch_tensor():
    seg = torch.randn(50, 3)
    assert compute_rom(seg) >= 0


# ---------- compute_movement_quality ----------
def test_compute_movement_quality_returns_score_in_range():
    np.random.seed(0)
    seg = np.cumsum(np.random.randn(120, 3), axis=0) * 0.01
    score = compute_movement_quality(seg)
    assert 0 <= score <= 100


def test_compute_movement_quality_static_returns_default():
    seg = np.zeros((120, 3))
    score = compute_movement_quality(seg)
    assert score == 50.0


# ---------- compute_compensation ----------
def test_compute_compensation_low_when_head_still():
    head = np.zeros((100, 3))
    left = np.cumsum(np.random.randn(100, 3) * 0.01, axis=0)
    right = np.cumsum(np.random.randn(100, 3) * 0.01, axis=0)
    comp = compute_compensation(head, left, right)
    assert 0 <= comp <= 100


def test_compute_compensation_high_when_head_dominates():
    head = np.cumsum(np.random.randn(100, 3) * 0.5, axis=0)
    left = np.zeros((100, 3))
    right = np.zeros((100, 3))
    comp = compute_compensation(head, left, right)
    # Comp should be clipped to 100 since head_disp >> wrist_disp
    assert comp == 100.0


# ---------- compute_motion_metrics ----------
def test_compute_motion_metrics_returns_full_structure():
    np.random.seed(42)
    sample = np.cumsum(np.random.randn(120, 18) * 0.01, axis=0)
    importance = np.random.dirichlet(np.ones(18))

    result = compute_motion_metrics(sample, importance)

    assert set(result.keys()) == {
        "rom",
        "aggregated_rom",
        "movement_quality",
        "aggregated_mq",
        "compensation",
        "aggregated_score",
        "injured_region",
    }
    assert set(result["rom"].keys()) == {"head", "left", "right"}
    assert set(result["movement_quality"].keys()) == {"head", "left", "right"}
    assert set(result["injured_region"].keys()) == {"head", "left", "right"}

    for key in ["aggregated_rom", "aggregated_mq", "compensation", "aggregated_score"]:
        assert 0 <= result[key] <= 100


# ---------- to_json_safe ----------
def test_to_json_safe_converts_numpy_floats():
    assert to_json_safe(np.float32(1.5)) == 1.5
    assert isinstance(to_json_safe(np.float64(2.5)), float)


def test_to_json_safe_converts_numpy_ints():
    assert to_json_safe(np.int32(7)) == 7
    assert isinstance(to_json_safe(np.int64(7)), int)


def test_to_json_safe_converts_arrays():
    assert to_json_safe(np.array([1, 2, 3])) == [1, 2, 3]


def test_to_json_safe_recurses_nested():
    payload = {
        "a": np.int32(1),
        "b": [np.float64(2.0), {"c": np.array([3, 4])}],
    }
    assert to_json_safe(payload) == {"a": 1, "b": [2.0, {"c": [3, 4]}]}


def test_to_json_safe_passes_through_native_types():
    assert to_json_safe("hello") == "hello"
    assert to_json_safe(42) == 42
    assert to_json_safe(None) is None


# ---------- get_fallback_response ----------
def test_get_fallback_response_has_required_keys():
    summary_inputs = {
        "prediction": 1,
        "metrics": {
            "rom": {"head": 60.0, "left": 40.0, "right": 80.0},
            "movement_quality": {"head": 70.0, "left": 50.0, "right": 60.0},
            "compensation": 45.0,
            "aggregated_rom": 60.0,
            "aggregated_mq": 60.0,
            "injured_region": {"head": 50.0, "left": 100.0, "right": 30.0},
        },
    }
    response = get_fallback_response(summary_inputs)

    required = {
        "one_sentence_summary",
        "key_findings",
        "counterfactual_analysis",
        "recommendations",
        "detailed_analysis",
    }
    assert required.issubset(response.keys())

    findings = response["key_findings"]
    assert any(k.startswith("Error:") for k in findings)
    assert any(k.startswith("Warning:") for k in findings)
    assert any(k.startswith("Success:") for k in findings)


# ---------- generate_plot ----------
def test_generate_plot_returns_base64_png():
    data = np.random.randn(30, 18)
    b64 = generate_plot(data)
    assert isinstance(b64, str)
    decoded = base64.b64decode(b64)
    # PNG magic number
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

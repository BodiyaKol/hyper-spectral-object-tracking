import pytest

from hsi_tracking.evaluation import (
    calculate_iou,
    calculate_center_distance,
    calculate_dp20,
    calculate_auc,
    calculate_precision_curve,
)


def test_identical_boxes():
    gt = [100, 100, 50, 50]
    pred = [100, 100, 50, 50]

    assert calculate_iou(gt, pred) == 1.0


def test_non_overlapping_boxes():
    gt = [0, 0, 50, 50]
    pred = [100, 100, 50, 50]

    assert calculate_iou(gt, pred) == 0.0


def test_partial_overlap():
    gt = [100, 100, 50, 50]
    pred = [110, 110, 50, 50]

    expected = 1600 / 3400

    assert abs(calculate_iou(gt, pred) - expected) < 1e-6


def test_center_distance():
    gt = [0, 0, 20, 20]
    pred = [3, 4, 20, 20]

    assert calculate_center_distance(gt, pred) == 5.0


def test_dp20():
    gt = [
        [0, 0, 20, 20],
        [0, 0, 20, 20],
        [0, 0, 20, 20],
    ]

    pred = [
        [0, 0, 20, 20],    # distance = 0
        [10, 0, 20, 20],   # distance = 10
        [30, 0, 20, 20],   # distance = 30
    ]

    assert calculate_dp20(gt, pred) == pytest.approx(2 / 3)


def test_auc_identical_boxes():
    gt = [[0, 0, 50, 50]]
    pred = [[0, 0, 50, 50]]

    assert calculate_auc(gt, pred) == pytest.approx(1.0)


def test_auc_non_overlapping_boxes():
    gt = [[0, 0, 50, 50]]
    pred = [[100, 100, 50, 50]]

    assert calculate_auc(gt, pred) == pytest.approx(0.005)

def test_precision_curve():
    gt = [
        [0, 0, 20, 20],
        [0, 0, 20, 20],
        [0, 0, 20, 20],
    ]

    pred = [
        [0, 0, 20, 20],
        [10, 0, 20, 20],
        [30, 0, 20, 20],
    ]

    thresholds, precision = calculate_precision_curve(gt, pred)

    assert precision[20] == pytest.approx(2 / 3)
    assert precision[30] == pytest.approx(1.0)


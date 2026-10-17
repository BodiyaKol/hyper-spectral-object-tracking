"""One-pass evaluation and HOT tracking metrics."""

def calculate_iou(ground_truth, prediction):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        ground_truth: [x, y, width, height]
        prediction: [x, y, width, height]

    Returns:
        IoU score between 0 and 1.
    """

    x1, y1, w1, h1 = ground_truth
    x2, y2, w2, h2 = prediction

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    # Find the coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    # Calculate union area
    area_gt = w1 * h1
    area_pred = w2 * h2
    union = area_gt + area_pred - intersection

    return intersection / union


import math


def calculate_center_distance(ground_truth, prediction):
    """Calculate the Euclidean distance between bounding box centers."""
    x1, y1, w1, h1 = ground_truth
    x2, y2, w2, h2 = prediction

    center_gt_x = x1 + w1 / 2
    center_gt_y = y1 + h1 / 2

    center_pred_x = x2 + w2 / 2
    center_pred_y = y2 + h2 / 2

    return math.hypot(
        center_gt_x - center_pred_x,
        center_gt_y - center_pred_y,
    )


def calculate_dp20(ground_truth_boxes, predicted_boxes):
    """Calculate Distance Precision at a threshold of 20 pixels."""
    if len(ground_truth_boxes) != len(predicted_boxes):
        raise ValueError("Ground truth and predictions must have equal length.")

    if len(ground_truth_boxes) == 0:
        raise ValueError("Bounding box sequences cannot be empty.")

    correct_frames = 0

    for gt, pred in zip(ground_truth_boxes, predicted_boxes):
        distance = calculate_center_distance(gt, pred)

        if distance <= 20:
            correct_frames += 1

    return correct_frames / len(ground_truth_boxes)


import numpy as np


def calculate_success_curve(
    ground_truth_boxes,
    predicted_boxes,
    thresholds=None,
):
    """Calculate tracking success rate at different IoU thresholds."""
    if len(ground_truth_boxes) != len(predicted_boxes):
        raise ValueError("Ground truth and predictions must have equal length.")

    if len(ground_truth_boxes) == 0:
        raise ValueError("Bounding box sequences cannot be empty.")

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    ious = np.array([
        calculate_iou(gt, pred)
        for gt, pred in zip(ground_truth_boxes, predicted_boxes)
    ])

    success_rates = np.array([
        np.mean(ious >= threshold)
        for threshold in thresholds
    ])

    return thresholds, success_rates


def calculate_auc(ground_truth_boxes, predicted_boxes):
    """Calculate Area Under the Success Curve."""
    thresholds, success_rates = calculate_success_curve(
        ground_truth_boxes,
        predicted_boxes,
    )

    return float(np.trapezoid(success_rates, thresholds))


def calculate_precision_curve(
    ground_truth_boxes,
    predicted_boxes,
    thresholds=None,
):
    """Calculate tracking precision at different distance thresholds."""
    if len(ground_truth_boxes) != len(predicted_boxes):
        raise ValueError("Ground truth and predictions must have equal length.")

    if len(ground_truth_boxes) == 0:
        raise ValueError("Bounding box sequences cannot be empty.")

    if thresholds is None:
        thresholds = np.arange(0, 51, 1)

    distances = np.array([
        calculate_center_distance(gt, pred)
        for gt, pred in zip(ground_truth_boxes, predicted_boxes)
    ])

    precision_rates = np.array([
        np.mean(distances <= threshold)
        for threshold in thresholds
    ])

    return thresholds, precision_rates



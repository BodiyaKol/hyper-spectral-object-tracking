
from hsi_tracking.visualization import plot_tracking_metrics

# Example ground truth bounding boxes
ground_truth = [
    [100, 100, 50, 50],
    [110, 100, 50, 50],
    [120, 100, 50, 50],
    [130, 100, 50, 50],
    [140, 100, 50, 50],
]

# Example model predictions
predictions = [
    [100, 100, 50, 50],
    [115, 100, 50, 50],
    [130, 100, 50, 50],
    [145, 100, 50, 50],
    [170, 100, 50, 50],
]

success_path, precision_path = plot_tracking_metrics(
    ground_truth_boxes=ground_truth,
    predicted_boxes=predictions,
    model_name="Demo Model",
    output_dir="results/plots",
)

print("Success Plot:", success_path)
print("Precision Plot:", precision_path)

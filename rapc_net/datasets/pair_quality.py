"""Conservative checks for gross input/GT separation, not completion quality."""
import math
import statistics


def pair_alignment_report(partial, complete):
    if len(partial) == 0 or len(complete) == 0:
        raise ValueError("Empty input or GT")
    lower = [min(float(p[j]) for p in complete) for j in range(3)]
    upper = [max(float(p[j]) for p in complete) for j in range(3)]
    diameter = math.sqrt(sum((b - a) ** 2 for a, b in zip(lower, upper)))
    if not math.isfinite(diameter) or diameter <= 1e-12:
        raise ValueError("Invalid GT bounding-box diameter")
    distances = []
    for point in partial:
        if not all(math.isfinite(float(x)) for x in point):
            raise ValueError("Nonfinite input point")
        distances.append(math.sqrt(sum(max(lower[j] - float(point[j]), 0.0, float(point[j]) - upper[j]) ** 2 for j in range(3))))
    ratio = statistics.median(distances) / diameter
    return {"gt_bbox_diagonal": diameter, "median_input_to_gt_bbox": statistics.median(distances),
            "separation_ratio": ratio, "grossly_disjoint": ratio > 1.0}

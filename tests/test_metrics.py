import numpy as np

from rapc_net.evaluation.metrics import evaluate_completion


def test_identical_cloud_has_zero_chamfer():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    metrics = evaluate_completion(points, points, points)
    assert metrics.cd_l1 == 0.0
    assert metrics.cd_l2 == 0.0


def test_shifted_cloud_is_worse_than_identical():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    same = evaluate_completion(points, points, points)
    shifted = evaluate_completion(points + 0.2, points, points)
    assert shifted.cd_l1 > same.cd_l1

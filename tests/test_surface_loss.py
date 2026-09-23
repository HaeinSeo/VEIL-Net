import torch

from rapc_net.losses.completion import CompletionLoss, surface_l2, chamfer_l1, soft_fscore_loss, local_repulsion


def test_repulsion_separates_close_points_without_self_neighbors():
    points = torch.tensor([[[0., 0., 0.], [0.01, 0., 0.]]], requires_grad=True)
    loss = local_repulsion(points)
    gradient = torch.autograd.grad(loss, points)[0]
    assert gradient[0, 0, 0] > 0 and gradient[0, 1, 0] < 0
    assert local_repulsion(points - 0.0001 * gradient) < loss
    assert local_repulsion(points.detach() * 10) == 0


def test_repulsion_duplicates_and_singleton_are_finite():
    for count in (1, 8):
        points = torch.zeros(2, count, 3, requires_grad=True)
        loss = local_repulsion(points, max_points=4)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(points.grad).all()


def test_repulsion_weight_and_raw_logging():
    points = torch.tensor([[[0., 0., 0.], [0.01, 0., 0.]]])
    criterion = CompletionLoss(lambda_cd=0, lambda_uniform=0, lambda_repulsion=0.02)
    loss, parts = criterion({"completed_points": points}, {"complete": points})
    torch.testing.assert_close(loss, local_repulsion(points) * 0.02)
    assert abs(parts["raw_repulsion"] - local_repulsion(points).item()) < 1e-6


def test_surface_loss_matches_dense_values_and_gradients():
    torch.manual_seed(4)
    pred = torch.randn(2, 9, 3, requires_grad=True)
    target = torch.randn(2, 13, 3)
    actual = surface_l2(pred, target, chunk=4)
    expected = torch.cdist(pred, target).min(dim=2).values.square().mean()
    actual_grad = torch.autograd.grad(actual, pred, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_surface_gradient_moves_outlier_toward_surface():
    target = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    pred = torch.tensor([[[0., 0., 2.], [1., 0., 0.]]], requires_grad=True)
    before = surface_l2(pred, target)
    gradient = torch.autograd.grad(before, pred)[0]
    assert surface_l2(pred - 0.1 * gradient, target) < before
    assert surface_l2(target, target) == 0


def test_raw_loss_is_independent_of_weight():
    pred = torch.tensor([[[0., 0., 2.]]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss_fn = CompletionLoss(lambda_cd=0, lambda_surface=0.5, lambda_uniform=0)
    total, parts = loss_fn({"completed_points": pred}, {"complete": target})
    assert parts["raw_surface"] == 4.0
    assert parts["surface"] == 2.0
    assert total.item() == 2.0


def test_chunked_chamfer_matches_dense_gradients():
    torch.manual_seed(9)
    pred = torch.randn(2, 17, 3, requires_grad=True)
    target = torch.randn(2, 23, 3)
    actual = chamfer_l1(pred, target)
    matrix = torch.cdist(pred, target)
    expected = matrix.min(1).values.mean() + matrix.min(2).values.mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])


def test_soft_fscore_improves_with_surface_approach():
    target = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    pred = (target + torch.tensor([0., 0., 0.25])).requires_grad_()
    loss = soft_fscore_loss(pred, target)
    grad = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(grad).all()
    assert soft_fscore_loss(pred - grad * 0.05, target) < loss
    assert soft_fscore_loss(target, target) < loss


def test_missing_supervision_targets_final_output_when_enabled():
    final = torch.tensor([[[0., 0., 0.]]], requires_grad=True)
    generated = torch.tensor([[[4., 0., 0.]]], requires_grad=True)
    target = torch.tensor([[[2., 0., 0.]]])
    loss_fn = CompletionLoss(lambda_cd=0, lambda_missing=1, lambda_preserve=0,
        lambda_uniform=0, missing_on_completed=True)
    loss, _ = loss_fn({"completed_points": final, "generated_points": generated}, {"complete": target, "partial": final.detach()})
    loss.backward()
    assert final.grad[0, 0, 0] < 0
    assert generated.grad is None


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_surface_loss_matches_dense_values_and_gradients()
    test_surface_gradient_moves_outlier_toward_surface()
    test_raw_loss_is_independent_of_weight()
    print("3 CPU surface loss tests passed")

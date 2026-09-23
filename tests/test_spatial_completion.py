import torch
from rapc_net.models.rapc_net import RAPCNet, RAPCNetConfig
from rapc_net.models.spatial_completion import SpatialCompletion, farthest_indices


def test_input_frame_similarity_and_risk_folding_gradients():
    torch.manual_seed(13)
    config = RAPCNetConfig(hidden_dim=32, output_points=48, use_spatial_transformer=True,
        two_stage_folding=True, input_frame_normalization=True, bbox_margin=0,
        uncertainty_moves_points=False)
    model = RAPCNet(config).eval()
    partial = torch.randn(2, 16, 3)
    output = model(partial)
    shift = torch.tensor([2., -3., 1.])
    transformed = model(partial * 3 + shift)
    for key in ('completed_points', 'generated_points', 'coarse_points'):
        torch.testing.assert_close(transformed[key], output[key] * 3 + shift, atol=1e-4, rtol=1e-4)
    output['generated_points'].square().mean().backward()
    for name in ('risk', 'router'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters())
    assert model.spatial_completion.fold_second[-1].weight.grad.abs().sum() > 0
    restored = RAPCNet(config)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_spatial_shapes_gradients_and_checkpoint():
    torch.manual_seed(7)
    config = RAPCNetConfig(hidden_dim=32, output_points=35, use_spatial_transformer=True,
                          bbox_margin=0, uncertainty_moves_points=False)
    model = RAPCNet(config)
    partial = torch.randn(2, 12, 3, requires_grad=True)
    output = model(partial)
    assert output['completed_points'].shape == (2, 35, 3)
    assert output['coarse_points'].shape == (2, 35, 3)
    output['completed_points'].square().mean().backward()
    assert torch.isfinite(partial.grad).all()
    for name in ('queries.weight', 'local.0.weight', 'centers.weight', 'fold.0.weight'):
        grad = dict(model.spatial_completion.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    restored = RAPCNet(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(partial)['completed_points'], output['completed_points'])


def test_spatial_patch_grid_and_input_dependence():
    torch.manual_seed(11)
    module = SpatialCompletion(32, 37, tokens=4, patches=8)
    partial = torch.randn(1, 9, 3)
    features = torch.randn(1, 9, 32)
    context = torch.randn(1, 32)
    risk = torch.rand(1, 9)
    points, _ = module(partial, features, context, risk)
    assert points.shape == (1, 37, 3)
    changed, _ = module(partial, features + 1, context, risk)
    assert not torch.allclose(points, changed)
    assert farthest_indices(partial, 4).unique().numel() == 4
    assert (points[:, 0] - points[:, 8]).abs().sum() > 0

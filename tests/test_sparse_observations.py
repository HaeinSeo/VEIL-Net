from dataclasses import replace

import torch

from veil_net.models.completion import CompletionConfig, CompletionNetwork
from veil_net.models.input_geometry import screen_and_resample, unique_point_indices
from veil_net.models.spatial_completion import SpatialCompletion


def test_sparse_screen_preserves_source_points_and_dense_behavior():
    sparse = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.1, 0.0, 0.0]])
    points = sparse.repeat_interleave(torch.tensor([20, 20, 20, 4]), dim=0)[None]
    old, _, old_mask = screen_and_resample(points)
    kept, source, mask = screen_and_resample(points, sparse_input_guard=True)
    assert len(unique_point_indices(old[0])) == 3 and not old_mask.all()
    assert torch.equal(kept, points) and mask.all()
    assert torch.equal(source[0], torch.arange(64))
    torch.manual_seed(103)
    dense = torch.randn(2, 64, 3)
    dense[:, -1] += 100
    for before, after in zip(screen_and_resample(dense), screen_and_resample(dense, True)):
        assert torch.equal(before, after)


def test_sparse_tokens_ignore_duplicate_multiplicity():
    torch.manual_seed(104)
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    features = torch.randn(1, 4, 32)
    risk = torch.rand(1, 4)
    context = torch.randn(1, 32)
    module = SpatialCompletion(
        32,
        128,
        tokens=8,
        patches=8,
        two_stage=True,
        geometry_queries=True,
        sparse_unique_tokens=True,
    ).eval()
    indices = torch.tensor([0] * 10 + [1] * 2 + [2] * 7 + [3] * 4)
    once = module(xyz, features, context, risk)
    repeated = module(xyz[:, indices], features[:, indices], context, risk[:, indices])
    for expected, actual in zip(once, repeated):
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)


def test_sparse_checkpoint_compatibility_gradients_and_no_oracle():
    torch.manual_seed(105)
    config = CompletionConfig(
        hidden_dim=32,
        output_points=128,
        spatial_tokens=8,
        spatial_patches=8,
        use_spatial_transformer=True,
        geometry_queries=True,
        two_stage_folding=True,
        robust_input_geometry=True,
        input_frame_normalization=True,
        bbox_margin=0,
        uncertainty_moves_points=False,
    )
    base = CompletionNetwork(config).eval()
    revised = CompletionNetwork(
        replace(config, sparse_input_guard=True, sparse_unique_tokens=True)
    ).eval()
    revised.load_state_dict(base.state_dict(), strict=True)
    partial = torch.randn(1, 4, 3).repeat(1, 16, 1)
    result = revised(partial)
    assert result["completed_points"].shape == (1, 128, 3)
    assert result["input_inlier_mask"].all()
    with torch.no_grad():
        torch.testing.assert_close(
            result["completed_points"], revised(partial, torch.ones(1, 64))["completed_points"]
        )
    result["generated_points"].square().mean().backward()
    for part in (revised.encoder, revised.risk, revised.router, revised.spatial_completion):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    dense = torch.randn(2, 64, 3)
    with torch.no_grad():
        torch.testing.assert_close(
            base(dense)["completed_points"], revised(dense)["completed_points"], atol=0, rtol=0
        )
        one = revised(torch.zeros(1, 64, 3))["completed_points"]
        assert torch.isfinite(one).all()

import torch

from veil_net.losses.completion import CompletionLoss, missing_region_l1
from veil_net.models.completion import CompletionConfig, CompletionNetwork
from veil_net.models.input_geometry import screen_and_resample
from veil_net.models.spatial_completion import SpatialCompletion


def test_screening_tracks_sources_and_rejects_distant_points():
    torch.manual_seed(23)
    points = torch.randn(2, 32, 3) * 0.01
    points[:, -5:] += 2
    cleaned, indices, mask = screen_and_resample(points)
    assert not mask[:, -5:].any()
    assert torch.equal(cleaned, points.gather(1, indices[..., None].expand(-1, -1, 3)))
    changed, idx2, mask2 = screen_and_resample(points * 5 + 3)
    assert torch.equal(mask, mask2) and torch.equal(indices, idx2)
    torch.testing.assert_close(changed, cleaned * 5 + 3)
    same, _, mask = screen_and_resample(torch.zeros(1, 8, 3))
    assert mask.all() and torch.isfinite(same).all()


def test_relative_geometry_loss_is_scale_invariant_and_keeps_output():
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.1]]])
    prediction = (target + torch.tensor([0.0, 0.01, 0.0])).requires_grad_()
    partial = target[:, :1]
    criterion = CompletionLoss(
        lambda_cd=1,
        lambda_missing=1,
        lambda_surface=0.5,
        lambda_preserve=0,
        lambda_uniform=0,
        relative_geometry_loss=True,
        missing_on_completed=True,
        missing_threshold=0.03,
    )
    original = prediction.detach().clone()
    output = {"completed_points": prediction}
    before, _ = criterion(output, {"complete": target, "partial": partial})
    after, _ = criterion(
        {"completed_points": prediction * 7 + 2},
        {"complete": target * 7 + 2, "partial": partial * 7 + 2},
    )
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    assert torch.equal(output["completed_points"].detach(), original)
    before.backward()
    assert torch.isfinite(prediction.grad).all()


def test_missing_loss_weights_samples_not_missing_counts():
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    pred = torch.zeros(2, 1, 3)
    partial = torch.zeros_like(pred)
    actual = missing_region_l1(pred, target, partial, threshold=0.1, per_sample=True)
    torch.testing.assert_close(actual, torch.tensor(1.25))


def test_patch_grid_has_two_dimensional_extent_without_fold_offsets():
    torch.manual_seed(24)
    model = SpatialCompletion(32, 32, tokens=4, patches=4, two_stage=True, grid_residual=True)
    for module in (model.fold, model.fold_second):
        for parameter in module.parameters():
            torch.nn.init.zeros_(parameter)
    partial = torch.randn(1, 12, 3)
    points, _ = model(partial, torch.randn(1, 12, 32), torch.randn(1, 32), torch.rand(1, 12))
    patches = points.reshape(1, 8, 4, 3).transpose(1, 2)
    singular = torch.linalg.svdvals(patches - patches.mean(2, keepdim=True))
    assert (singular[..., 1] > 0.001).all()


def test_geometry_fix_end_to_end_risk_gradients_and_strict_reload():
    torch.manual_seed(25)
    config = CompletionConfig(
        hidden_dim=32,
        output_points=1024,
        use_spatial_transformer=True,
        input_frame_normalization=True,
        two_stage_folding=True,
        robust_input_geometry=True,
        patch_grid_residual=True,
        uncertainty_moves_points=False,
        bbox_margin=0,
    )
    model = CompletionNetwork(config)
    partial = torch.randn(2, 32, 3) * 0.02
    partial[:, -5:] += 2
    target = torch.randn(2, 48, 3) * 0.03
    outputs = model(partial)
    assert outputs["completed_points"].shape == (2, 1024, 3)
    assert not outputs["input_inlier_mask"][:, -5:].any()
    assert (outputs["input_source_indices"] < 27).all()
    criterion = CompletionLoss(
        lambda_cd=1,
        lambda_coarse=0.5,
        lambda_missing=1,
        lambda_preserve=0.1,
        lambda_uniform=0,
        lambda_patch_repulsion=0.02,
        missing_on_completed=True,
        relative_geometry_loss=True,
    )
    risk_target = torch.linspace(0, 1, 32)[None].expand(2, -1)
    loss, _ = criterion(
        outputs, {"complete": target, "partial": partial, "risk_target": risk_target}
    )
    loss.backward()
    for part in (model.risk, model.router, model.spatial_completion.patch_basis):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    clone = CompletionNetwork(config)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(partial)["completed_points"], outputs["completed_points"])

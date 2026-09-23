from dataclasses import replace

import torch

from veil_net.models import CompletionConfig, CompletionNetwork


def test_disabled_bbox_allows_missing_geometry_and_gradients():
    class ConstantDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.points = torch.nn.Parameter(torch.ones(1, 8, 3))

        def forward(self, feature, risk):
            return self.points.expand(feature.shape[0], -1, -1)

    model = CompletionNetwork(
        CompletionConfig(
            output_points=8, hidden_dim=32, use_preservation=False, use_uncertainty=False
        )
    )
    model.decoder = ConstantDecoder()
    partial = torch.zeros(2, 8, 3)
    clipped = model(partial)["completed_points"]
    assert torch.allclose(clipped, torch.full_like(clipped, 0.2))
    clipped.sum().backward()
    assert torch.count_nonzero(model.decoder.points.grad) == 0
    model.zero_grad(set_to_none=True)
    model.config = replace(model.config, bbox_margin=0.0)
    unclipped = model(partial)["completed_points"]
    assert torch.allclose(unclipped, torch.ones_like(unclipped))
    unclipped.sum().backward()
    assert torch.all(model.decoder.points.grad > 0)


def test_veil_net_forward_shapes():
    model = CompletionNetwork(CompletionConfig(input_points=32, output_points=64, hidden_dim=32))
    outputs = model(torch.randn(2, 32, 3))
    assert outputs["completed_points"].shape == (2, 64, 3)
    assert outputs["generated_points"].shape == (2, 64, 3)
    assert outputs["point_risk"].shape == (2, 32)
    assert outputs["predicted_risk"].shape == (2, 32)
    assert outputs["uncertainty"].shape == (2, 64)
    assert outputs["point_uncertainty"].shape == (2, 64)
    assert outputs["global_confidence"].shape == (2,)
    assert outputs["route_weights"].shape == (2, 4)


def test_uncertainty_can_report_without_moving_points():
    model = CompletionNetwork(
        CompletionConfig(
            output_points=16,
            hidden_dim=32,
            use_preservation=False,
            bbox_margin=0,
            uncertainty_moves_points=False,
        )
    )
    outputs = model(torch.randn(2, 8, 3))
    torch.testing.assert_close(outputs["completed_points"], outputs["generated_points"])
    assert torch.isfinite(outputs["uncertainty"]).all()
    outputs["completed_points"].sum().backward()
    assert model.decoder.seed.grad is not None


def test_oracle_risk_forward_uses_target_risk():
    model = CompletionNetwork(
        CompletionConfig(input_points=32, output_points=64, hidden_dim=32, oracle_risk=True)
    )
    partial = torch.randn(2, 32, 3)
    risk_target = torch.rand(2, 32)
    outputs = model(partial, risk_target)
    assert torch.allclose(outputs["point_risk"], risk_target)


def test_surface_patch_shapes_gradients_and_warm_start():
    original = CompletionNetwork(CompletionConfig(output_points=512, hidden_dim=32))
    model = CompletionNetwork(
        CompletionConfig(
            output_points=512,
            hidden_dim=32,
            use_surface_refiner=True,
            uncertainty_moves_points=False,
            bbox_margin=0,
        )
    )
    report = model.load_state_dict(original.state_dict(), strict=False)
    assert not report.unexpected_keys
    assert report.missing_keys
    assert all(k.startswith("surface_refiner.") for k in report.missing_keys)
    outputs = model(torch.randn(2, 32, 3))
    assert outputs["completed_points"].shape == (2, 512, 3)
    assert outputs["coarse_points"].shape == (2, 256, 3)
    from veil_net.losses.completion import CompletionLoss

    loss, _ = CompletionLoss(lambda_coarse=0.5, lambda_surface=0.5)(
        outputs, {"complete": torch.randn(2, 64, 3)}
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.surface_refiner.fold[-1].weight.grad.abs().sum() > 0
    assert model.decoder.seed.grad.abs().sum() > 0


def test_surface_patch_nonmultiple_output_count():
    from veil_net.models.surface_refiner import SurfacePatchRefiner

    refiner = SurfacePatchRefiner(32, 517)
    points, coarse = refiner(torch.randn(1, 517, 3), torch.randn(1, 32))
    assert points.shape == (1, 517, 3)
    assert coarse.shape == (1, 256, 3)

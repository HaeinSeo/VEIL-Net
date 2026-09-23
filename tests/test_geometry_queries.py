import torch
from rapc_net.models.rapc_net import RAPCNet, RAPCNetConfig


def test_geometry_queries_link_centers_to_encoder_and_route():
    torch.manual_seed(33)
    config = RAPCNetConfig(hidden_dim=32, output_points=256, spatial_tokens=8, spatial_patches=16,
        use_spatial_transformer=True, geometry_queries=True, two_stage_folding=True,
        input_frame_normalization=True, robust_input_geometry=True, bbox_margin=0,
        uncertainty_moves_points=False)
    model = RAPCNet(config)
    partial = torch.randn(2, 32, 3)
    result = model(partial)
    assert result['completed_points'].shape == (2, 256, 3)
    assert result['coarse_points'].shape == (2, 16, 3)
    assert not torch.allclose(result['coarse_points'][0], result['coarse_points'][1])
    (result['coarse_points'].square().mean()+result['generated_points'].square().mean()).backward()
    for part in (model.encoder, model.router, model.risk, model.spatial_completion.center_prediction,
                 model.spatial_completion.query_builder, model.spatial_completion.fold_second):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    clone = RAPCNet(config)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(partial)['completed_points'], result['completed_points'])

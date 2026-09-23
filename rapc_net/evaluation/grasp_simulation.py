"""MuJoCo isolated-object lift pilot, not an arm or real-robot benchmark.

The planner sees only a point cloud in meters. The independent simulator sees
CAD geometry. Mesh contact uses MuJoCo's convex hull: cavities are NOT resolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class GripperConfig:
    max_width: float = 0.12
    finger_thickness: float = 0.012
    finger_depth: float = 0.036
    finger_height: float = 0.044
    finger_force: float = 8.0
    mass: float = 0.1
    friction: float = 0.8
    timestep: float = 0.002
    approach_height: float = 0.15
    lift_height: float = 0.10
    success_height: float = 0.08
    hold_seconds: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0 or (value == 0 and name != 'friction'):
                raise ValueError(f'Invalid gripper parameter {name}={value}')
        if self.success_height >= self.lift_height:
            raise ValueError('Success height must be below commanded lift height')


def plan_top_down(points_m: np.ndarray, config: GripperConfig) -> dict:
    """A fixed geometric baseline, not a learned/optimal grasp planner.

    PCA is calculated on unique points with central 98% marginal bounds.
    It is never given CAD, GT bounds, object pose, or simulation outcomes.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Planner requires finite Nx3 points in meters')
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return dict(valid=False, reason='fewer_than_three_unique_planning_points')
    median = np.median(points, axis=0)
    radius = np.linalg.norm(points - median, axis=1)
    core = points[radius <= np.quantile(radius, 0.98)]
    if len(core) < 3:
        core = points
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(core[:, :2].T))
    closing = eigenvectors[:, np.argmin(eigenvalues)]
    if closing[np.argmax(np.abs(closing))] < 0:
        closing = -closing
    yaw = float(np.arctan2(closing[1], closing[0]))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = core[:, :2] @ rotation
    low, high = np.quantile(local, [0.01, 0.99], axis=0)
    width = float(high[0] - low[0])
    center_xy = ((low + high) / 2) @ rotation.T
    z = float(np.quantile(core[:, 2], [0.1, 0.9]).mean())
    z = max(z, config.finger_height / 2 + 0.004)
    return dict(valid=width + 0.008 <= config.max_width,
                reason='planned' if width + 0.008 <= config.max_width else 'jaw_width_exceeded',
                center=[float(center_xy[0]), float(center_xy[1]), z],
                yaw=yaw, estimated_width_m=width)


def _vec(values) -> str:
    return ' '.join(f'{float(x):.9g}' for x in values)


def make_scene(vertices_m: np.ndarray, plan: dict, config: GripperConfig) -> str:
    """Ground-truth CAD goes to physics only, never to plan_top_down."""
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('Physics requires at least four finite CAD vertices')
    center = vertices.mean(axis=0)
    root = ET.Element('mujoco', model='rapc_isolated_grasp_pilot')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep=str(config.timestep), gravity='0 0 -9.81',
                  integrator='implicitfast', iterations='80', cone='elliptic')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', condim='4', friction=f'{config.friction} 0.005 0.0001',
                  solref='0.01 1', solimp='0.95 0.99 0.001')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='cad_convex', vertex=_vec((vertices - center).ravel()))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.5 1', dir='0 0 -1', diffuse='0.9 0.9 0.9')
    ET.SubElement(world, 'geom', name='table', type='plane', size='1 1 0.1', rgba='0.8 0.82 0.84 1')
    obj = ET.SubElement(world, 'body', name='object', pos=_vec(center))
    ET.SubElement(obj, 'freejoint', name='object_free')
    ET.SubElement(obj, 'geom', name='object_geom', type='mesh', mesh='cad_convex',
                  mass=str(config.mass), rgba='0.15 0.62 0.42 1')
    base = ET.SubElement(world, 'body', name='gripper', pos=_vec(plan['center']),
                         euler=f'0 0 {plan["yaw"]}')
    ET.SubElement(base, 'joint', name='lift', type='slide', axis='0 0 1',
                  range='-0.3 1', damping='10')
    ET.SubElement(base, 'geom', name='palm', type='box', pos='0 0 0.042',
                  size=_vec([config.max_width / 2 + config.finger_thickness, 0.025, 0.012]),
                  mass='0.3', rgba='0.2 0.24 0.28 1')
    for sign, name in [(-1, 'left'), (1, 'right')]:
        body = ET.SubElement(base, 'body', name=name,
                             pos=f'{sign * config.finger_thickness / 2} 0 0')
        ET.SubElement(body, 'joint', name=name, type='slide', axis=f'{sign} 0 0',
                      range=f'0 {config.max_width / 2}', damping='1')
        ET.SubElement(body, 'geom', name=f'{name}_pad', type='box', mass='0.05',
                      size=_vec([config.finger_thickness / 2, config.finger_depth / 2,
                                 config.finger_height / 2]), rgba='0.3 0.35 0.4 1')
    actuators = ET.SubElement(root, 'actuator')
    ET.SubElement(actuators, 'position', name='lift_control', joint='lift', kp='10000', kv='150',
                  forcelimited='true', forcerange='-200 200')
    for name in ['left', 'right']:
        ET.SubElement(actuators, 'position', name=f'{name}_control', joint=name, kp='1000', kv='5',
                      forcelimited='true', forcerange=f'-{config.finger_force} {config.finger_force}')
    return ET.tostring(root, encoding='unicode')


def simulate_lift(vertices_m: np.ndarray, plan: dict, config: GripperConfig,
                  output: Path | None = None, render: bool = False) -> dict:
    import mujoco

    if not plan['valid']:
        return dict(success=False, reason=plan['reason'], lift_m=0.0, hold_contact_fraction=0.0)
    xml = make_scene(vertices_m, plan, config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.joint('lift').qpos[0] = config.approach_height
    for name in ['left', 'right']:
        data.joint(name).qpos[0] = config.max_width / 2
    data.ctrl[:] = [config.approach_height, config.max_width / 2, config.max_width / 2]
    mujoco.mj_forward(model, data)
    initial_z = float(data.body('object').xpos[2])
    geom_ids = {name: model.geom(name).id for name in ['object_geom', 'table', 'left_pad', 'right_pad', 'palm']}
    trace, frames, warnings = [], [], []
    renderer = None
    if render:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as exc:
            warnings.append(f'Rendering unavailable: {exc}')

    def snapshot(label):
        if renderer is None:
            return
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.08]
        camera.distance = max(0.45, float(np.linalg.norm(np.ptp(vertices_m, axis=0))) * 3)
        camera.azimuth, camera.elevation = 120, -25
        renderer.update_scene(data, camera)
        frames.append((label, renderer.render().copy()))

    def contacts():
        pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in data.contact[:data.ncon]}
        obj_id = geom_ids['object_geom']
        touch = lambda name: frozenset((obj_id, geom_ids[name])) in pairs
        return touch('left_pad'), touch('right_pad'), touch('table'), touch('palm')

    def phase(label, seconds, start, end):
        for i in range(max(1, round(seconds / config.timestep))):
            alpha = (i + 1) / max(1, round(seconds / config.timestep))
            data.ctrl[:] = (1 - alpha) * np.asarray(start) + alpha * np.asarray(end)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Nonfinite simulation state')
            left, right, table, palm = contacts()
            trace.append([float(data.time), label, float(data.body('object').xpos[2]),
                          int(left), int(right), int(table), int(palm)])
        snapshot(label)

    width = config.max_width / 2
    try:
        snapshot('initial')
        phase('approach', 1.0, [config.approach_height, width, width], [0, width, width])
        approach_collision = any(row[3] or row[4] or row[6] for row in trace)
        phase('close', 0.7, [0, width, width], [0, 0, 0])
        phase('lift', 1.0, [0, 0, 0], [config.lift_height, 0, 0])
        phase('hold', config.hold_seconds, [config.lift_height, 0, 0], [config.lift_height, 0, 0])
        hold = [row for row in trace if row[1] == 'hold']
        min_lift = min(row[2] for row in hold) - initial_z
        fraction = float(np.mean([row[3] and row[4] and not row[5] for row in hold]))
        numerical_warning = any(int(w.number) for w in data.warning)
        success = bool(min_lift >= config.success_height and fraction >= 0.9
                       and not approach_collision and not numerical_warning)
        reason = ('numerical_warning' if numerical_warning else 'approach_collision' if approach_collision
                  else 'lift_and_hold' if success else 'failed_lift_or_hold')
        result = dict(success=success, reason=reason, lift_m=float(min_lift),
                      hold_contact_fraction=fraction, approach_collision=bool(approach_collision),
                      numerical_warning=numerical_warning, config=asdict(config),
                      mujoco_version=mujoco.__version__, warnings=warnings)
        if output is not None:
            import csv
            import json
            output.mkdir(parents=True, exist_ok=True)
            (output / 'scene.xml').write_text(xml, encoding='utf-8')
            (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'phase', 'object_z_m', 'left_contact', 'right_contact', 'table_contact', 'palm_contact'])
                writer.writerows(trace)
            if frames:
                from PIL import Image, ImageDraw
                strip = Image.new('RGB', (640 * len(frames), 510), 'white')
                draw = ImageDraw.Draw(strip)
                for i, (label, frame) in enumerate(frames):
                    strip.paste(Image.fromarray(frame), (640 * i, 30))
                    draw.text((640 * i + 15, 8), label, fill='black')
                strip.save(output / 'stages.png')
        return result
    finally:
        if renderer is not None:
            renderer.close()

"""A2 feet rewards that run on mjlab.

feet_air_time_ref_mj: port of mimic_lite.feet_air_time_ref. The upstream one (track.py) reads Isaac
Lab's scene["feet_ground_contact"] sensor and its current_contact_time field, so it cannot run on
mjlab. Same math; robot contact from the mjlab "contact_forces" sensor via feet.py's helper.

feet_slip_mj: replacement for mimic_lite.feet_slip, whose asset.data.body_com_lin_vel_w needs
data.xipos, which mjlab's WarpBridge does not expose inside the compiled reward step
("'WarpBridge' object has no attribute 'xipos'"). Foot speed here is the per-step change of the
foot link position, which the tracking command already reads (sim xpos).

Reference/robot feet (body_names) must be tracking bodies (A2: .*_toe_pitch); contact uses the
sensor bodies (contact_body_names, A2: .*_toe_roll), paired left/right in order. Height defaults
are shifted for A2: both foot bodies sit 0.061 m above the sole when standing (G1 ~0.035).
"""

import torch

from active_adaptation.envs.utils import find_bodies, find_sensor_bodies

from .feet import TrackReward, _current_in_contact
from .track import _select_tracking_body_names


def _pair_feet(reward, body_names, contact_body_names, name):
    """Tracking-body indices of the feet and contact-sensor ids of their contact bodies, paired."""
    body_indices_tracking, ref_names = _select_tracking_body_names(reward.command_manager, body_names)
    _, contact_names = find_bodies(reward.asset, contact_body_names)
    if len(contact_names) != len(ref_names):
        raise ValueError(
            f"{name}: {len(ref_names)} feet {ref_names} vs {len(contact_names)} contact bodies {contact_names}"
        )
    for ref_name, contact_name in zip(ref_names, contact_names):
        if ref_name.split("_")[0] != contact_name.split("_")[0]:
            raise ValueError(f"{name}: side mismatch {ref_name} vs {contact_name}")
    sensor_ids, sensor_names = find_sensor_bodies(reward.asset, reward.contact_sensor, contact_names)
    missing = sorted(set(contact_names) - set(sensor_names))
    if missing:
        raise RuntimeError(f"{name}: missing feet in contact sensor: {missing}")
    ordered = [sensor_ids[list(sensor_names).index(n)] for n in contact_names]
    return list(body_indices_tracking), torch.as_tensor(ordered, dtype=torch.long, device=reward.device)


class feet_slip_mj(TrackReward, namespace="mimic_lite"):
    """-sum over feet in contact of clamp(horizontal foot speed - tolerance, 0, 1)."""

    supported_backends = ("isaaclab", "mjlab")

    def _initialize_impl(
        self,
        body_names: str | list[str],
        contact_body_names: str | list[str],
        tolerance: float = 0.0,
        **kwargs,
    ):
        self.tolerance = float(tolerance)
        self.asset = self.command_manager.asset
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        self.body_indices_tracking, self.sensor_body_ids = _pair_feet(
            self, body_names, contact_body_names, "feet_slip_mj"
        )
        num_feet = len(self.body_indices_tracking)
        self.prev_pos = torch.zeros(self.num_envs, num_feet, 2, device=self.device)
        self.valid = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids, reset_td=None):
        self.valid[env_ids] = 0.0  # no previous position after a reset

    def _compute(self):
        pos = self.command_manager.robot_body_link_pos_w[:, self.body_indices_tracking, :2]
        speed = ((pos - self.prev_pos).norm(dim=-1) / self.env.step_dt - self.tolerance).clamp(0.0, 1.0)
        self.prev_pos[:] = pos
        speed = speed * self.valid
        self.valid[:] = 1.0
        in_contact = _current_in_contact(self.contact_sensor, self.sensor_body_ids)
        return -(in_contact * speed).sum(dim=1, keepdim=True)


class feet_air_time_ref_mj(TrackReward, namespace="mimic_lite"):
    """Air time gated by the reference contact state.

    A reference foot "stands" when it is slow and low. Per step, while the robot foot's contact
    state matches the reference, air time grows (scaled up with foot height); while it does not,
    air time counts down. At each robot touchdown the reward is min(air_time - thres, 0), then the
    air time resets. A lift the reference also lifts is not charged; lifting or tapping while the
    reference stands is.
    """

    supported_backends = ("isaaclab", "mjlab")

    def _initialize_impl(
        self,
        body_names: str | list[str],
        contact_body_names: str | list[str],
        thres: float,
        ref_vel_thres: float = 0.2,
        ref_height_thres: float = 0.176,
        height_range: tuple[float, float] | list[float] = (0.061, 0.146),
        coef_range: tuple[float, float] | list[float] = (0.5, 2.0),
        **kwargs,
    ):
        self.thres = float(thres)
        self.ref_vel_thres = float(ref_vel_thres)
        self.ref_height_thres = float(ref_height_thres)
        self.asset = self.command_manager.asset
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        self.body_indices_tracking, self.sensor_body_ids = _pair_feet(
            self, body_names, contact_body_names, "feet_air_time_ref_mj"
        )

        num_feet = len(self.body_indices_tracking)
        self.reward_time = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.last_contact = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device)

        self.h_low, self.h_high = float(height_range[0]), float(height_range[1])
        self.c_low, self.c_high = float(coef_range[0]), float(coef_range[1])
        self.log_c_ratio = float(torch.log(torch.tensor(self.c_high / self.c_low)))

    def reset(self, env_ids, reset_td=None):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def _compute(self):
        current_contact = _current_in_contact(self.contact_sensor, self.sensor_body_ids)
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        ref_vel = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_tracking]
        ref_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_tracking]
        ref_standing = (ref_vel.norm(dim=-1) < self.ref_vel_thres) & (ref_pos[..., 2] < self.ref_height_thres)

        feet_height = self.command_manager.robot_body_link_pos_w[:, self.body_indices_tracking, 2]
        t = ((feet_height - self.h_low) / (self.h_high - self.h_low)).clamp(0.0, 1.0)
        height_coef = self.c_low * torch.exp(self.log_c_ratio * t)

        mismatch = ref_standing ^ current_contact
        self.reward_time = self.reward_time + torch.where(
            mismatch, -self.env.step_dt, self.env.step_dt * height_coef
        )
        reward = torch.sum((self.reward_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True)
        self.reward_time = self.reward_time * (~current_contact)
        return reward

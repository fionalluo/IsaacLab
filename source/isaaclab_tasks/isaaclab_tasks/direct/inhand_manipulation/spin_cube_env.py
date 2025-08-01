# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for spinning cube task."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform, saturate

from .inhand_manipulation_env import InHandManipulationEnv, unscale

if TYPE_CHECKING:
    from .spin_cube_env_cfg import (
        SpinCubeEnvCfg, 
        SpinCubeShadowHandEnvCfg,
        SpinCubeShadowHandWithContactSensorsEnvCfg,
        SpinCubeShadowHandWithBinaryContactSensorsEnvCfg,
        SpinCubeShadowHandWithMagnitudeContactSensorsEnvCfg
    )


class SpinCubeEnv(InHandManipulationEnv):
    """Environment where the robot tries to spin the cube around its Z-axis as much as possible."""
    
    cfg: (
        SpinCubeEnvCfg | 
        SpinCubeShadowHandEnvCfg |
        SpinCubeShadowHandWithContactSensorsEnvCfg |
        SpinCubeShadowHandWithBinaryContactSensorsEnvCfg |
        SpinCubeShadowHandWithMagnitudeContactSensorsEnvCfg
    )

    def __init__(self, cfg: (
        SpinCubeEnvCfg | 
        SpinCubeShadowHandEnvCfg |
        SpinCubeShadowHandWithContactSensorsEnvCfg |
        SpinCubeShadowHandWithBinaryContactSensorsEnvCfg |
        SpinCubeShadowHandWithMagnitudeContactSensorsEnvCfg
    ), render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Track rotation history for spin calculation
        self.initial_rotation = None
        self.total_z_rotation = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.rotation_history = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Track object velocity for reward calculation
        self.object_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the environment and initialize rotation tracking."""
        super()._reset_idx(env_ids)
        
        # Initialize rotation tracking for reset environments
        if env_ids is not None:
            self.total_z_rotation[env_ids] = 0.0
            self.rotation_history[env_ids] = 0.0
            self.object_vel[env_ids] = 0.0
            
            # Reset spin-specific tracking
            if hasattr(self, 'spin_consecutive_successes'):
                self.spin_consecutive_successes[env_ids] = 0.0
            
            # Store initial rotation for these environments
            if self.initial_rotation is None:
                self.initial_rotation = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
            self.initial_rotation[env_ids] = self.object_rot[env_ids].clone()

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on Z-axis rotation."""
        # Update object velocity for reward calculation
        self._update_object_velocity()
        
        # Get base rewards from parent class
        total_reward = super()._get_rewards()
        
        # Compute spin reward
        spin_reward = self._compute_spin_reward()
        
        # Add spin reward to total
        total_reward += spin_reward
        
        # Track spin success for logging
        self._track_spin_success()
        
        return total_reward

    def _track_spin_success(self):
        """Track spin success for logging purposes."""
        if "log" not in self.extras:
            self.extras["log"] = dict()
        
        # Check if minimum rotations achieved
        rotations_achieved = self.total_z_rotation / (2 * np.pi)
        spin_success = rotations_achieved >= self.cfg.min_rotations_for_success
        
        # Log spin success rate (similar to original success rate logging)
        if torch.any(self.reset_buf):
            # Count how many episodes ended this step
            num_episodes_ended = torch.sum(self.reset_buf).item()
            
            # Collect spin success data for environments that just reset
            if num_episodes_ended > 0:
                reset_env_ids = torch.nonzero(self.reset_buf).squeeze(-1)
                if len(reset_env_ids.shape) == 0:  # Single environment
                    reset_env_ids = reset_env_ids.unsqueeze(0)
                
                for env_id in reset_env_ids:
                    # Check if this episode ended with spin success
                    episode_spin_success = 1.0 if spin_success[env_id].item() else 0.0
                    if not hasattr(self, 'spin_success_eval_batch'):
                        self.spin_success_eval_batch = []
                    self.spin_success_eval_batch.append(episode_spin_success)
            
            # Log spin success rate when we reach evaluation interval
            if not hasattr(self, 'spin_success_eval_count'):
                self.spin_success_eval_count = 0
            self.spin_success_eval_count += num_episodes_ended
            
            if self.spin_success_eval_count >= 100:  # Log every 100 episodes
                if hasattr(self, 'spin_success_eval_batch') and len(self.spin_success_eval_batch) > 0:
                    # Calculate percentage of successful spin episodes
                    spin_success_rate = sum(self.spin_success_eval_batch) / len(self.spin_success_eval_batch)
                    self.extras["log"]["spin_success_rate"] = spin_success_rate
                    
                    # Reset evaluation batch
                    self.spin_success_eval_count = 0
                    self.spin_success_eval_batch = []
        
        # Log current spin progress
        self.extras["log"]["avg_rotations_achieved"] = rotations_achieved.mean().item()
        self.extras["log"]["max_rotations_achieved"] = rotations_achieved.max().item()

    def _update_object_velocity(self):
        """Update object velocity for reward calculation."""
        # Get current object position
        current_pos = self.object_pos
        
        # Calculate velocity as finite difference (if we have previous position)
        if hasattr(self, 'prev_object_pos'):
            dt = 1.0 / 120.0  # Assuming 120 Hz physics
            self.object_vel = (current_pos - self.prev_object_pos) / dt
        else:
            self.object_vel = torch.zeros_like(current_pos)
        
        # Store current position for next step
        self.prev_object_pos = current_pos.clone()

    def _compute_spin_reward(self) -> torch.Tensor:
        """Compute spin reward similar to original orientation task structure."""
        # Get current object state
        current_rot = self.object_rot
        current_pos = self.object_pos
        
        # 1. POSITION REWARD - keep cube near center of hand (like original task)
        goal_dist = torch.norm(current_pos - self.in_hand_pos, p=2, dim=-1)
        dist_rew = goal_dist * self.cfg.dist_reward_scale
        
        # 2. ROTATION REWARD - reward for spinning (new for spin task)
        # Calculate Z-axis rotation change
        z_rotation = self._quaternion_to_z_rotation(current_rot)
        rotation_change = z_rotation - self.rotation_history
        
        # Handle wrapping around 2π
        rotation_change = torch.where(rotation_change > np.pi, rotation_change - 2*np.pi, rotation_change)
        rotation_change = torch.where(rotation_change < -np.pi, rotation_change + 2*np.pi, rotation_change)
        
        # Update total rotation and history
        self.total_z_rotation += rotation_change
        self.rotation_history = z_rotation
        
        # Rotation reward (similar to original rot_rew structure)
        rot_rew = rotation_change * self.cfg.spin_reward_scale
        
        # 3. ACTION PENALTY - same as original task
        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale
        
        # 4. FALL PENALTY - same as original task
        fall_penalty = torch.where(
            goal_dist >= self.cfg.fall_dist,
            self.cfg.fall_penalty,
            0.0
        )
        
        # 5. SUCCESS BONUS - for achieving minimum rotations
        success_bonus = 0.0
        if hasattr(self.cfg, 'min_rotations_for_success'):
            rotations_achieved = self.total_z_rotation / (2 * np.pi)
            success_bonus = torch.where(
                rotations_achieved >= self.cfg.min_rotations_for_success,
                self.cfg.success_bonus,
                0.0
            )
        
        # Total reward: position + rotation + action penalty + fall penalty + success bonus
        total_spin_reward = dist_rew + rot_rew + action_penalty + fall_penalty + success_bonus
        
        return total_spin_reward



    def _quaternion_to_z_rotation(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract Z-axis rotation from quaternion."""
        # Convert quaternion to Z rotation (simplified)
        # This is a simplified version - for more accuracy, you'd convert to euler angles
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Extract Z rotation from quaternion
        # This is an approximation - for exact Z rotation, convert to euler angles
        z_rotation = 2 * torch.atan2(z, w)
        
        return z_rotation

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episode should end."""
        # Get base termination conditions
        out_of_reach, time_out = super()._get_dones()
        
        # Add spin-specific termination logic
        if hasattr(self.cfg, 'max_consecutive_success') and self.cfg.max_consecutive_success > 0:
            # Check if minimum rotations achieved for success
            rotations_achieved = self.total_z_rotation / (2 * np.pi)
            spin_success = rotations_achieved >= self.cfg.min_rotations_for_success
            
            # Update consecutive successes
            if not hasattr(self, 'spin_consecutive_successes'):
                self.spin_consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
            # Increment consecutive successes for successful spins
            self.spin_consecutive_successes = torch.where(
                spin_success,
                self.spin_consecutive_successes + 1.0,
                torch.zeros_like(self.spin_consecutive_successes)
            )
            
            # Check if max consecutive successes reached
            max_success_reached = self.spin_consecutive_successes >= self.cfg.max_consecutive_success
            
            # Add to time_out condition
            time_out = time_out | max_success_reached
        
        return out_of_reach, time_out

    def _get_observations(self) -> dict:
        """Get observations without object state information."""
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_spin_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_spin_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_spin_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
            
        # Add contact sensor data if available
        if hasattr(self, 'contact_sensors'):
            contact_data = self.contact_sensors.data
            # Get raw contact forces (shape: [num_envs, num_bodies, 3])
            raw_contact_forces = contact_data.net_forces_w  # Shape: [8192, 26, 3]
            
            # Process contact data based on configuration type
            if hasattr(self.cfg, 'contact_sensors'):
                # Determine the type of contact sensor processing based on config class name
                config_class_name = self.cfg.__class__.__name__
                
                if "Binary" in config_class_name:
                    # Binary contact sensors: 0 or 1 for any contact
                    contact_magnitudes = torch.norm(raw_contact_forces, dim=-1)  # Shape: [8192, 26]
                    binary_contacts = (contact_magnitudes > 0.001).float()  # Threshold for contact detection
                    contact_forces = binary_contacts.view(self.num_envs, -1)  # Shape: [8192, 26]
                    
                elif "Magnitude" in config_class_name:
                    # Magnitude-only contact sensors: √(Fx² + Fy² + Fz²)
                    contact_magnitudes = torch.norm(raw_contact_forces, dim=-1)  # Shape: [8192, 26]
                    contact_forces = contact_magnitudes.view(self.num_envs, -1)  # Shape: [8192, 26]
                    
                else:
                    # Full 3D contact sensors: (Fx, Fy, Fz) for each body
                    contact_forces = raw_contact_forces.view(self.num_envs, -1)  # Shape: [8192, 78]
            
            # Extend the observation space with contact data
            if "policy" in observations:
                observations["policy"] = torch.cat([observations["policy"], contact_forces], dim=-1)
            if "critic" in observations:
                observations["critic"] = torch.cat([observations["critic"], contact_forces], dim=-1)
        
        return observations

    def compute_spin_reduced_observations(self):
        """Compute reduced observations for spin task without object state."""
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   NO Object Position (removed for spin task)
        #   NO Relative target orientation (removed for spin task)
        #   NO total rotation (removed - object state related)
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_spin_full_observations(self):
        """Compute full observations for spin task without object state."""
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # NO object state (removed for spin task)
                # NO goal state (removed for spin task)
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_spin_full_state(self):
        """Compute full state for spin task without object state."""
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # NO object state (removed for spin task)
                # NO goal state (removed for spin task)
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states 
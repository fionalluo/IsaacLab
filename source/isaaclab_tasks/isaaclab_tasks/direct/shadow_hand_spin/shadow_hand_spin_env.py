# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

if TYPE_CHECKING:
    from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
    from isaaclab_tasks.direct.shadow_hand_spin.shadow_hand_spin_cfg import (
        IsaacSpinCubeShadowWithObjectStateV0Cfg,
        IsaacSpinCubeShadowV0Cfg,
        IsaacSpinCubeShadowNoContactSensorsV0Cfg,
    )


class ShadowHandSpinEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg | IsaacSpinCubeShadowWithObjectStateV0Cfg | IsaacSpinCubeShadowV0Cfg | IsaacSpinCubeShadowNoContactSensorsV0Cfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg | IsaacSpinCubeShadowWithObjectStateV0Cfg | IsaacSpinCubeShadowV0Cfg | IsaacSpinCubeShadowNoContactSensorsV0Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04

        # track rotations for logging
        self.total_rotations = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)  # No success concept for spinning
        
        # track previous rotation for rotation reward
        self.prev_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        # batched evaluation for rotation logging
        self.rotation_eval_count = 0
        self.rotation_eval_interval = 100  # Log average rotations every 100 episodes
        self.rotation_eval_batch = []
        
        # Flag to track if intermediate values have been computed
        self._intermediate_values_computed = False

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        
        # Add contact sensors if configured
        if hasattr(self.cfg, 'contact_sensors'):
            from isaaclab.sensors import ContactSensor
            self.contact_sensors = ContactSensor(self.cfg.contact_sensors)
            self.scene.sensors["contact_sensors"] = self.contact_sensors
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        # Compute intermediate values on first call
        if not self._intermediate_values_computed:
            self._compute_intermediate_values()
            self._intermediate_values_computed = True
            
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
            
        # Debug: Print observation space structure occasionally
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
            
        if self._step_count % 100 == 0:  # Print every 100 steps
            print(f"\n[OBSERVATION DEBUG] Step {self._step_count}:")
            print(f"[OBSERVATION DEBUG] Observation keys: {list(observations.keys())}")
            
            # Analyze the policy observation structure
            policy_obs = observations["policy"]
            print(f"[OBSERVATION DEBUG] Policy observation shape: {policy_obs.shape}")
            
            # Break down the observation based on obs_type
            if self.cfg.obs_type == "full":
                self._debug_full_observation_structure(policy_obs)
            elif self.cfg.obs_type == "openai":
                self._debug_openai_observation_structure(policy_obs)
            
            if self.cfg.asymmetric_obs and "critic" in observations:
                critic_obs = observations["critic"]
                print(f"[OBSERVATION DEBUG] Critic observation shape: {critic_obs.shape}")
            
        # Add contact sensor data if available and configured for observations
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
            
            # Debug: Print detailed contact force information occasionally
            if self._step_count % 100 == 0:  # Print every 100 steps
                max_force = torch.max(contact_forces).item()
                avg_force = torch.mean(contact_forces).item()
                print(f"[CONTACT DEBUG] Step {self._step_count}: Max force = {max_force:.4f}, Avg force = {avg_force:.4f}")
                print(f"[CONTACT DEBUG] Contact forces shape: {contact_forces.shape}")
                print(f"[CONTACT DEBUG] Number of bodies: {self.contact_sensors.num_bodies}")
                print(f"[CONTACT DEBUG] Contact sensor type: {config_class_name}")
                
                # Print detailed contact data for environment 0
                print(f"\n[CONTACT DEBUG] Detailed contact data for environment 0:")
                print(f"Body names: {self.contact_sensors.body_names}")
                
                # Get contact forces for environment 0
                env0_forces = contact_forces[0]  # Shape depends on type
                
                print(f"\n[CONTACT DEBUG] Contact forces for each body (env 0):")
                for i, body_name in enumerate(self.contact_sensors.body_names):
                    if "Binary" in config_class_name:
                        contact_val = env0_forces[i].item()
                        print(f"  {body_name:20s}: Binary contact = {contact_val:.0f}")
                    elif "Magnitude" in config_class_name:
                        magnitude = env0_forces[i].item()
                        print(f"  {body_name:20s}: Magnitude = {magnitude:8.4f}")
                    else:
                        # Full 3D forces
                        fx, fy, fz = raw_contact_forces[0, i].tolist()
                        force_magnitude = torch.norm(raw_contact_forces[0, i]).item()
                        print(f"  {body_name:20s}: Fx={fx:8.4f}, Fy={fy:8.4f}, Fz={fz:8.4f}, |F|={force_magnitude:8.4f}")
                
                # Show which bodies have non-zero contact
                if "Binary" in config_class_name:
                    non_zero_bodies = [self.contact_sensors.body_names[i] for i in range(len(self.contact_sensors.body_names)) if env0_forces[i] > 0.5]
                elif "Magnitude" in config_class_name:
                    non_zero_bodies = [self.contact_sensors.body_names[i] for i in range(len(self.contact_sensors.body_names)) if env0_forces[i] > 0.001]
                else:
                    non_zero_bodies = [self.contact_sensors.body_names[i] for i in range(len(self.contact_sensors.body_names)) if torch.norm(raw_contact_forces[0, i]) > 0.001]
                
                print(f"\n[CONTACT DEBUG] Bodies with non-zero contact (env 0): {non_zero_bodies}")
            
            # Only add contact data to observations if not the "NoContactSensors" config
            if "NoContactSensors" not in config_class_name:
                # Extend the observation space with contact data
                if "policy" in observations:
                    # print(f"[OBSERVATION DEBUG] Adding contact forces to policy observation")
                    # print(f"[OBSERVATION DEBUG] Original policy obs shape: {observations['policy'].shape}")
                    # print(f"[OBSERVATION DEBUG] Contact forces shape: {contact_forces.shape}")
                    observations["policy"] = torch.cat([observations["policy"], contact_forces], dim=-1)
                    # print(f"[OBSERVATION DEBUG] Final policy obs shape: {observations['policy'].shape}")
                if "critic" in observations:
                    observations["critic"] = torch.cat([observations["critic"], contact_forces], dim=-1)
                
        return observations

    def _debug_full_observation_structure(self, obs):
        """Debug the full observation structure."""
        print(f"[OBSERVATION DEBUG] Full observation breakdown:")
        
        # Based on compute_full_observations() method
        start_idx = 0
        
        # Hand joint positions (scaled)
        joint_pos_size = self.num_hand_dofs
        print(f"  [{start_idx:3d}:{start_idx+joint_pos_size:3d}] Hand joint positions (scaled): {joint_pos_size} dims")
        start_idx += joint_pos_size
        
        # Hand joint velocities
        joint_vel_size = self.num_hand_dofs
        print(f"  [{start_idx:3d}:{start_idx+joint_vel_size:3d}] Hand joint velocities: {joint_vel_size} dims")
        start_idx += joint_vel_size
        
        # Object position
        obj_pos_size = 3
        print(f"  [{start_idx:3d}:{start_idx+obj_pos_size:3d}] Object position: {obj_pos_size} dims")
        start_idx += obj_pos_size
        
        # Object rotation (quaternion)
        obj_rot_size = 4
        print(f"  [{start_idx:3d}:{start_idx+obj_rot_size:3d}] Object rotation: {obj_rot_size} dims")
        start_idx += obj_rot_size
        
        # Object linear velocity
        obj_linvel_size = 3
        print(f"  [{start_idx:3d}:{start_idx+obj_linvel_size:3d}] Object linear velocity: {obj_linvel_size} dims")
        start_idx += obj_linvel_size
        
        # Object angular velocity
        obj_angvel_size = 3
        print(f"  [{start_idx:3d}:{start_idx+obj_angvel_size:3d}] Object angular velocity: {obj_angvel_size} dims")
        start_idx += obj_angvel_size
        
        # No goal components for spinning task
        
        # Fingertip positions
        fingertip_pos_size = self.num_fingertips * 3
        print(f"  [{start_idx:3d}:{start_idx+fingertip_pos_size:3d}] Fingertip positions: {fingertip_pos_size} dims")
        start_idx += fingertip_pos_size
        
        # Fingertip rotations
        fingertip_rot_size = self.num_fingertips * 4
        print(f"  [{start_idx:3d}:{start_idx+fingertip_rot_size:3d}] Fingertip rotations: {fingertip_rot_size} dims")
        start_idx += fingertip_rot_size
        
        # Fingertip velocities
        fingertip_vel_size = self.num_fingertips * 6
        print(f"  [{start_idx:3d}:{start_idx+fingertip_vel_size:3d}] Fingertip velocities: {fingertip_vel_size} dims")
        start_idx += fingertip_vel_size
        
        # Actions
        action_size = self.cfg.action_space
        print(f"  [{start_idx:3d}:{start_idx+action_size:3d}] Actions: {action_size} dims")
        start_idx += action_size
        
        print(f"[OBSERVATION DEBUG] Total base observation size: {start_idx}")
        
        # Only mention contact sensors if they exist
        if hasattr(self, 'contact_sensors'):
            contact_size = self.contact_sensors.num_bodies * 3
            print(f"[OBSERVATION DEBUG] Expected total with contact sensors: {start_idx + contact_size}")
        else:
            print(f"[OBSERVATION DEBUG] No contact sensors configured")

    def _debug_openai_observation_structure(self, obs):
        """Debug the OpenAI observation structure."""
        print(f"[OBSERVATION DEBUG] OpenAI observation breakdown:")
        
        # Based on compute_reduced_observations() method
        start_idx = 0
        
        # Fingertip positions
        fingertip_pos_size = self.num_fingertips * 3
        print(f"  [{start_idx:3d}:{start_idx+fingertip_pos_size:3d}] Fingertip positions: {fingertip_pos_size} dims")
        start_idx += fingertip_pos_size
        
        # Object position
        obj_pos_size = 3
        print(f"  [{start_idx:3d}:{start_idx+obj_pos_size:3d}] Object position: {obj_pos_size} dims")
        start_idx += obj_pos_size
        
        # No goal components for spinning task
        
        # Actions
        action_size = self.cfg.action_space
        print(f"  [{start_idx:3d}:{start_idx+action_size:3d}] Actions: {action_size} dims")
        start_idx += action_size
        
        print(f"[OBSERVATION DEBUG] Total base observation size: {start_idx}")
        
        # Only mention contact sensors if they exist
        if hasattr(self, 'contact_sensors'):
            contact_size = self.contact_sensors.num_bodies * 3
            print(f"[OBSERVATION DEBUG] Expected total with contact sensors: {start_idx + contact_size}")
        else:
            print(f"[OBSERVATION DEBUG] No contact sensors configured")

    def _get_rewards(self) -> torch.Tensor:
        # Get contact forces if available
        if hasattr(self, 'contact_sensors'):
            contact_data = self.contact_sensors.data
            raw_contact_forces = contact_data.net_forces_w  # Shape: [8192, 26, 3]
            
            # Process contact data based on configuration type
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
        else:
            # No contact sensors - create dummy tensor with all zeros
            contact_forces = torch.zeros((self.num_envs, 1), device=self.device)
        
        (
            total_reward,
            rotation_delta,
            self.reset_goal_buf,
            self.total_rotations[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.total_rotations,
            # self.consecutive_successes,  # No success concept for spinning
            self.max_episode_length,
            self.prev_object_rot,
            self.object_rot,
            self.object_pos,
            self.in_hand_pos,
            self.object_linvel,
            self.fingertip_pos,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.rotation_reward_scale,
            self.cfg.linear_velocity_penalty_scale,
            self.cfg.distance_reward_scale,
            contact_forces,
            # self.cfg.reach_goal_bonus,  # No goal success for spinning
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.tilt_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        # self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()  # No success concept for spinning
        
        # Batched evaluation for rotation logging - only log every rotation_eval_interval episodes
        if torch.any(self.reset_buf):
            # Count how many episodes ended this step
            num_episodes_ended = torch.sum(self.reset_buf).item()
            self.rotation_eval_count += num_episodes_ended
            
            # Collect rotation data for environments that just reset
            if num_episodes_ended > 0:
                reset_env_ids = torch.nonzero(self.reset_buf).squeeze(-1)
                if len(reset_env_ids.shape) == 0:  # Single environment
                    reset_env_ids = reset_env_ids.unsqueeze(0)
                
                for env_id in reset_env_ids:
                    # Get total rotations for this episode
                    episode_rotations = self.total_rotations[env_id].item()
                    self.rotation_eval_batch.append(episode_rotations)
            
            # Log average rotations when we reach evaluation interval
            if self.rotation_eval_count >= self.rotation_eval_interval:
                if len(self.rotation_eval_batch) > 0:
                    # Calculate average rotations per episode
                    avg_rotations = sum(self.rotation_eval_batch) / len(self.rotation_eval_batch)
                    self.extras["log"]["avg_rotations_per_episode"] = avg_rotations
                    
                    # Reset evaluation batch
                    self.rotation_eval_count = 0
                    self.rotation_eval_batch = []

        # No goal reset for spinning task

        return total_reward

    def get_contact_info(self):
        """Get detailed contact information for debugging or analysis."""
        if not hasattr(self, 'contact_sensors'):
            return None
        
        contact_data = self.contact_sensors.data
        return {
            'net_forces': contact_data.net_forces_w,
            'body_names': self.contact_sensors.body_names,
            'num_bodies': self.contact_sensors.num_bodies,
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        # Check if cube's z-axis has tilted more than 60 degrees from global z-axis
        tilt_angle = compute_tilt_angle(self.object_rot)
        
        # Reset if tilt angle is greater than 60 degrees
        excessive_tilt = tilt_angle > 60.0
        
        # Combine both reset conditions
        out_of_reach = out_of_reach | excessive_tilt

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # No goal reset for spinning task

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # Only randomize rotation around Z-axis (yaw) to keep Z-axis pointing up
        z_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)  # noise for Z rotation only
        object_default_state[:, 3:7] = randomize_z_rotation(
            z_rot_noise[:, 0], self.z_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.total_rotations[env_ids] = 0
        self._compute_intermediate_values()
        
        # Initialize previous rotation for newly reset environments
        self.prev_object_rot[env_ids] = self.object_rot[env_ids].clone()

    # No goal reset for spinning task

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # Store previous rotation before updating current (only if object_rot exists and is not zeros)
        if hasattr(self, 'object_rot') and torch.any(self.object_rot != 0):
            self.prev_object_rot = self.object_rot.clone()
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   No goal orientation for spinning task
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        """Compute full observations based on configuration."""
        config_class_name = self.cfg.__class__.__name__
        
        # Start with hand observations (always included)
        obs_components = [
            # hand
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.cfg.vel_obs_scale * self.hand_dof_vel,
        ]
        
        # Add object state if configured
        if "WithObjectState" in config_class_name:
            obs_components.extend([
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
            ])
        
        # Add fingertip observations (always included)
        obs_components.extend([
            # fingertips
            self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
            self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
            self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
        ])
        
        # Add actions (always included)
        obs_components.append(self.actions)
        
        obs = torch.cat(obs_components, dim=-1)
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
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


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def randomize_z_rotation(rand0, z_unit_tensor):
    return quat_from_angle_axis(rand0 * np.pi, z_unit_tensor)


@torch.jit.script
def compute_clockwise_rotation_reward(prev_rot: torch.Tensor, curr_rot: torch.Tensor, rotation_reward_scale: float, contact_forces: torch.Tensor):
    """Compute reward for clockwise rotation around the z-axis, only when in contact."""
    # Extract the x-axis of the cube in both previous and current rotations
    # For quaternion (w, x, y, z), the x-axis is: (1-2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y))
    
    # Previous rotation x-axis
    w_prev, x_prev, y_prev, z_prev = prev_rot[:, 0], prev_rot[:, 1], prev_rot[:, 2], prev_rot[:, 3]
    prev_x_axis = torch.stack([
        1 - 2 * (y_prev * y_prev + z_prev * z_prev),
        2 * (x_prev * y_prev + w_prev * z_prev),
        2 * (x_prev * z_prev - w_prev * y_prev)
    ], dim=-1)
    
    # Current rotation x-axis
    w_curr, x_curr, y_curr, z_curr = curr_rot[:, 0], curr_rot[:, 1], curr_rot[:, 2], curr_rot[:, 3]
    curr_x_axis = torch.stack([
        1 - 2 * (y_curr * y_curr + z_curr * z_curr),
        2 * (x_curr * y_curr + w_curr * z_curr),
        2 * (x_curr * z_curr - w_curr * y_curr)
    ], dim=-1)
    
    # Project both x-axes to the global xy plane (set z component to 0)
    prev_x_xy = torch.stack([prev_x_axis[:, 0], prev_x_axis[:, 1], torch.zeros_like(prev_x_axis[:, 0])], dim=-1)
    curr_x_xy = torch.stack([curr_x_axis[:, 0], curr_x_axis[:, 1], torch.zeros_like(curr_x_axis[:, 0])], dim=-1)
    
    # Normalize the projected vectors
    prev_x_xy_norm = torch.norm(prev_x_xy, dim=-1, keepdim=True)
    curr_x_xy_norm = torch.norm(curr_x_xy, dim=-1, keepdim=True)
    
    # Avoid division by zero
    prev_x_xy = torch.where(prev_x_xy_norm > 1e-6, prev_x_xy / prev_x_xy_norm, prev_x_xy)
    curr_x_xy = torch.where(curr_x_xy_norm > 1e-6, curr_x_xy / curr_x_xy_norm, curr_x_xy)
    
    # Calculate the cross product to determine rotation direction
    cross_product = torch.cross(prev_x_xy, curr_x_xy, dim=-1)
    cross_z = cross_product[:, 2]  # z-component of cross product
    
    # Calculate the angle between the two vectors
    dot_product = torch.sum(prev_x_xy * curr_x_xy, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp to avoid numerical issues
    angle = torch.acos(dot_product)
    
    # Positive reward for clockwise rotation (negative cross_z), negative for counterclockwise
    # The angle is always positive, so we use the sign of cross_z to determine direction
    rotation_delta_true = torch.where(cross_z < 0, angle, -angle)
    
    # Clip the rotation delta to not reward excessive spinning
    rotation_delta = torch.clamp(rotation_delta_true, -0.157, 0.157)
    
    # Check if there's any contact between hand and object
    total_contact_per_env = torch.sum(torch.abs(contact_forces), dim=-1)
    has_contact = total_contact_per_env > 0.001
    
    # Zero out rotation reward if there's no contact
    rotation_delta = torch.where(has_contact, rotation_delta, torch.zeros_like(rotation_delta))
    
    # Apply the reward scale
    rotation_reward = rotation_delta * rotation_reward_scale
    
    # return the true rotation delta, regardless of contact, clipping, etc. 
    return rotation_reward, rotation_delta_true


@torch.jit.script
def compute_action_penalty(actions: torch.Tensor, action_penalty_scale: float):
    """Compute action regularization penalty."""
    action_penalty = torch.sum(actions**2, dim=-1)
    return action_penalty * action_penalty_scale


@torch.jit.script
def compute_linear_velocity_penalty(object_linvel: torch.Tensor, linear_velocity_penalty_scale: float):
    """Compute penalty for object linear velocity to prevent tossing."""
    object_linear_velocity = torch.norm(object_linvel, dim=-1)
    linear_velocity_penalty = object_linear_velocity * linear_velocity_penalty_scale
    return linear_velocity_penalty


@torch.jit.script
def compute_distance_reward(fingertip_pos: torch.Tensor, object_pos: torch.Tensor, distance_reward_scale: float):
    """Compute reward for fingertips staying close to object."""
    # Calculate distance from each fingertip to object
    # fingertip_pos shape: [num_envs, num_fingertips, 3]
    # object_pos shape: [num_envs, 3]
    # Expand object_pos to match fingertip_pos shape
    object_pos_expanded = object_pos.unsqueeze(1).expand(-1, fingertip_pos.shape[1], -1)
    
    # Calculate distances from each fingertip to object
    distances = torch.norm(fingertip_pos - object_pos_expanded, dim=-1)  # Shape: [num_envs, num_fingertips]
    
    # Apply the distance reward formula: clip(0.1 / (0.02 + 4*distance), 0, 1)
    distance_rewards = torch.clamp(0.1 / (0.02 + 4.0 * distances), 0.0, 1.0)
    
    # Take the mean across all fingertips
    mean_distance_reward = torch.mean(distance_rewards, dim=-1)  # Shape: [num_envs]
    
    # Scale by the distance reward scale
    distance_reward = mean_distance_reward * distance_reward_scale
    return distance_reward


@torch.jit.script
def compute_tilt_angle(object_rot: torch.Tensor):
    """Compute the tilt angle of the object in degrees."""
    # Extract cube's z-axis from quaternion
    w, x, y, z = object_rot[:, 0], object_rot[:, 1], object_rot[:, 2], object_rot[:, 3]
    cube_z_axis = torch.stack([
        2 * (x * y - w * z),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y)
    ], dim=-1)
    
    # Calculate angle between cube's z-axis and global z-axis
    global_z = torch.tensor([0.0, 0.0, 1.0], device=object_rot.device).repeat(object_rot.shape[0], 1)
    cos_angle = torch.sum(cube_z_axis * global_z, dim=-1)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    tilt_angle = torch.acos(cos_angle) * 180.0 / 3.14159  # Convert to degrees
    return tilt_angle


@torch.jit.script
def compute_tilt_penalty(object_rot: torch.Tensor, tilt_penalty: float):
    """Compute penalty for excessive tilt of the object."""
    tilt_angle = compute_tilt_angle(object_rot)
    
    # Apply tilt penalty if angle exceeds 60 degrees
    tilt_penalty_rew = torch.where(tilt_angle > 60.0, tilt_penalty, 0.0)
    return tilt_penalty_rew


@torch.jit.script
def compute_fall_penalty(object_pos: torch.Tensor, target_pos: torch.Tensor, fall_dist: float, fall_penalty: float):
    """Compute penalty when object falls too far from target position."""
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    fall_condition = goal_dist >= fall_dist
    fall_penalty_rew = torch.where(fall_condition, fall_penalty, 0.0)
    return fall_penalty_rew


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    total_rotations: torch.Tensor,
    # consecutive_successes: torch.Tensor,  # No success concept for spinning
    max_episode_length: float,
    prev_object_rot: torch.Tensor,
    object_rot: torch.Tensor,
    object_pos: torch.Tensor,
    target_pos: torch.Tensor,
    object_linvel: torch.Tensor,
    fingertip_pos: torch.Tensor,
    actions: torch.Tensor,
    action_penalty_scale: float,
    rotation_reward_scale: float,
    linear_velocity_penalty_scale: float,
    distance_reward_scale: float,
    contact_forces: torch.Tensor,
    # reach_goal_bonus: float,  # No goal success for spinning
    fall_dist: float,
    fall_penalty: float,
    tilt_penalty: float,
    av_factor: float,
):
    # Calculate clockwise rotation reward (includes contact checking)
    rotation_rew, rotation_delta = compute_clockwise_rotation_reward(prev_object_rot, object_rot, rotation_reward_scale, contact_forces)
    
    # Track total rotations (both positive and negative, contact checking is done in the function)
    total_rotations = total_rotations + rotation_delta
    
    # Compute individual reward components
    action_penalty_rew = compute_action_penalty(actions, action_penalty_scale)
    linear_velocity_penalty_rew = compute_linear_velocity_penalty(object_linvel, linear_velocity_penalty_scale)
    distance_reward_rew = compute_distance_reward(fingertip_pos, object_pos, distance_reward_scale)
    tilt_penalty_rew = compute_tilt_penalty(object_rot, tilt_penalty)
    fall_penalty_rew = compute_fall_penalty(object_pos, target_pos, fall_dist, fall_penalty)

    # Total reward is: clockwise rotation + action regularization + linear velocity penalty + distance reward + fall penalty + tilt penalty
    reward = rotation_rew + action_penalty_rew + linear_velocity_penalty_rew + distance_reward_rew + fall_penalty_rew + tilt_penalty_rew

    return reward, rotation_delta, reset_goal_buf, total_rotations 
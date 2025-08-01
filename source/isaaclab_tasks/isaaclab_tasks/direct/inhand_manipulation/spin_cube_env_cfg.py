# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the spin cube task."""

from __future__ import annotations

from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import (
    ShadowHandEnvCfg, 
    ShadowHandWithContactSensorsEnvCfg,
    ShadowHandWithBinaryContactSensorsEnvCfg,
    ShadowHandWithMagnitudeContactSensorsEnvCfg
)
from isaaclab.utils import configclass


@configclass
class SpinCubeEnvCfg(ShadowHandEnvCfg):
    """Configuration for the spin cube task - robot tries to spin cube around Z-axis as much as possible."""
    
    # Environment spaces (reduced - no object state observations)
    action_space = 20
    observation_space = 133  # Reduced: original 157 - 24 (object state) = 133
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # Reward scales (similar to original task)
    dist_reward_scale = -10.0  # Keep cube near center of hand (same as original)
    action_penalty_scale = -0.0002  # Action penalty (same as original)
    
    # Spin-specific parameters
    spin_axis = "z"  # Spin around Z-axis
    spin_reward_scale = 10.0  # Reward for spinning
    episode_length_s = 15.0  # Longer episodes for more spinning (15 seconds)
    
    # Success criteria
    min_rotations_for_success = 2.0  # Minimum full rotations to count as success
    success_bonus = 100.0  # Bonus for achieving success
    success_tolerance = 0.1  # Tolerance for success (in rotations)
    max_consecutive_success = 10  # Maximum consecutive successes before episode ends
    
    # Episode termination
    fall_dist = 0.24  # Still terminate if cube falls
    fall_penalty = -50.0  # Penalty for dropping cube


@configclass
class SpinCubeShadowHandWithContactSensorsEnvCfg(ShadowHandWithContactSensorsEnvCfg):
    """Shadow Hand configuration for spin cube task with full contact sensors."""
    
    # Environment spaces (reduced - no object state observations)
    action_space = 20
    observation_space = 211  # Reduced: original 235 - 24 (object state) = 211
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # Reward scales (similar to original task)
    dist_reward_scale = -10.0  # Keep cube near center of hand (same as original)
    action_penalty_scale = -0.0002  # Action penalty (same as original)
    
    # Spin-specific parameters
    spin_axis = "z"  # Spin around Z-axis
    spin_reward_scale = 10.0  # Reward for spinning
    episode_length_s = 15.0  # Longer episodes for more spinning (15 seconds)
    
    # Success criteria
    min_rotations_for_success = 2.0  # Minimum full rotations to count as success
    success_bonus = 100.0  # Bonus for achieving success
    
    # Episode termination
    fall_dist = 0.24  # Still terminate if cube falls
    fall_penalty = -50.0  # Penalty for dropping cube


@configclass
class SpinCubeShadowHandWithBinaryContactSensorsEnvCfg(ShadowHandWithBinaryContactSensorsEnvCfg):
    """Shadow Hand configuration for spin cube task with binary contact sensors."""
    
    # Environment spaces (reduced - no object state observations)
    action_space = 20
    observation_space = 159  # Reduced: original 183 - 24 (object state) = 159
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # Reward scales (similar to original task)
    dist_reward_scale = -10.0  # Keep cube near center of hand (same as original)
    action_penalty_scale = -0.0002  # Action penalty (same as original)
    
    # Spin-specific parameters
    spin_axis = "z"  # Spin around Z-axis
    spin_reward_scale = 10.0  # Reward for spinning
    episode_length_s = 15.0  # Longer episodes for more spinning (15 seconds)
    
    # Success criteria
    min_rotations_for_success = 2.0  # Minimum full rotations to count as success
    success_bonus = 100.0  # Bonus for achieving success
    success_tolerance = 0.1  # Tolerance for success (in rotations)
    max_consecutive_success = 10  # Maximum consecutive successes before episode ends
    
    # Episode termination
    fall_dist = 0.24  # Still terminate if cube falls
    fall_penalty = -50.0  # Penalty for dropping cube


@configclass
class SpinCubeShadowHandWithMagnitudeContactSensorsEnvCfg(ShadowHandWithMagnitudeContactSensorsEnvCfg):
    """Shadow Hand configuration for spin cube task with magnitude contact sensors."""
    
    # Environment spaces (reduced - no object state observations)
    action_space = 20
    observation_space = 159  # Reduced: original 183 - 24 (object state) = 159
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # Reward scales (similar to original task)
    dist_reward_scale = -10.0  # Keep cube near center of hand (same as original)
    action_penalty_scale = -0.0002  # Action penalty (same as original)
    
    # Spin-specific parameters
    spin_axis = "z"  # Spin around Z-axis
    spin_reward_scale = 10.0  # Reward for spinning
    episode_length_s = 15.0  # Longer episodes for more spinning (15 seconds)
    
    # Success criteria
    min_rotations_for_success = 2.0  # Minimum full rotations to count as success
    success_bonus = 100.0  # Bonus for achieving success
    success_tolerance = 0.1  # Tolerance for success (in rotations)
    max_consecutive_success = 10  # Maximum consecutive successes before episode ends
    
    # Episode termination
    fall_dist = 0.24  # Still terminate if cube falls
    fall_penalty = -50.0  # Penalty for dropping cube


@configclass
class SpinCubeShadowHandEnvCfg(ShadowHandEnvCfg):
    """Shadow Hand configuration for spin cube task."""
    
    # Environment spaces (reduced - no object state observations)
    action_space = 20
    observation_space = 133  # Reduced: original 157 - 24 (object state) = 133
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # Reward scales (similar to original task)
    dist_reward_scale = -10.0  # Keep cube near center of hand (same as original)
    action_penalty_scale = -0.0002  # Action penalty (same as original)
    
    # Spin-specific parameters
    spin_axis = "z"  # Spin around Z-axis
    spin_reward_scale = 10.0  # Reward for spinning
    episode_length_s = 15.0  # Longer episodes for more spinning (15 seconds)
    
    # Success criteria
    min_rotations_for_success = 2.0  # Minimum full rotations to count as success
    success_bonus = 100.0  # Bonus for achieving success
    
    # Episode termination
    fall_dist = 0.24  # Still terminate if cube falls
    fall_penalty = -50.0  # Penalty for dropping cube 
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import os
import importlib.util

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..', 'build'))
_MODULE_PATH = os.path.join(_BUILD_DIR, 'csim_env.cpython-39-x86_64-linux-gnu.so')

spec = importlib.util.spec_from_file_location("csim_env", _MODULE_PATH)
if spec is None:
    raise ImportError(f"Could not load module spec from {_MODULE_PATH}")
csim_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csim_env)

import time
from contextlib import contextmanager
from collections import defaultdict



class BatchedAgiSimEnv(py_environment.PyEnvironment):
    def __init__(self, sim_config_path, agi_param_dir, sim_base_dir, num_drones,
                 track_layout):
        super().__init__()
        print(f"Python Env: Initializing C++ AgiSimBatch with {num_drones} drones for per-drone reset...")
        self._num_drones = num_drones

        cpp_track_layout = [list(map(list, gate)) for gate in track_layout]

        self._env = csim_env.AgiSimBatch(
            num_drones,
            sim_config_path,
            agi_param_dir,
            sim_base_dir,
            cpp_track_layout
        )

        action_min = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        action_max = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=action_min, maximum=action_max, name='action'
        )

        self._shared_obs_dim = 19
        self._task_specific_obs_dim = 24
        self._raw_observation_dim = self._shared_obs_dim + self._task_specific_obs_dim # Should be 43

        # Observation spec uses float32 as this is typical for TF-Agents models
        self._observation_spec = {
            'shared_obs': array_spec.ArraySpec(
                shape=(self._shared_obs_dim,), dtype=np.float32, name='shared_observation'
            ),
            'task_specific_obs': array_spec.ArraySpec(
                shape=(self._task_specific_obs_dim,), dtype=np.float32, name='task_specific_racing_observation'
            ),
            'task_id': array_spec.ArraySpec(
                shape=(1,), dtype=np.int32, name='task_id'
            )
        }

        self._last_success_flags = np.zeros(self._num_drones, dtype=np.bool_)

        # Internal state for managing per-drone episodes
        self._episode_ended = np.zeros(self._num_drones, dtype=bool) # Still useful for external queries or if _split_observation uses it

        # Stores the raw observation (float64, from C++) for drones that have just been reset.
        # This will be used as S0 for their next FIRST step.
        self._next_obs_for_first_step_raw = np.empty((self._num_drones, self._raw_observation_dim), dtype=np.float64)
        # Flag: True if a drone has been reset and its next TimeStep should be FIRST.
        self._is_first_step_for_drone = np.zeros(self._num_drones, dtype=bool)

        # Initialize all drones (they all start with a FIRST step)
        self._perform_initial_batch_reset()

        self._timing_acc = defaultdict(float) # Keep if used
        self._timing_steps = 0 # Keep if used

    def get_last_success_flags(self):
        return self._last_success_flags

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._num_drones

    def \
            _split_observation(self, obs_batch_flat, task_id_batch):
        """Splits the flat observation batch into shared and task-specific parts."""
        obs_batch_flat_float32 = obs_batch_flat.astype(np.float32)
        shared_part = obs_batch_flat_float32[:, :self._shared_obs_dim]
        task_specific_part = obs_batch_flat_float32[:, self._shared_obs_dim:]
        return {
            'shared_obs': shared_part,
            'task_specific_obs': task_specific_part,
            'task_id': task_id_batch.astype(np.int32).reshape(-1, 1)
        }

    def _perform_initial_batch_reset(self):
        """Resets all drones in the C++ environment and stores their initial raw observations."""
        reset_all_mask = [True] * self._num_drones
        # C++ returns float64 observations
        all_initial_obs_raw = np.asarray(self._env.reset(reset_all_mask), dtype=np.float64)

        if all_initial_obs_raw.shape != (self._num_drones, self._raw_observation_dim):
            raise ValueError(f"C++ env.reset with all True returned unexpected shape. Expected {(self._num_drones, self._raw_observation_dim)}, got {all_initial_obs_raw.shape}")

        self._next_obs_for_first_step_raw = all_initial_obs_raw.copy()
        self._is_first_step_for_drone.fill(True)
        self._episode_ended.fill(False) # All episodes are starting

    def _reset(self):
        # This function no longer needs to add a 'success' key.
        self._perform_initial_batch_reset()
        initial_task_ids = np.zeros(self._num_drones, dtype=np.int32)
        structured_initial_observations = self._split_observation(self._next_obs_for_first_step_raw, initial_task_ids)
        return ts.restart(structured_initial_observations, batch_size=self._num_drones)

    def _step(self, action_batch: np.ndarray):
        action_batch_np = np.asarray(action_batch, dtype=self._action_spec.dtype) # Should be float32

        action_batch_np = np.asarray(action_batch, dtype=self._action_spec.dtype)
        step_results_list = self._env.step(action_batch_np)

        current_step_rewards_float32 = np.empty(self._num_drones, dtype=np.float32)
        current_step_dones = np.empty(self._num_drones, dtype=bool)
        s_prime_raw_observations_float64 = np.empty((self._num_drones, self._raw_observation_dim), dtype=np.float64)
        current_step_task_ids = np.empty(self._num_drones, dtype=np.int32)

        current_step_success_flags = np.empty(self._num_drones, dtype=np.bool_)
        for i, res in enumerate(step_results_list):
            s_prime_raw_observations_float64[i, :] = res.observation
            current_step_rewards_float32[i] = res.reward
            current_step_dones[i] = res.done
            current_step_success_flags[i] = res.success
            current_step_task_ids[i] = res.task_id


        self._last_success_flags = current_step_success_flags
        # --- Phase 3: Construct the TimeStep to return ---
        output_step_types = np.empty(self._num_drones, dtype=np.int32)
        # output_rewards will be current_step_rewards_float32
        output_discounts_float32 = np.empty(self._num_drones, dtype=np.float32)

        drones_to_reset_in_cpp_mask = np.zeros(self._num_drones, dtype=bool)

        for i in range(self._num_drones):
            if self._is_first_step_for_drone[i]:
                # Drone `i` was at S0. Action was A0. C++ produced S1, R1.
                # TimeStep is (FIRST, R1, discount, S1).
                output_step_types[i] = ts.StepType.FIRST

                if current_step_dones[i]: # Episode ended in one step (S0 -> A0 -> S1 is terminal)
                    output_discounts_float32[i] = 0.0
                    output_step_types[i] = ts.StepType.LAST # Override: S1 is terminal
                    drones_to_reset_in_cpp_mask[i] = True
                else: # Still ongoing
                    output_discounts_float32[i] = 1.0

                self._is_first_step_for_drone[i] = False

            else: # Drone `i` was in a MID step (St, At -> S(t+1), R(t+1))
                if current_step_dones[i]: # Episode ended
                    output_step_types[i] = ts.StepType.LAST
                    output_discounts_float32[i] = 0.0
                    drones_to_reset_in_cpp_mask[i] = True
                else: # Still MID
                    output_step_types[i] = ts.StepType.MID
                    output_discounts_float32[i] = 1.0

            self._episode_ended[i] = current_step_dones[i]

        # The observation part of the returned TimeStep is s_prime_raw_observations_float64,
        # which will be structured and converted to float32 by _split_observation.
        structured_s_prime_observations = self._split_observation(
            s_prime_raw_observations_float64,
            current_step_task_ids
        )

        # --- Phase 4: Perform resets in C++ for drones that just finished their episode ---
        if np.any(drones_to_reset_in_cpp_mask):
            # This C++ call resets the specified drones.
            # Assumed to return float64 observations for ALL drones,
            # where reset drones have new S0, and non-reset drones have their current s_prime values.
            all_obs_after_cpp_reset_raw_float64 = np.asarray(
                self._env.reset(drones_to_reset_in_cpp_mask.tolist()), dtype=np.float64
            )

            if all_obs_after_cpp_reset_raw_float64.shape != (self._num_drones, self._raw_observation_dim):
                raise ValueError(f"C++ env.reset with mask returned unexpected shape. Expected {(self._num_drones, self._raw_observation_dim)}, got {all_obs_after_cpp_reset_raw_float64.shape}")

            for i in range(self._num_drones):
                if drones_to_reset_in_cpp_mask[i]:
                    # Store the new initial raw observation (S0_new, float64) for drone `i`
                    self._next_obs_for_first_step_raw[i, :] = all_obs_after_cpp_reset_raw_float64[i, :].copy()
                    self._is_first_step_for_drone[i] = True

        return ts.TimeStep(
            output_step_types,
            current_step_rewards_float32, # Rewards are already float32
            output_discounts_float32,
            structured_s_prime_observations
        )

    def update_track_layout(self, new_track_layout_py):

        self._env.update_track_layout(new_track_layout_py)


    def getStates(self):
        return self._env.getStates()

    def setEval(self):
        return self._env.setEval()

    def get_successful_gate_pass_states_data_(self):
        return self._env.get_successful_gate_pass_states_data()

    def set_successful_gate_pass_states_data(self, data):
        return self._env.set_successful_gate_pass_states_data(data)

    def setNoiseIntensity(self, vio_pos_drift_std, vio_att_drift_std_deg, gate_reset_pos_std, gate_reset_att_std_deg):
        self._env.setNoiseIntensity(
            vio_pos_drift_std,
            vio_att_drift_std_deg,
            gate_reset_pos_std,
            gate_reset_att_std_deg
        )
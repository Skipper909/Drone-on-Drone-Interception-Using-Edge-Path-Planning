import numpy as np

from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts


class InterceptionTargetWrapper(wrappers.PyEnvironmentBaseWrapper):
    """
    Wrapper around BatchedAgiSimEnv for the interception task.

    It keeps the original observation_spec and action_spec, but overwrites
    part of `task_specific_obs` to encode a *synthetic moving target*:

      task_specific_obs[..., 0:3] = p_rel(t)   (target - drone, scaled coords)
      task_specific_obs[..., 6]   = ||p_rel(t)||  (distance)
      task_specific_obs[..., 7]   = ||v_rel||     (relative speed magnitude)

    Everything else in task_specific_obs is left untouched, so the TF-Agents
    specs remain consistent.

    We model p_rel(t) with a simple constant-velocity model in observation space:
        p_rel(t+dt) = p_rel(t) + v_rel * dt

    This is *synthetic* target motion purely for training and analysis; the
    underlying C++ env still thinks in terms of its own task, but the policy
    "sees" a moving target.
    """

    def __init__(
        self,
        env,
        dt: float = 0.02,
        rel_speed: float = 0.05,
        seed: int = 0,
    ):
        """
        Args:
            env: A BatchedAgiSimEnv instance.
            dt:  Simulation time step in seconds (RK4 dt ~ 0.02).
            rel_speed: Magnitude of synthetic target velocity in obs units.
            seed: RNG seed for sampling target direction per episode.
        """
        super().__init__(env)
        self._dt = float(dt)
        self._rel_speed = float(rel_speed)
        self._rng = np.random.RandomState(seed)

        # Per-episode state (set in _reset)
        self._p_rel = None  # shape [batch, 3]
        self._v_rel = None  # shape [batch, 3]

    # ------------------------------------------------------------------
    # Core methods we override: _reset and _step
    # All spec methods (observation_spec, action_spec, time_step_spec)
    # are provided by PyEnvironmentBaseWrapper and delegated to self._env.
    # ------------------------------------------------------------------

    def _reset(self):
        """Reset env and initialize synthetic target state."""
        base_ts = self._env.reset()
        return self._process_reset(base_ts)

    def _step(self, action):
        """Step env and update synthetic target state."""
        base_ts = self._env.step(action)
        return self._process_step(base_ts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_reset(self, base_ts: ts.TimeStep) -> ts.TimeStep:
        """Initialize target relative position and velocity at episode start."""
        obs = base_ts.observation

        if not isinstance(obs, dict):
            # If your env doesn't use a dict observation, don't wrap
            return base_ts

        if "task_specific_obs" not in obs:
            # Nothing to modify
            return base_ts

        task = obs["task_specific_obs"]
        # Expect shape [batch, D]
        task_np = np.array(task, copy=True)

        if task_np.ndim != 2 or task_np.shape[1] < 3:
            # Not enough dims to encode p_rel; bail out
            return base_ts

        batch_size = task_np.shape[0]

        # Use existing p_rel(0) from env as starting point if present
        # (first 3 dims of task_specific_obs).
        self._p_rel = task_np[:, 0:3].astype(np.float32)

        # Sample random directions for v_rel per drone
        dirs = self._rng.normal(size=(batch_size, 3)).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-6)
        self._v_rel = dirs * self._rel_speed  # [batch, 3]

        # Write initial synthetic p_rel into task_specific_obs
        task_np[:, 0:3] = self._p_rel

        # Optionally overwrite dist and rel_speed if we have enough dims
        if task_np.shape[1] >= 8:
            dists = np.linalg.norm(self._p_rel, axis=1)
            speeds = np.linalg.norm(self._v_rel, axis=1)
            task_np[:, 6] = dists
            task_np[:, 7] = speeds

        new_obs = dict(obs)
        new_obs["task_specific_obs"] = task_np

        return base_ts._replace(observation=new_obs)

    def _process_step(self, base_ts: ts.TimeStep) -> ts.TimeStep:
        """Advance p_rel with constant v_rel and update task_specific_obs."""
        # If for some reason we never initialized, just pass through
        if self._p_rel is None or self._v_rel is None:
            return base_ts

        obs = base_ts.observation
        if not isinstance(obs, dict) or "task_specific_obs" not in obs:
            return base_ts

        task = obs["task_specific_obs"]
        task_np = np.array(task, copy=True)

        if task_np.ndim != 2 or task_np.shape[1] < 3:
            return base_ts

        # Advance synthetic relative position: p_rel(t+dt) = p_rel(t) + v_rel * dt
        self._p_rel = self._p_rel + self._v_rel * self._dt

        # Overwrite first 3 dims with our updated p_rel
        task_np[:, 0:3] = self._p_rel

        # Update dist and rel_speed if dims allow
        if task_np.shape[1] >= 8:
            dists = np.linalg.norm(self._p_rel, axis=1)
            speeds = np.linalg.norm(self._v_rel, axis=1)
            task_np[:, 6] = dists
            task_np[:, 7] = speeds

        new_obs = dict(obs)
        new_obs["task_specific_obs"] = task_np

        return base_ts._replace(observation=new_obs)

# mtrl_trainer/target_motion.py
import numpy as np

class StraightLineTarget:
    """
    Simple constant-velocity target:
      p_t(0)  = start_pos
      v_t     = constant (m/s)
      p_t(k)  = p_t(0) + k * dt * v_t
    """

    def __init__(self,
                 start_pos,
                 velocity,
                 dt=0.02,
                 max_steps=1000):
        self.start_pos = np.asarray(start_pos, dtype=np.float32)
        self.velocity  = np.asarray(velocity, dtype=np.float32)
        self.dt        = dt
        self.max_steps = max_steps

        self.reset()

    def reset(self, start_pos=None, velocity=None):
        if start_pos is not None:
            self.start_pos = np.asarray(start_pos, dtype=np.float32)
        if velocity is not None:
            self.velocity = np.asarray(velocity, dtype=np.float32)
        self.t = 0
        self.pos = self.start_pos.copy()
        return self.pos.copy()

    def step(self):
        """Advance one time step, return new target position."""
        self.t += 1
        self.pos = self.start_pos + self.velocity * (self.t * self.dt)
        return self.pos.copy()

    def trajectory(self, num_steps=None):
        """Convenience for visualizer – return full trajectory array (T,3)."""
        if num_steps is None:
            num_steps = self.max_steps
        steps = np.arange(num_steps, dtype=np.float32).reshape(-1, 1)
        return self.start_pos[None, :] + steps * self.dt * self.velocity[None, :]

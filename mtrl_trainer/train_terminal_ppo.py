# train_terminal_ppo.py

import os
import time
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.summary import create_file_writer

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file
from mtrl_lib.obervation_encoders import SharedEncoderKeras

# ============================================================
# Paths & basic config
# ============================================================

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "mtrl_trainer", "parameters", "simulation.yaml")
AGI_PARAM_DIR   = os.path.join(_PROJECT_ROOT, "agilib", "params")
SIM_BASE_DIR    = os.path.join(_PROJECT_ROOT, "mtrl_trainer", "parameters")
TRACK_JSON      = os.path.join(SIM_BASE_DIR, "tracks", "interception_track.json")

SAVE_ROOT       = os.path.join(_CURRENT_DIR, "policies_terminal")
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# Training hyperparameters (copy from interception, adjust later)
# ============================================================

num_drones_train = 96
num_drones_eval  = 16

collect_steps_per_iteration = 800
num_epochs       = 10
num_iterations   = 350
eval_interval    = 50
log_interval     = 1
num_eval_episodes = 6

learning_rate = 8e-4
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=num_iterations * num_epochs,
    end_learning_rate=8e-5,
    power=1.8,
)

discount_factor = 0.99
gae_lambda      = 0.95
entropy_regularization = 0.02
importance_ratio_clipping = 0.2
gradient_clipping        = 0.3
value_pred_loss_coef     = 0.3

# ============================================================
# Helper: build env with terminal-pursuit task
# ============================================================

def build_training_env(task_id: int):
    track_layout_world = load_track_from_file(TRACK_JSON)

    py_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        track_layout_world=track_layout_world,
        num_drones_train=num_drones_train,
        num_drones_eval=num_drones_eval,
        task_id=task_id,                # <--- key change
    )
    return tf_py_environment.TFPyEnvironment(py_env)


# ============================================================
# Network definition (same as interception)
# ============================================================

class MTRLPolicyNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, name="MTRLPolicyNetwork"):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)

        self._shared_encoder = SharedEncoderKeras(
            shared_obs_size=observation_spec["shared_obs"].shape[-1],
            task_obs_size=observation_spec["task_specific_obs"].shape[-1],
            num_tasks=observation_spec["task_id"].shape[-1],
        )

        self._actor_dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(action_spec.shape[-1]),
            ]
        )

        self._value_dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, observations, step_type=None, network_state=(), training=False):
        shared = observations["shared_obs"]
        task   = observations["task_specific_obs"]
        task_id = observations["task_id"]

        encoded = self._shared_encoder([shared, task, task_id], training=training)
        logits  = self._actor_dense(encoded, training=training)
        value   = self._value_dense(encoded, training=training)

        return (logits, tf.squeeze(value, axis=-1)), network_state


# ============================================================
# Main training loop
# ============================================================

def main():
    train_env = build_training_env(task_id=1)  # 1 = TerminalPursuit
    eval_env  = build_training_env(task_id=1)

    observation_spec = train_env.observation_spec()
    action_spec      = train_env.action_spec()
    time_step_spec   = train_env.time_step_spec()

    policy_net = MTRLPolicyNetwork(observation_spec, action_spec)
    optimizer  = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = ppo_agent.PPOAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        optimizer=optimizer,
        actor_net=lambda obs, *args, **kwargs: policy_net(obs, *args, **kwargs)[0],
        value_net=lambda obs, *args, **kwargs: (policy_net(obs, *args, **kwargs)[1], ()),
        num_epochs=num_epochs,
        use_gae=True,
        lambda_value=gae_lambda,
        discount_factor=discount_factor,
        entropy_regularization=entropy_regularization,
        importance_ratio_clipping=importance_ratio_clipping,
        gradient_clipping=gradient_clipping,
        value_pred_loss_coef=value_pred_loss_coef,
        train_step_counter=global_step,
    )
    agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    policy_dir = os.path.join(SAVE_ROOT, "best_terminal")
    os.makedirs(policy_dir, exist_ok=True)
    saver = policy_saver.PolicySaver(agent.policy)

    summary_writer = create_file_writer(os.path.join(SAVE_ROOT, "logs"))
    summary_writer.set_as_default()

    def collect_step(env, policy, buffer):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)

    replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=collect_steps_per_iteration,
    )

    best_eval_return = -1e9

    for iteration in range(num_iterations):
        # Data collection
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience).loss
        replay_buffer.clear()

        if iteration % log_interval == 0:
            tf.summary.scalar("train_loss", train_loss, step=global_step)
            print(f"Iter {iteration}, Loss {train_loss:.3f}")

        if (iteration + 1) % eval_interval == 0:
            results = metric_utils.eager_compute(
                train_metrics,
                eval_env,
                agent.policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=summary_writer,
                summary_prefix="Eval",
            )
            avg_return = results[train_metrics[2].name]
            print(f"--- EVAL Iter {iteration+1}, AvgReturn={avg_return:.3f} ---")

            if avg_return > best_eval_return:
                best_eval_return = avg_return
                saver.save(policy_dir)
                print(f"Saved new best terminal policy to {policy_dir}")

    print("Training complete. Best eval return:", best_eval_return)


if __name__ == "__main__":
    main()

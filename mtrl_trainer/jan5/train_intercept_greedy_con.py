#!/usr/bin/env python3
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
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv
from mtrl_lib.gate_utils import load_track_from_file
from mtrl_lib.obervation_encoders import SharedEncoderKeras

# ============================================================
# Paths & basic config
# ============================================================

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR   = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR    = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

REWARD_YAML_PATH = os.path.join(SIM_BASE_DIR, "reward_params.yaml")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# NOTE: These output dirs are relative to your *launch directory* (os.getcwd()).
# If you want them fixed relative to repo, change os.getcwd() to _PROJECT_ROOT or _CURRENT_DIR.
log_dir    = os.path.join(os.getcwd(), 'logs', f'ppo_intercept_{current_time}')
policy_dir = os.path.join(os.getcwd(), 'policies_intercept')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(policy_dir, exist_ok=True)

BEST_POLICY_PATH = os.path.join(policy_dir, 'best_intercept')

# ============================================================
# Warm-start config
# ============================================================

# If True: on startup, attempt to load 'BEST_POLICY_PATH' and copy its variables into the new agent.policy,
# then continue training from there.
WARMSTART_FROM_BEST = True

# ============================================================
# Hyperparameters
# ============================================================

actor_fc_layers = (256, 256)
value_fc_layers = (256, 256)

num_drones_train = 96
num_drones_eval  = 16
collect_steps_per_iteration = 800

num_epochs = 10
num_iterations = 1850
eval_interval = 50
log_interval = 1
num_eval_episodes = 6

learning_rate = 8e-4
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=num_iterations * num_epochs,
    end_learning_rate=8e-5,
    power=1.8
)

discount_factor = 0.99
gae_lambda = 0.95
entropy_regularization = 0.02
importance_ratio_clipping = 0.2
gradient_clipping = 0.3
value_pred_loss_coef = 0.3

# ============================================================
# Greedy eval config (IMPORTANT)
# ============================================================

# If True => greedy eval uses setEval() (deterministic target reset).
# If False => greedy eval runs WITHOUT setEval() (randomized resets like training).
GREEDY_EVAL_DETERMINISTIC = False

# Must match C++ POS_SCALE in TaskInterceptionMoving
REL_POS_SCALE_M = 300.0

# ============================================================
# Simple reward_params.yaml parsing (no pyyaml)
# ============================================================

def _read_simple_yaml_value(path: str, key: str, fallback: float) -> float:
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if not line.startswith(key):
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                return float(parts[1].strip())
    except Exception:
        return fallback
    return fallback


def load_capture_radius_m():
    # Default fallback intentionally conservative; you WANT this set in reward_params.yaml.
    return _read_simple_yaml_value(REWARD_YAML_PATH, "INTERCEPTION_CAPTURE_RADIUS", 2.0)

# ============================================================
# TanhNormal distribution
# ============================================================

class TanhNormal(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, bijector=None, name='TanhNormalDistribution'):
        if bijector is None:
            bijector = tfp.bijectors.Tanh()
        parameters = dict(locals())
        super(TanhNormal, self).__init__(distribution=distribution, bijector=bijector, name=name)
        self._parameters = parameters

    def mode(self, **kwargs):
        return self._bijector.forward(self.distribution.mode(**kwargs))

    def entropy(self, **kwargs):
        return self.distribution.entropy(**kwargs)

    def mean(self, **kwargs):
        return self._bijector.forward(self.distribution.mean(**kwargs))

    def stddev(self, **kwargs):
        return self.distribution.stddev(**kwargs)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            distribution=tfp.util.ParameterProperties(
                event_ndims=lambda self: self.distribution.event_shape.rank),
            bijector=tfp.util.ParameterProperties(is_tensor=False)
        )

# ============================================================
# Encoders & Networks
# ============================================================

class TaskSpecificInterceptionEncoderKeras(tf.keras.Model):
    def __init__(self, output_size=32, name='TaskSpecificInterceptionEncoder'):
        super().__init__(name=name)
        self._dense1 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self._dense2 = tf.keras.layers.Dense(output_size, activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self._dense1(x)
        x = self._dense2(x)
        return x


class InterceptionActorNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 shared_encoder,
                 task_encoder,
                 actor_fc_layers=(256, 256),
                 activation_fn=tf.nn.leaky_relu,
                 name='InterceptionActorNetwork'):
        super().__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._shared_encoder = shared_encoder
        self._task_encoder = task_encoder
        self._combiner = tf.keras.layers.Concatenate(axis=-1)

        self._actor_layers = [tf.keras.layers.Dense(units, activation=activation_fn)
                              for units in actor_fc_layers]

        num_actions = action_spec.shape.num_elements()
        self._projection_layer = tf.keras.layers.Dense(num_actions * 2, activation=None)

    def call(self, observation, step_type=None, network_state=(), training=False, outer_rank=1):
        shared_obs = observation['shared_obs']
        task_obs = observation['task_specific_obs']

        batch_squash = network_utils.BatchSquash(outer_rank)
        shared_obs_squashed = batch_squash.flatten(shared_obs)
        task_obs_squashed = batch_squash.flatten(task_obs)

        shared_embedding = self._shared_encoder(shared_obs_squashed)
        task_embedding = self._task_encoder(task_obs_squashed)

        x = self._combiner([shared_embedding, task_embedding])
        for layer in self._actor_layers:
            x = layer(x)

        projection = self._projection_layer(x)
        mean, log_std = tf.split(projection, 2, axis=-1)

        std = tf.math.softplus(log_std)
        mean = tf.nn.tanh(mean)
        std = tf.clip_by_value(std, 1e-6, 1.0)

        mean = batch_squash.unflatten(mean)
        std = batch_squash.unflatten(std)

        base_dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        action_distribution = TanhNormal(distribution=base_dist, bijector=tfp.bijectors.Tanh())
        return action_distribution, network_state


class InterceptionValueNetwork(network.Network):
    def __init__(self, input_tensor_spec, fc_layer_params=(256, 256), name='InterceptionValueNetwork'):
        super().__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._combiner = tf.keras.layers.Concatenate(axis=-1)
        self._layers = [tf.keras.layers.Dense(params, activation=tf.nn.leaky_relu)
                        for params in fc_layer_params]
        self._layers.append(tf.keras.layers.Dense(1, name='value_head'))

    def call(self, observation, step_type=None, network_state=(), training=False, outer_rank=1):
        shared_obs = observation['shared_obs']
        task_obs = observation['task_specific_obs']

        batch_squash = network_utils.BatchSquash(outer_rank)
        shared_obs_squashed = batch_squash.flatten(shared_obs)
        task_obs_squashed = batch_squash.flatten(task_obs)

        full_obs = self._combiner([shared_obs_squashed, task_obs_squashed])

        x = full_obs
        for layer in self._layers:
            x = layer(x)

        x = batch_squash.unflatten(x)
        return tf.squeeze(x, axis=-1), network_state

# ============================================================
# Warm-start from best_intercept SavedModel
# ============================================================

def warmstart_from_saved_policy(agent, train_step_counter: tf.Variable, saved_policy_dir: str) -> bool:
    """
    Best-effort warm start: copy SavedModel policy variables into current agent.policy vars.

    Notes:
      - This primarily restores the *policy/actor* weights.
      - PPO optimizer state and value network weights are NOT restored from PolicySaver export.
    """
    if not tf.io.gfile.exists(saved_policy_dir):
        print(f"[warmstart] No saved policy dir found at: {saved_policy_dir}")
        return False

    try:
        sm = tf.saved_model.load(saved_policy_dir)
    except Exception as e:
        print(f"[warmstart] Failed to load SavedModel from {saved_policy_dir}: {e}")
        return False

    if not hasattr(sm, "model_variables"):
        print("[warmstart] Loaded SavedModel has no model_variables attribute; cannot copy weights.")
        return False

    src = {v.name.split(":")[0]: v for v in sm.model_variables}

    dst_vars = list(agent.policy.variables())
    dst = {v.name.split(":")[0]: v for v in dst_vars}

    matched = 0
    for name, dv in dst.items():
        sv = src.get(name, None)
        if sv is None:
            continue
        if tuple(sv.shape) != tuple(dv.shape):
            continue
        try:
            dv.assign(sv)
            matched += 1
        except Exception:
            pass

    # If train step exists in the SavedModel, copy it (optional).
    try:
        if hasattr(sm, "get_train_step"):
            ts = sm.get_train_step()
            ts_val = int(ts.numpy()) if hasattr(ts, "numpy") else int(ts)
            train_step_counter.assign(ts_val)
            print(f"[warmstart] train_step_counter set to {ts_val}")
    except Exception as e:
        print(f"[warmstart] Could not restore train step (ok): {e}")

    print(f"[warmstart] Copied {matched}/{len(dst_vars)} policy variables from SavedModel.")
    return matched > 0


# ============================================================
# Greedy eval helper (uses p_rel to compute distance)
# ============================================================

def greedy_eval_capture_stats(eval_env, policy, capture_radius_m: float, num_episodes: int):
    """
    Evaluate with deterministic (greedy) actions.
    Capture if min_dist <= capture_radius_m, where dist is derived from p_rel.
    """
    greedy_pol = greedy_policy.GreedyPolicy(policy)

    cap_count = 0
    rets = []
    steps_done = []
    steps_to_cap = []
    min_dists = []

    for _ in range(num_episodes):
        ts = eval_env.reset()
        st = greedy_pol.get_initial_state(eval_env.batch_size)

        ep_ret = 0.0
        ep_min_dist = float("inf")
        cap_step = None

        step = 0
        while True:
            obs = ts.observation
            task_obs = obs["task_specific_obs"].numpy()  # (B,K)
            p_rel_scaled = task_obs[0, 0:3].astype(np.float32)
            p_rel_world = p_rel_scaled * REL_POS_SCALE_M
            dist = float(np.linalg.norm(p_rel_world))

            if dist < ep_min_dist:
                ep_min_dist = dist
            if cap_step is None and dist <= capture_radius_m:
                cap_step = step

            ep_ret += float(ts.reward.numpy()[0])

            if ts.is_last():
                break

            action_step = greedy_pol.action(ts, st)
            st = action_step.state
            ts = eval_env.step(action_step.action)
            step += 1

        if cap_step is not None:
            cap_count += 1
            steps_to_cap.append(cap_step)

        rets.append(ep_ret)
        steps_done.append(step)
        min_dists.append(ep_min_dist)

    cap_rate = 100.0 * cap_count / max(1, num_episodes)
    return {
        "cap_rate": cap_rate,
        "avg_ret": float(np.mean(rets)) if rets else 0.0,
        "avg_steps_done": float(np.mean(steps_done)) if steps_done else 0.0,
        "avg_steps_to_cap": float(np.mean(steps_to_cap)) if steps_to_cap else float("nan"),
        "avg_min_dist": float(np.mean(min_dists)) if min_dists else float("inf"),
    }

# ============================================================
# Training / Evaluation
# ============================================================

def train_eval():
    capture_radius_m = load_capture_radius_m()
    print(f"[train] reward_params.yaml INTERCEPTION_CAPTURE_RADIUS = {capture_radius_m} m")
    print(f"[train] best policy path: {BEST_POLICY_PATH}")
    print(f"[paths] log_dir={log_dir}")
    print(f"[paths] policy_dir={policy_dir}")

    best_eval_avg_return = -np.inf

    summary_writer = create_file_writer(log_dir, flush_millis=10000)
    summary_writer.set_as_default()

    track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    print("Creating environments...")

    base_py_train_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_train,
        track_layout=track_layout
    )

    base_py_eval_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_eval,
        track_layout=track_layout
    )

    train_env = tf_py_environment.TFPyEnvironment(base_py_train_env)
    eval_env  = tf_py_environment.TFPyEnvironment(base_py_eval_env)

    if hasattr(base_py_eval_env, "setEval"):
        base_py_eval_env.setEval()

    greedy_py_eval_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=1,
        track_layout=track_layout
    )
    if GREEDY_EVAL_DETERMINISTIC and hasattr(greedy_py_eval_env, "setEval"):
        greedy_py_eval_env.setEval()
        print("[train] Greedy eval uses setEval() => deterministic")
    else:
        print("[train] Greedy eval does NOT call setEval() => randomized like training")
    greedy_eval_env = tf_py_environment.TFPyEnvironment(greedy_py_eval_env)

    print("Environments created.")

    shared_encoder = SharedEncoderKeras(output_size=32, name='SharedEncoder')
    task_encoder_interception = TaskSpecificInterceptionEncoderKeras(output_size=32)

    actor_net = InterceptionActorNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        shared_encoder=shared_encoder,
        task_encoder=task_encoder_interception,
        actor_fc_layers=actor_fc_layers,
        activation_fn=tf.nn.leaky_relu
    )

    value_net = InterceptionValueNetwork(
        input_tensor_spec=train_env.observation_spec(),
        fc_layer_params=value_fc_layers
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    train_step_counter = tf.Variable(0, dtype=tf.int64, name='train_step_counter')

    agent = ppo_agent.PPOAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=num_epochs,
        gradient_clipping=gradient_clipping,
        entropy_regularization=entropy_regularization,
        importance_ratio_clipping=importance_ratio_clipping,
        discount_factor=discount_factor,
        use_gae=True,
        lambda_value=gae_lambda,
        value_pred_loss_coef=value_pred_loss_coef,
        normalize_observations=True,
        normalize_rewards=True,
        train_step_counter=train_step_counter,
        debug_summaries=True,
        summarize_grads_and_vars=True
    )
    agent.initialize()
    print("Agent created and initialized.")

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    agent.train = common.function(agent.train)
    agent.collect_policy.action = common.function(agent.collect_policy.action)

    collect_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
    ]

    print("Starting training loop...")

    time_step = train_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)

    # Build policy variables (safe for Keras lazy creation)
    _ = agent.collect_policy.action(time_step, policy_state)

    # Warm-start from last exported best policy (if present)
    if WARMSTART_FROM_BEST:
        warmstart_from_saved_policy(agent, train_step_counter, BEST_POLICY_PATH)

    iteration_times = []

    try:
        for i in range(num_iterations):
            iteration_start_time = time.time()

            step_before = int(agent.train_step_counter.numpy())
            print(f"\n--- Iteration {i}, Global Step {step_before} ---")

            collection_start_time = time.time()

            all_step_types = []
            all_observations = []
            all_actions = []
            all_policy_infos = []
            all_next_step_types = []
            all_rewards = []
            all_discounts = []

            for _ in range(collect_steps_per_iteration):
                action_step = agent.collect_policy.action(time_step, policy_state)
                next_time_step = train_env.step(action_step.action)
                current_traj = trajectory.from_transition(time_step, action_step, next_time_step)

                all_step_types.append(current_traj.step_type)
                all_observations.append(current_traj.observation)
                all_actions.append(current_traj.action)
                all_policy_infos.append(current_traj.policy_info)
                all_next_step_types.append(current_traj.next_step_type)
                all_rewards.append(current_traj.reward)
                all_discounts.append(current_traj.discount)

                for metric in collect_metrics:
                    metric(current_traj)

                time_step = next_time_step
                policy_state = action_step.state

            collection_duration = time.time() - collection_start_time
            print(f"Collection finished in {collection_duration:.3f}s")

            stacked_observations = tf.nest.map_structure(lambda *x: tf.stack(x, axis=1), *all_observations)
            stacked_policy_infos = tf.nest.map_structure(lambda *x: tf.stack(x, axis=1), *all_policy_infos)

            experience_for_train = trajectory.Trajectory(
                step_type=tf.stack(all_step_types, axis=1),
                observation=stacked_observations,
                action=tf.stack(all_actions, axis=1),
                policy_info=stacked_policy_infos,
                next_step_type=tf.stack(all_next_step_types, axis=1),
                reward=tf.stack(all_rewards, axis=1),
                discount=tf.stack(all_discounts, axis=1)
            )

            train_call_start_time = time.time()
            loss_info = agent.train(experience=experience_for_train)
            train_call_duration = time.time() - train_call_start_time

            # IMPORTANT: Use the *post-train* step for logging/eval so eval fires exactly on multiples.
            current_step = int(agent.train_step_counter.numpy())

            iteration_total_time = time.time() - iteration_start_time
            iteration_times.append(iteration_total_time)
            avg_iter_time_smooth = np.mean(iteration_times[-10:])
            actual_steps_collected_this_iter = collect_steps_per_iteration * train_env.batch_size

            iter_steps_per_sec = actual_steps_collected_this_iter / iteration_total_time if iteration_total_time > 0 else 0
            steps_per_sec_smooth = actual_steps_collected_this_iter / avg_iter_time_smooth if avg_iter_time_smooth > 0 else 0

            print(
                f"LOG Step={current_step}, Loss={float(loss_info.loss):.4f}, "
                f"Steps/sec={iter_steps_per_sec:.2f} (Smoothed: {steps_per_sec_smooth:.2f}), "
                f"IterTime={iteration_total_time:.3f}s (Collect: {collection_duration:.3f}s, Train: {train_call_duration:.3f}s)"
            )

            for metric in collect_metrics:
                metric.reset()

            if current_step > 0 and current_step % eval_interval == 0:
                eval_step = current_step

                eval_results = metric_utils.eager_compute(
                    metrics=[tf_metrics.AverageReturnMetric(batch_size=eval_env.batch_size)],
                    environment=eval_env,
                    policy=agent.policy,
                    num_episodes=num_eval_episodes,
                    train_step=tf.convert_to_tensor(eval_step, dtype=tf.int64),
                    summary_writer=summary_writer,
                    summary_prefix='Eval',
                )
                avg_return = eval_results['AverageReturn']
                print(f'-------- EVALUATION (TF-Agents, stochastic) Step={eval_step}, AvgReturn={avg_return.numpy():.4f} --------')

                stats = greedy_eval_capture_stats(
                    greedy_eval_env,
                    agent.policy,
                    capture_radius_m=capture_radius_m,
                    num_episodes=20
                )
                print(
                    f"-------- GREEDY EVAL Step={eval_step} | "
                    f"cap_rate={stats['cap_rate']:.1f}% | avg_ret={stats['avg_ret']:.2f} | "
                    f"avg_steps_done={stats['avg_steps_done']:.1f} | avg_steps_to_cap={stats['avg_steps_to_cap']} | "
                    f"avg_min_dist={stats['avg_min_dist']:.2f} m --------"
                )

                if avg_return > best_eval_avg_return:
                    best_eval_avg_return = avg_return
                    print(f"[train] New best avg_return: {best_eval_avg_return.numpy():.4f} => saving policy to {BEST_POLICY_PATH}")
                    tf_policy_saver.save(BEST_POLICY_PATH)

    except KeyboardInterrupt:
        step_now = int(agent.train_step_counter.numpy())
        print(f"\n[train] Caught KeyboardInterrupt at step={step_now}. Exporting policy to {BEST_POLICY_PATH} ...")
        try:
            tf_policy_saver.save(BEST_POLICY_PATH)
            print(f"[train] Policy exported to: {BEST_POLICY_PATH}")
        except Exception as e:
            print(f"[train] WARNING: Failed to export policy: {e}")
        return

    print("Training finished.")
    tf_policy_saver.save(BEST_POLICY_PATH)
    print(f"[train] Final policy saved to: {BEST_POLICY_PATH}")


if __name__ == '__main__':
    train_eval()

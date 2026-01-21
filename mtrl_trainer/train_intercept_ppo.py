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
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(os.getcwd(), 'logs', f'ppo_intercept_{current_time}')
policy_dir = os.path.join(os.getcwd(), 'policies')
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints_intercept')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(policy_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================

actor_fc_layers = (256, 256)
value_fc_layers = (256, 256)

num_drones_train = 96
num_drones_eval = 16
collect_steps_per_iteration = 800

num_epochs = 10
num_iterations = 850 #550
eval_interval = 50
log_interval = 1
num_eval_episodes = 6

learning_rate = 8e-4 #stable 8e-4
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=num_iterations * num_epochs,
    end_learning_rate=8e-5 , #Stable 8e-5
    power=1.8
)

discount_factor = 0.99
gae_lambda = 0.95
entropy_regularization = 0.02 #stable 0.02
importance_ratio_clipping = 0.2
gradient_clipping = 0.3
value_pred_loss_coef = 0.3

# ============================================================
# TanhNormal distribution
# ============================================================

class TanhNormal(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, bijector=None, name='TanhNormalDistribution'):
        if bijector is None:
            bijector = tfp.bijectors.Tanh()
        parameters = dict(locals())
        super(TanhNormal, self).__init__(
            distribution=distribution,
            bijector=bijector,
            name=name)
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
    """
    Encoder for interception task-specific observation (p_rel, v_rel, dist, rel_speed, ...).
    It will adapt to whatever length the env exposes as 'task_specific_obs'.
    """
    def __init__(self, output_size=32, name='TaskSpecificInterceptionEncoder'):
        super().__init__(name=name)
        self._dense1 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self._dense2 = tf.keras.layers.Dense(output_size, activation=tf.nn.leaky_relu)

    def call(self, x):
        x = self._dense1(x)
        x = self._dense2(x)
        return x


class InterceptionActorNetwork(network.Network):
    """
    Single-task actor network for interception.

    - Uses SharedEncoderKeras for shared_obs
    - Uses TaskSpecificInterceptionEncoderKeras for task_specific_obs
    - Ignores 'task_id' in the observation dict
    - Outputs a TanhNormal distribution over continuous actions
    """
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 shared_encoder,
                 task_encoder,
                 actor_fc_layers=(256, 256),
                 activation_fn=tf.nn.leaky_relu,
                 name='InterceptionActorNetwork'):
        super(InterceptionActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._shared_encoder = shared_encoder
        self._task_encoder = task_encoder
        self._combiner = tf.keras.layers.Concatenate(axis=-1)

        self._actor_layers = [
            tf.keras.layers.Dense(units, activation=activation_fn)
            for units in actor_fc_layers
        ]

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
    """
    Single-head critic. Concatenates shared_obs and task_specific_obs and
    predicts scalar value.
    """
    def __init__(self, input_tensor_spec, fc_layer_params=(256, 256), name='InterceptionValueNetwork'):
        super(InterceptionValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._combiner = tf.keras.layers.Concatenate(axis=-1)
        self._layers = [
            tf.keras.layers.Dense(params, activation=tf.nn.leaky_relu)
            for params in fc_layer_params
        ]
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
# Training / Evaluation
# ============================================================

def train_eval():
    best_eval_avg_return = -np.inf

    summary_writer = create_file_writer(log_dir, flush_millis=10000)
    summary_writer.set_as_default()

    final_track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    print("Creating environments...")
    final_track_layout = load_track_from_file(os.path.join(SIM_BASE_DIR, "track.json"))

    # Base envs (pure C++ dynamics)
    base_py_train_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_train,
        track_layout=final_track_layout
    )

    base_py_eval_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_eval,
        track_layout=final_track_layout
    )

    # C++ TaskInterceptionMoving already injects moving-target info
    # into task_specific_obs (p_rel, v_rel, dist, |v_rel|, ...).
    # Use the bare envs so the policy trains on that directly.
    py_train_env = base_py_train_env
    py_eval_env  = base_py_eval_env

    train_env = tf_py_environment.TFPyEnvironment(py_train_env)
    eval_env  = tf_py_environment.TFPyEnvironment(py_eval_env)
    print("Environments created.")

    # Put eval in deterministic/eval mode if supported
    if hasattr(py_eval_env, "setEval"):
        py_eval_env.setEval()
    elif hasattr(base_py_eval_env, "setEval"):
        base_py_eval_env.setEval()


    # Encoders
    shared_encoder = SharedEncoderKeras(output_size=32, name='SharedEncoder')
    task_encoder_interception = TaskSpecificInterceptionEncoderKeras(
        output_size=32,
        name='TaskSpecificInterceptionEncoder'
    )

    # Networks
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

    # Checkpointing (saving only; loading disabled for clean start)
    network_checkpointer = tf.train.Checkpoint(
        agent=agent,
        optimizer=optimizer,
        train_step=train_step_counter
    )
    checkpoint_manager = tf.train.CheckpointManager(
        network_checkpointer,
        directory=checkpoint_dir,
        max_to_keep=5,
        checkpoint_name="ckpt"
    )

    # ------ DISABLE CHECKPOINT RESTORE ------
    print("Checkpoint loading disabled: starting training from scratch every run.")

    collect_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
    ]

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    # JIT-compile for speed
    agent.train = common.function(agent.train)
    agent.collect_policy.action = common.function(agent.collect_policy.action)

    print("Starting training loop...")
    time_step = train_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    iteration_times = []

    for i in range(num_iterations):
        iteration_start_time = time.time()
        current_step = agent.train_step_counter.numpy()
        print(f"\n--- Iteration {i}, Global Step {current_step} ---")

        collection_start_time = time.time()

        all_step_types = []
        all_observations = []
        all_actions = []
        all_policy_infos = []
        all_next_step_types = []
        all_rewards = []
        all_discounts = []

        # Collect trajectories
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

        experience_for_train = None
        total_loss_tensor = tf.constant(0.0, dtype=tf.float32)
        train_call_duration = 0.0

        try:
            # Stack along time axis -> [B, T, ...]
            stacked_observations = tf.nest.map_structure(
                lambda *x: tf.stack(x, axis=1),
                *all_observations
            )
            stacked_policy_infos = tf.nest.map_structure(
                lambda *x: tf.stack(x, axis=1),
                *all_policy_infos
            )

            experience_for_train = trajectory.Trajectory(
                step_type=tf.stack(all_step_types, axis=1),
                observation=stacked_observations,
                action=tf.stack(all_actions, axis=1),
                policy_info=stacked_policy_infos,
                next_step_type=tf.stack(all_next_step_types, axis=1),
                reward=tf.stack(all_rewards, axis=1),
                discount=tf.stack(all_discounts, axis=1)
            )

            if hasattr(experience_for_train, 'step_type') and tf.size(experience_for_train.step_type) > 0:
                train_call_start_time = time.time()
                loss_info = agent.train(experience=experience_for_train)
                train_call_duration = time.time() - train_call_start_time
                total_loss_tensor = tf.cast(loss_info.loss, dtype=tf.float32)
            else:
                print(f"Iter {current_step}: No trajectories stacked, skipping training.")

        except Exception as e:
            print(f"Exception during stacking or training at global step {current_step}: {e}")
            raise

        iteration_total_time = time.time() - iteration_start_time
        iteration_times.append(iteration_total_time)
        avg_iter_time_smooth = np.mean(iteration_times[-10:])
        actual_steps_collected_this_iter = collect_steps_per_iteration * train_env.batch_size

        iter_steps_per_sec = actual_steps_collected_this_iter / iteration_total_time if iteration_total_time > 0 else 0
        steps_per_sec_smooth = actual_steps_collected_this_iter / avg_iter_time_smooth if avg_iter_time_smooth > 0 else 0

        print(
            f"LOG Step={current_step}, Loss={total_loss_tensor.numpy():.4f}, "
            f"Steps/sec={iter_steps_per_sec:.2f} (Smoothed: {steps_per_sec_smooth:.2f}), "
            f"IterTime={iteration_total_time:.3f}s (Collect: {collection_duration:.3f}s, Train: {train_call_duration:.3f}s)"
        )

        log_dict = {metric.name: metric.result() for metric in collect_metrics}

        with summary_writer.as_default(step=current_step):
            tf.summary.scalar('Loss/TotalLoss', total_loss_tensor, step=current_step)
            for key, value in log_dict.items():
                metric_val = value.numpy() if hasattr(value, 'numpy') else value
                tf.summary.scalar(f'Metrics/{key}', metric_val, step=current_step)

        for metric in collect_metrics:
            metric.reset()

        # Periodic evaluation
        if current_step > 0 and current_step % eval_interval == 0:
            eval_start_time = time.time()
            eval_step = agent.train_step_counter.numpy()
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
            eval_duration = time.time() - eval_start_time
            print(
                f'-------- EVALUATION (INTERCEPTION) Step={eval_step}, '
                f'AvgReturn={avg_return.numpy():.4f} ({eval_duration:.2f} sec) --------'
            )

            if avg_return > best_eval_avg_return:
                best_eval_avg_return = avg_return
                policy_artifact_dir = os.path.join(policy_dir, f'ppo_policy_{avg_return}')
                tf_policy_saver.save(policy_artifact_dir)
                checkpoint_manager.save(checkpoint_number=train_step_counter)

    print("Training finished.")
    tf_policy_saver.save(os.path.join(policy_dir, 'best_intercept'))


if __name__ == '__main__':
    train_eval()

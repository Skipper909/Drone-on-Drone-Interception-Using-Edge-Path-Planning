# train_ppo.py
import os
import time
import numpy as np
import tensorflow as tf
import sys
import datetime
from tensorflow.summary import create_file_writer
from functools import partial
import json
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network, encoding_network
from tf_agents.policies import policy_saver
# from tf_agents.replay_buffers import tf_uniform_replay_buffer # No longer used for training data
from tf_agents.utils import common
from tf_agents.trajectories import trajectory, time_step as ts_lib
from tf_agents.networks import network   # For base Network class
from tf_agents.specs import tensor_spec
from tf_agents.networks import normal_projection_network # <-- AND THIS
from tf_agents.networks import utils as network_utils

from mtrl_lib.agisim_environment_MTRL import BatchedAgiSimEnv

from mtrl_lib.gate_utils import load_track_from_file
from mtrl_lib.obervation_encoders import TaskSpecificRacingEncoderKeras, TaskSpecificStabilizationEncoderKeras, \
    SharedEncoderKeras

"""
Usage Notes:

1.  Iteration 0 will take a lot longer than subsequent iterations because a lot of things are initialized.
    For some reason the next few iterations until around iteration 100 will also take around 4 times longer
    than all others. After that, iteration time should settle at a lower value until the end. Use only this
    final value to make assumptions about the expected total runtime.

2.  Depending on hyperparameters and system architecture each iteration takes around 50-80% in 
    the collection phase which uses the C++ environment and can as of now only run on the CPU.
    The rest of the time is spent on training with tensorflow which can be done
    on the GPU if the tf installation is configured correctly. This can speed
    up the process by up to 100%. However, on some laptops this may actually slow
    it down.
    
3.  num_drones controls how many drones it simulates in parallel AND how many
    treads it uses. A multiple of your CPU threads is advised but note that the
    other parameters are tuned for 96 per default. Having less drones might mean
    its necessary to run more epochs and vice versa.

4.  The script automatically saves the best-performing policies
    with the respective evaluation reward as a floating point number like this 
    ppo_policy_-4.675509452819. The one with the highest number is usually the
    one we want to transfer to the ROS environment. For debugging purposes it 
    also saves a policy at every single log step like this: ppo_policy_190, 
    these get overwritten.
    
5.  The warnings you see at every log interval are related to the saving and
    can be ignored
"""

# --- Force CPU-Only Operation ---
# print("Attempting to configure TensorFlow for CPU-only operation...")
# try:
#    tf.config.set_visible_devices([], 'GPU')
#    physical_devices = tf.config.list_physical_devices('GPU')
#    if not physical_devices:
#        print("GPUs are not visible to TensorFlow. Running on CPU.")
#    else:
#        print("WARNING: GPUs are still visible to TensorFlow despite trying to disable them.")
# except Exception as e:
#    print(f"Error disabling GPUs: {e}. TensorFlow might still attempt to use them.")
# --- --- --- --- --- --- --- ---

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

SIM_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters', 'simulation.yaml')
AGI_PARAM_DIR = os.path.join(_PROJECT_ROOT, 'agilib', 'params')
SIM_BASE_DIR = os.path.join(_PROJECT_ROOT, 'mtrl_trainer', 'parameters')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join(os.getcwd(), 'logs', f'ppo_{current_time}')
policy_dir = os.path.join(os.getcwd(), 'policies')
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoin')

# --- Hyperparameters ---

# Model architecture
actor_fc_layers = (256, 256)
value_fc_layers = (256, 256)

# Performance parameters
num_drones_train = 96
num_drones_eval = 16
collect_steps_per_iteration = 800

num_epochs = 10
num_iterations = 10 #originally 550
eval_interval = 50
log_interval = 1
num_eval_episodes = 6

# ML hyperparameters
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


class TanhNormal(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, bijector, name='TanhNormalDistribution'):
        # Store parameters for the TFP subclassing pattern.
        parameters = dict(locals())

        super(TanhNormal, self).__init__(
            distribution=distribution,
            bijector=bijector,
            name=name)

        # Ensure that the subclass's parameters are stored.
        self._parameters = parameters

    def mode(self, **kwargs):
        """The mode is the tanh of the base distribution's mode (which is its mean)."""
        return self._bijector.forward(self.distribution.mode(**kwargs))

    def entropy(self, **kwargs):
        """
        Calculates the entropy of the base (pre-squashed) distribution.
        This is the standard approach for entropy regularization with squashed distributions.
        """
        return self.distribution.entropy(**kwargs)

    def mean(self, **kwargs):
        """
        The analytical mean of a Tanh-squashed Normal is not trivial.
        As is standard practice, we use the mean of the base (pre-squashed)
        normal distribution for the agent's calculations.
        """
        return self.distribution.mean(**kwargs)

    def stddev(self, **kwargs):
        """Returns the standard deviation of the base (pre-squashed) distribution."""
        return self.distribution.stddev(**kwargs)

    # ADD THIS METHOD to properly define the distribution's parameters.
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            distribution=tfp.util.ParameterProperties(
                event_ndims=lambda self: self.distribution.event_shape.rank),
            bijector=tfp.util.ParameterProperties(is_tensor=False))
class MTRLActorNetwork(network.Network):
    """
    A complete, robust MTRL Actor Network that handles multi-task encoding,
    time-series data, and manual action distribution projection.
    """
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 shared_encoder,
                 task_encoders,
                 actor_fc_layers=(256, 256),
                 activation_fn=tf.nn.leaky_relu,
                 name='MTRLActorNetwork'):

        super(MTRLActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._shared_encoder = shared_encoder
        self._task_encoders = {str(k): v for k, v in task_encoders.items()}
        self._combiner = tf.keras.layers.Concatenate(axis=-1)
        self._cast_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self._actor_layers = [
            tf.keras.layers.Dense(units, activation=activation_fn) for units in actor_fc_layers
        ]
        num_actions = action_spec.shape.num_elements()
        self._projection_layer = tf.keras.layers.Dense(num_actions * 2, activation=None)

    def call(self, observation, step_type=None, network_state=(), outer_rank=0):
        # This utility correctly flattens [Batch, Time] dimensions for training
        # and handles [Batch] dimension for collection.
        batch_squash = network_utils.BatchSquash(outer_rank + 1)

        # Squash all input tensors to be 2D
        shared_obs_squashed = batch_squash.flatten(observation['shared_obs'])
        task_obs_squashed = batch_squash.flatten(observation['task_specific_obs'])
        task_id_squashed = batch_squash.flatten(observation['task_id'])

        # Process the 2D squashed tensors through the encoders
        shared_embedding = self._shared_encoder(shared_obs_squashed)   # [B*, 32]

        # 2. Build the per-task inputs *including* the shared embedding
        racing_in        = tf.concat([shared_embedding, task_obs_squashed], axis=-1)      # 32+24 = 56
        stabilization_in = tf.concat([shared_embedding, task_obs_squashed[..., :4]], axis=-1) # 32+4 = 36

        # 3. Run the dedicated encoders
        racing_embedding        = self._task_encoders['0'](racing_in)        # → 32
        stabilization_embedding = self._task_encoders['1'](stabilization_in)

        # Use tf.where to select the correct embedding for each item
        is_racing_mask = tf.equal(task_id_squashed, 0)
        task_embedding = tf.where(is_racing_mask, racing_embedding, stabilization_embedding)

        task_id_float = self._cast_layer(task_id_squashed)
        combined_features = self._combiner([shared_embedding, task_embedding, task_id_float])

        x = combined_features
        for layer in self._actor_layers:
            x = layer(x)

        projection = self._projection_layer(x)
        mean, log_std = tf.split(projection, 2, axis=-1)

        # Ensure standard deviation is positive
        std = tf.math.softplus(log_std)

        mean = tf.nn.tanh(mean)

        # --- FIX: Clip the standard deviation to a safe range ---
        # This prevents the log_prob from exploding and causing NaNs.
        # These are common, safe values for continuous control policies.
        std = tf.clip_by_value(std, 1e-6, 1.0)

        # Un-squash the distribution parameters to restore the time dimension
        mean = batch_squash.unflatten(mean)
        std = batch_squash.unflatten(std)

        action_distribution = TanhNormal(
            distribution=tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=tfp.bijectors.Tanh()
        )

        return action_distribution, network_state




class TaskSwitchingEncoder(network.Network):
    def __init__(self, input_tensor_spec, name='TaskSwitchingEncoder'):
        """
        Initializes the switching network with a simple spec.
        The actual encoders are set later via set_task_encoders.
        """
        super(TaskSwitchingEncoder, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(), name=name)
        self._task_encoders = None

    def set_task_encoders(self, task_encoders):
        """
        Sets the internal encoders.
        Args:
            task_encoders (dict): Maps task_id (int) to the encoder network.
                                  e.g., {0: racing_encoder, 1: stabilization_encoder}
        """
        self._task_encoders = task_encoders

    def call(self, observation, step_type=None, network_state=()):
        if self._task_encoders is None:
            raise ValueError("Task encoders have not been set. Call set_task_encoders() after initialization.")

        task_specific_obs = observation['task_specific_obs']
        task_id = tf.cast(observation['task_id'][0][0], tf.int32)

        # We need to slice the observation correctly for each task before passing it
        def branch_racing():
            # Racing task uses the full 24 dimensions
            return self._task_encoders[0](task_specific_obs)[0]
        def branch_stabilization():
            # Stabilization task only uses the first 4 dimensions
            return self._task_encoders[1](task_specific_obs[:, :4])[0]

        output = tf.switch_case(task_id, branch_fns={0: branch_racing, 1: branch_stabilization})
        return output, network_state


class MultiHeadValueNetwork(network.Network):
    """
    Implements a multi-critic architecture with completely separate networks
    for each task, as described in the paper.
    """
    def __init__(self, input_tensor_spec, fc_layer_params=(256, 256), name='MultiHeadValueNetwork'):
        super(MultiHeadValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._combiner = tf.keras.layers.Concatenate(axis=-1)

        # --- Define a completely separate network for the Racing critic ---
        self._racing_critic_net = [
            tf.keras.layers.Dense(params, activation=tf.nn.leaky_relu) for params in fc_layer_params
        ]
        self._racing_critic_net.append(tf.keras.layers.Dense(1, name='value_head_racing'))

        # --- Define a completely separate network for the Stabilization critic ---
        self._stabilization_critic_net = [
            tf.keras.layers.Dense(params, activation=tf.nn.leaky_relu) for params in fc_layer_params
        ]
        self._stabilization_critic_net.append(tf.keras.layers.Dense(1, name='value_head_stabilization'))


    def call(self, observation, step_type=None, network_state=()):
        # Handle the time dimension for training
        outer_rank = observation['shared_obs'].shape.rank - 1
        batch_squash = network_utils.BatchSquash(outer_rank)

        # Squash inputs and create the full observation vector
        shared_obs_squashed = batch_squash.flatten(observation['shared_obs'])
        task_obs_squashed = batch_squash.flatten(observation['task_specific_obs'])
        task_id_squashed = batch_squash.flatten(observation['task_id'])
        full_obs_vector = self._combiner([shared_obs_squashed, task_obs_squashed])

        # --- Define the separate computational paths ---
        def run_racing_critic():
            x = full_obs_vector
            for layer in self._racing_critic_net:
                x = layer(x)
            return x

        def run_stabilization_critic():
            x = full_obs_vector
            for layer in self._stabilization_critic_net:
                x = layer(x)
            return x

        # Use tf.where to select the output from the correct separate network
        is_racing_mask = tf.equal(task_id_squashed, 0)
        value = tf.where(is_racing_mask, run_racing_critic(), run_stabilization_critic())

        # Un-squash the output to restore the time dimension if it existed
        value = batch_squash.unflatten(value)

        return tf.squeeze(value, axis=-1), network_state




def train_eval():
    best_eval_avg_return = -np.inf

    summary_writer = create_file_writer(log_dir, flush_millis=10000)
    summary_writer.set_as_default()

    final_track_layout = load_track_from_file(SIM_BASE_DIR + "/track.json")

    print("Creating environments...")
    py_train_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_train,
        track_layout=final_track_layout
    )
    py_eval_env = BatchedAgiSimEnv(
        sim_config_path=SIM_CONFIG_PATH,
        agi_param_dir=AGI_PARAM_DIR,
        sim_base_dir=SIM_BASE_DIR,
        num_drones=num_drones_eval,
        track_layout=final_track_layout,
    )

    train_env = tf_py_environment.TFPyEnvironment(py_train_env)
    eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
    print("Environments created.")

    actual_py_env = py_train_env
    actual_py_e_env = eval_env

    actual_py_e_env.setEval()

    shared_encoder = SharedEncoderKeras(output_size=32, name='SharedEncoder')
    task_encoder_racing = TaskSpecificRacingEncoderKeras(output_size=32, name='TaskSpecificRacingEncoder')
    task_encoder_stabilization = TaskSpecificStabilizationEncoderKeras(output_size=32, name='TaskSpecificStabilizationEncoder')

    # --- 2. Create the task-switching preprocessor ---
    # First, create the encoder with a simple spec.
    task_switching_encoder = TaskSwitchingEncoder(
        input_tensor_spec=train_env.observation_spec()
    )
    # THEN, set the internal networks. This avoids the serialization error.
    task_switching_encoder.set_task_encoders({
        0: task_encoder_racing,
        1: task_encoder_stabilization
    })

    preprocessing_layers = {
        'shared_obs': shared_encoder,
        'task_specific_obs': task_switching_encoder
    }

    # the critic does not use observation encoders, see MTRL paper
    critic_preprocessing_layers = {
        'shared_obs': tf.keras.layers.Flatten(),
        'task_specific_obs': tf.keras.layers.Flatten()
    }

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    shared_encoder = SharedEncoderKeras(output_size=32, name='SharedEncoder')
    task_encoder_racing = TaskSpecificRacingEncoderKeras(output_size=32, name='TaskSpecificRacingEncoder')
    task_encoder_stabilization = TaskSpecificStabilizationEncoderKeras(output_size=32, name='TaskSpecificStabilizationEncoder')

    # --- 2. Create the custom Actor Network ---
    actor_net = MTRLActorNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        shared_encoder=shared_encoder,
        task_encoders={
            0: task_encoder_racing,
            1: task_encoder_stabilization
        },
        actor_fc_layers=actor_fc_layers,
        activation_fn=tf.nn.leaky_relu
    )

    value_net = MultiHeadValueNetwork(
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
        value_pred_loss_coef=value_pred_loss_coef,
        use_gae=True,
        lambda_value=gae_lambda,
        discount_factor=discount_factor,
        normalize_observations=True,
        normalize_rewards=True,
        train_step_counter=train_step_counter,
        debug_summaries=True,  # Keep True to monitor agent internals
        summarize_grads_and_vars=True
    )

    agent.initialize()
    print("Agent created and initialized.")

    # checkpoint restore logic, this is not doing anything currently
    network_checkpoint_dir = os.path.join(log_dir, f'network-checkpoints')
    os.makedirs(network_checkpoint_dir, exist_ok=True)

    network_checkpointer_agent = tf.train.Checkpoint(
        actor_keras_model=actor_net
    )

    network_checkpointer = tf.train.Checkpoint(
        actor_keras_model=actor_net,
        value_keras_model=value_net
    )

    checkpoint_manager = tf.train.CheckpointManager(
        network_checkpointer,
        directory=network_checkpoint_dir,
        max_to_keep=5,
        checkpoint_name="ckpt"
    )

    latest_ckpt_to_load = tf.train.latest_checkpoint(checkpoint_dir)
    network_checkpointer_agent.restore(latest_ckpt_to_load)

    collect_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
    ]

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

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

        for _ in range(collect_steps_per_iteration):  # Loop T times
            action_step = agent.collect_policy.action(time_step, policy_state)
            tf.debugging.assert_all_finite(action_step.action, "actor produced NaN/Inf")

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

        collection_duration = time.time() - collection_start_time

        experience_for_train = None
        total_loss_tensor = tf.constant(0.0, dtype=tf.float32)
        train_call_duration = 0.0

        try:
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

            if hasattr(experience_for_train, 'step_type') and tf.size(experience_for_train.step_type) > 0:
                train_call_start_time = time.time()
                loss_info = agent.train(experience=experience_for_train)
                train_call_duration = time.time() - train_call_start_time
                total_loss_tensor = tf.cast(loss_info.loss, dtype=tf.float32)
            else:
                print(f"Iter {current_step}: No trajectories stacked, skipping training.")  # Should be current_step

        except Exception as e_train:
            print(f"ERROR during data stacking or agent.train: {e_train}")
            import traceback
            traceback.print_exc()

        iter_total_duration = time.time() - iteration_start_time
        iteration_times.append(iter_total_duration)
        train_duration_this_iter = train_call_duration

        if current_step % log_interval == 0:
            avg_states = actual_py_env.getStates()

            print(f"AVERAGE STATE = {avg_states}")

            avg_iter_time_smooth = np.mean(iteration_times[-log_interval:]) if len(
                iteration_times) >= log_interval else iter_total_duration
            actual_steps_collected_this_iter = collect_steps_per_iteration * train_env.batch_size
            iter_steps_per_sec = actual_steps_collected_this_iter / iter_total_duration if iter_total_duration > 0 else 0
            steps_per_sec_smooth = actual_steps_collected_this_iter / avg_iter_time_smooth if avg_iter_time_smooth > 0 else 0

            print(
                f"LOG Step={current_step}, Loss={total_loss_tensor.numpy():.4f}, "
                f"Steps/sec={iter_steps_per_sec:.2f} (Smoothed: {steps_per_sec_smooth:.2f}), "
                f"IterTime={iter_total_duration:.3f}s (Collect: {collection_duration:.3f}s, Train: {train_duration_this_iter:.3f}s)"
            )
            log_dict = {metric.name: metric.result() for metric in collect_metrics}

            with summary_writer.as_default(step=current_step):
                tf.summary.scalar('Loss/TotalLoss', total_loss_tensor, step=current_step)
                for key, value in log_dict.items():
                    metric_val = value.numpy() if hasattr(value, 'numpy') else value
                    tf.summary.scalar(f'Metrics/{key}', metric_val, step=current_step)
                tf.summary.scalar('Performance/StepsPerSecond', iter_steps_per_sec, step=current_step)
                tf.summary.scalar('Performance/StepsPerSecondSmoothed', steps_per_sec_smooth, step=current_step)
                tf.summary.scalar('Performance/IterationTime_sec', iter_total_duration, step=current_step)
                tf.summary.scalar('Performance/CollectionTime_sec', collection_duration, step=current_step)
                tf.summary.scalar('Performance/TrainTime_sec', train_duration_this_iter, step=current_step)
            summary_writer.flush()

            policy_artifact_dir = os.path.join(policy_dir, f'ppo_policy_{current_step}')

            tf_policy_saver.save(policy_artifact_dir)
            checkpoint_manager.save(checkpoint_number=train_step_counter)

            for metric in collect_metrics: metric.reset()

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
                f'-------- EVALUATION FINAL TRACK Step={eval_step}, Average Return={avg_return.numpy():.4f} ({eval_duration:.2f} sec) --------')

            if avg_return > best_eval_avg_return:
                best_eval_avg_return = avg_return

                policy_artifact_dir = os.path.join(policy_dir, f'ppo_policy_{avg_return}')

                tf_policy_saver.save(policy_artifact_dir)
                checkpoint_manager.save(checkpoint_number=train_step_counter)

    print("Training finished.")
    tf_policy_saver.save(os.path.join(policy_dir, 'ppo_policy_end_of_run'))


if __name__ == '__main__':
    os.makedirs(log_dir, exist_ok=True)
    train_eval()

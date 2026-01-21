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

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies import policy_saver
# from tf_agents.replay_buffers import tf_uniform_replay_buffer # No longer used for training data
from tf_agents.utils import common
from tf_agents.trajectories import trajectory, time_step as ts_lib
from tf_agents.networks import network  # For base Network class
from tf_agents.specs import tensor_spec

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
num_drones_eval = 8
collect_steps_per_iteration = 1500

num_epochs = 10
num_iterations = 200 #Originally 400
eval_interval = 50
log_interval = 1
num_eval_episodes = 4

# ML hyperparameters
learning_rate = 8e-4
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=num_iterations * num_epochs,
    end_learning_rate=8e-5,
    power=2.0
)

discount_factor = 0.99
gae_lambda = 0.95
entropy_regularization = 0.02
importance_ratio_clipping = 0.2
gradient_clipping = 0.3
value_pred_loss_coef = 0.3


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

    preprocessing_layers = {
        'shared_obs': shared_encoder,
        'task_specific_obs': task_encoder_racing,
        'task_id': tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
    }

    # the critic does not use observation encoders, see MTRL paper
    critic_preprocessing_layers = {
        'shared_obs': tf.keras.layers.Flatten(),
        'task_specific_obs': tf.keras.layers.Flatten(),
        'task_id': tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
    }

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    print("Creating networks...")
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        activation_fn=tf.nn.leaky_relu
    )
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        preprocessing_layers=critic_preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=value_fc_layers,
        activation_fn=tf.nn.leaky_relu
    )

    # temp agent that is only needed if loading weights from a checkpoint
    temp_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Dummy LR for temp agent
    temp_train_step = tf.Variable(0)
    temp_agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(), train_env.action_spec(), optimizer=temp_optimizer,
        actor_net=actor_net, value_net=value_net, train_step_counter=temp_train_step, num_epochs=1
    )
    temp_agent.initialize()

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

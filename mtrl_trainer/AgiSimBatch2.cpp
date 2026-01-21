#include "AgiSimBatch.hpp"
#include "agilib/simulator/quadrotor_simulator.hpp"
#include "agilib/utils/filesystem.hpp"
#include "agilib/types/command.hpp"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "ITask.hpp"
#include "TaskRacing.hpp"
#include "TaskStabilization.hpp"
#include "TaskInterceptionMoving.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace agi;
namespace py = pybind11;

namespace {

    SimulatorParams loadSimParamsFromFile(const std::string &sim_config_filepath,
                                          const std::string &agi_param_dir,
                                          const std::string &sim_base_dir) // Renamed last arg for clarity
    {
        std::cout << "[loadSimParamsFromFile] Loading Sim Params from " << sim_config_filepath << std::endl;
        // FIX 1: Use correct namespace for filesystem 'fs' (assuming it's within 'agi')
        if (!fs::exists(sim_config_filepath)) {
            throw std::runtime_error("Simulator config file not found: " + sim_config_filepath);
        }
        try {
            return SimulatorParams(sim_config_filepath, agi_param_dir, sim_base_dir);
        } catch (const std::exception &e) {
            std::cerr << "Error loading/parsing SimulatorParams: " << e.what() << std::endl;
            throw;
        }
    }

}

namespace agi_env {

    AgiSimBatch::AgiSimBatch(int num_drones,
                             const std::string &sim_config_path,
                             const std::string &agi_param_dir,
                             const std::string &sim_base_dir,
                             const std::vector<std::vector<std::vector<double>>> &track_layout_world
    )
            :
            num_drones_(num_drones),
            sim_params_(loadSimParamsFromFile(sim_config_path, agi_param_dir, sim_base_dir)),
            sim_dt_(sim_params_.sim_dt_),
            evalMode_(false),
            reward_params_(sim_base_dir + "/reward_params.yaml") {

        std::cout << "AgiSimBatch Initializing..." << std::endl;

const std::string reward_yaml_path = sim_base_dir + "/reward_params.yaml";
std::cout << "[AgiSimBatch] sim_base_dir      = " << sim_base_dir << std::endl;
std::cout << "[AgiSimBatch] reward_yaml_path  = " << reward_yaml_path << std::endl;
{
    std::ifstream f(reward_yaml_path);
    if (!f.good()) {
        std::cout << "[AgiSimBatch][WARN] reward_params.yaml not readable at: " << reward_yaml_path << std::endl;
    }
}
        std::cout << "[AgiSimBatch] reward_params.yaml = " << (sim_base_dir + "/reward_params.yaml") << std::endl;
        std::cout << std::fixed << std::setprecision(3)
                  << "[AgiSimBatch] INTERCEPTION params (loaded): "
                  << "CAPTURE_RADIUS=" << reward_params_.INTERCEPTION_CAPTURE_RADIUS
                  << " MAX_TIME_S=" << reward_params_.INTERCEPTION_MAX_EPISODE_TIME_S
                  << " SUCCESS_REWARD=" << reward_params_.INTERCEPTION_SUCCESS_REWARD
                  << " DIST_DELTA_SCALE=" << reward_params_.INTERCEPTION_DIST_DELTA_SCALE
                  << " DISTANCE_SCALE=" << reward_params_.INTERCEPTION_DISTANCE_SCALE
                  << " TIME_PENALTY=" << reward_params_.INTERCEPTION_TIME_PENALTY_PER_STEP
                  << " VEL_CLOSING_SCALE=" << reward_params_.INTERCEPTION_VEL_CLOSING_SCALE
                  << " CRASH_PENALTY=" << reward_params_.INTERCEPTION_CRASH_PENALTY
                  << std::endl;


        // --- 1. Instantiate all available tasks ---
        // This is where you add new tasks to make them available to the environment.
        auto interception_task = std::make_shared<TaskInterceptionMoving>(*this);
        available_tasks_.push_back(interception_task);
        //available_tasks_.push_back(std::make_shared<RacingTask>());
        //available_tasks_.push_back(std::make_shared<StabilizationTask>());
        // available_tasks_.push_back(std::make_shared<YourNextTask>());

        // Configure interception task (world-frame target velocity)
        agi::Vector<3> target_vel_world(-1.0, 0.0, 0.0);  // [m/s]
        interception_task->setTargetVelocityWorld(target_vel_world);

        // --- 2. Initialize all available tasks ---
        for (const auto &task: available_tasks_) {
            task->initialize(reward_params_, track_layout_world, num_drones_);
        }

        // --- 3. Initialize generic simulation components and per-drone vectors ---
        drone_current_tasks_.resize(num_drones_);
        current_states_.resize(num_drones_);
        previous_states_.resize(num_drones_);
        prev_actions_raw_.resize(num_drones_);
        prev_actions_scaled_.resize(num_drones_);

        per_drone_episode_rewards_.resize(num_drones_);
        per_drone_episode_step_count_.assign(num_drones_, 0);

        vio_drift_offsets_.resize(num_drones_, Vector<6>::Zero());

        // Initialize the simulators
        simulators_.reserve(num_drones_);
        for (int i = 0; i < num_drones_; ++i) {
            simulators_.emplace_back(sim_params_);
        }

        // Initialize random number generators (per thread)
        int max_threads = 1;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#endif
        rngs_.resize(max_threads);
        std::random_device rd;
        for (int i = 0; i < max_threads; ++i) {
            rngs_[i].seed(rd() + i);
        }

        std::cout << "Initialization complete. Environment is ready." << std::endl;
    }

// Utility function to print stats of saved gate pass states in the training script.
    double AgiSimBatch::getStates() const {
        for (const auto &task_ptr: drone_current_tasks_) {
            if (task_ptr) {
                auto racing_task_ptr = std::dynamic_pointer_cast<const RacingTask>(task_ptr);

                if (racing_task_ptr) {
                    racing_task_ptr->getStates();
                    return 0.0;
                }
            }
        }

        //return is currently unused
        return 0.0;
    }

// The eval mode flag is used whenever we want different behavious in eval vs training.
// e.g. in evaluation, all drones should start at the global start point.
    void AgiSimBatch::setEval() {
        evalMode_ = true;

    }

    // Normalize all observation components to [-1, 1] by using min, max bound constants.
    ObservationType AgiSimBatch::scaleAndClipObservation(const ObservationType &obs_raw, int drone_idx) const {
        ObservationType obs_scaled = obs_raw;

        auto scale_func = [](double val, double min_b, double max_b) {
            val = std::max(min_b, std::min(max_b, val));
            if (std::abs(max_b - min_b) < 1e-6) {
                return 0.0;
            }
            return 2.0 * (val - min_b) / (max_b - min_b) - 1.0;
        };

        int current_idx = 0;

        // Position p (indices 0-2)
        for (int i = 0; i < 2; ++i) {
            obs_scaled(current_idx + i) = scale_func(obs_raw(current_idx + i), reward_params_.OBS_POS_MIN(i),
                                                     reward_params_.OBS_POS_MAX(i));
        }
        current_idx += 2;
        double adj_z = obs_raw(current_idx);
        obs_scaled(current_idx) = scale_func(adj_z, reward_params_.OBS_POS_MIN(2), reward_params_.OBS_POS_MAX(2));
        current_idx += 1;

        // R_tilde (indices 3-8) - No scaling, assumed to be in [-1, 1]
        current_idx += 6;

        // Velocity v (indices 9-11)
        for (int i = 0; i < 3; ++i) {
            obs_scaled(current_idx + i) = scale_func(obs_raw(current_idx + i), reward_params_.OBS_VEL_MIN(i),
                                                     reward_params_.OBS_VEL_MAX(i));
        }
        current_idx += 3;

        // Angular velocity omega (indices 12-14)
        for (int i = 0; i < 3; ++i) {
            obs_scaled(current_idx + i) = scale_func(obs_raw(current_idx + i), reward_params_.OBS_ANG_VEL_MIN(i),
                                                     reward_params_.OBS_ANG_VEL_MAX(i));
        }
        current_idx += 3;

        // Previous action a_prev (indices 15-18) - No scaling, assumed to be in [-1, 1]
        current_idx += 4;


        // --- 2. Delegate scaling of the TASK-SPECIFIC part ---
        if (drone_current_tasks_[drone_idx]) {
            const auto task_obs_raw = obs_raw.tail<TASK_SPECIFIC_OBS_SIZE>();
            const auto task_obs_scaled = drone_current_tasks_[drone_idx]->scaleAndClipTaskSpecificObservation(
                    task_obs_raw);
            obs_scaled.tail<TASK_SPECIFIC_OBS_SIZE>() = task_obs_scaled;
        }

        return obs_scaled;
    }


    // De-normalize all action components from [-1, 1] by multiplying with their max values
    ActionScaledType AgiSimBatch::scaleAction(const PolicyActionType &raw_action) const {
        ActionScaledType sim_action;

        const PolicyActionType &policy_min = reward_params_.POLICY_ACTION_MIN;
        const PolicyActionType &policy_max = reward_params_.POLICY_ACTION_MAX;

        double policy_range_thrust = policy_max(0) - policy_min(0);
        double norm_thrust = (policy_range_thrust > 1e-6) ?
                             ((raw_action(0) - policy_min(0)) / policy_range_thrust) : 0.0;
        sim_action(0) = reward_params_.MIN_THRUST_CMD +
                        norm_thrust * (reward_params_.MAX_THRUST_CMD - reward_params_.MIN_THRUST_CMD);


        const Vector<3> policy_min_rates = policy_min.tail(3);
        const Vector<3> policy_max_rates = policy_max.tail(3);

        const Vector<3> policy_range_rates_vec = policy_max_rates - policy_min_rates;

        // Check that all components of the range vector are valid.
        if (policy_range_rates_vec.minCoeff() > 1e-6) {
            Vector<3> norm_rates = (
                    ((raw_action.tail(3) - policy_min_rates).array() / policy_range_rates_vec.array()) * 2.0 -
                    1.0).matrix();
            sim_action.tail(3) = norm_rates.array() * reward_params_.MAX_RATE_CMD;
        } else {
            sim_action.tail(3).setZero();
        }

        return sim_action;
    }

    // helper to progress the strength of induced state perturbations
    void AgiSimBatch::setNoiseIntensity(double vio_pos_drift_std, double vio_att_drift_std_deg,
                                        double gate_reset_pos_std, double gate_reset_att_std_deg) {
        vio_pos_drift_std_ = vio_pos_drift_std;
        gate_reset_pos_std_ = gate_reset_pos_std;
        vio_att_drift_std_ = vio_att_drift_std_deg * M_PI / 180.0;
        gate_reset_att_std_ = gate_reset_att_std_deg * M_PI / 180.0;
    }

    // helper function for testing how the policy performs with a pertrurbed input state. Might also be useful for sim2real
    QuadState AgiSimBatch::perturbState(QuadState raw_state, int drone_idx) {
        QuadState estimated_state = raw_state;

        // This const_cast is a safe way to access the non-const RNG from a const method
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        std::mt19937 &local_rng = const_cast<AgiSimBatch *>(this)->rngs_[thread_id];

        // 1. Update the current drift with a small random walk
        std::normal_distribution<double> pos_drift_dist(0.0, vio_pos_drift_std_); // USE MEMBER VARIABLE
        std::normal_distribution<double> att_drift_dist(0.0, vio_att_drift_std_); // USE MEMBER VARIABLE

        Vector<6> &current_drift = const_cast<AgiSimBatch *>(this)->vio_drift_offsets_[drone_idx]; // Assuming drone_idx is available or passed
        current_drift.segment<3>(0) += Vector<3>(pos_drift_dist(local_rng), pos_drift_dist(local_rng),
                                                 pos_drift_dist(local_rng));
        current_drift.segment<3>(3) += Vector<3>(att_drift_dist(local_rng), att_drift_dist(local_rng),
                                                 att_drift_dist(local_rng));

        // 2. Apply the accumulated drift to the state to create the "estimated" state
        estimated_state.p += current_drift.segment<3>(0);

        Vector<3> attitude_drift_rad = current_drift.segment<3>(3);
        Quaternion attitude_drift_q(
                Eigen::AngleAxis(attitude_drift_rad.z(), Vector<3>::UnitZ()) *
                Eigen::AngleAxis(attitude_drift_rad.y(), Vector<3>::UnitY()) *
                Eigen::AngleAxis(attitude_drift_rad.x(), Vector<3>::UnitX())
        );
        estimated_state.q((estimated_state.q() * attitude_drift_q).normalized());

        return estimated_state;
    }

    // assembles the observation from shared and task-specific components
    ObservationType AgiSimBatch::assembleObservation(int drone_idx) {
        ObservationType obs;
        obs.setZero();

        QuadState estimated_state = perturbState(current_states_[drone_idx], drone_idx);

        agi::Vector<SHARED_OBS_SIZE> shared_obs;
        int current_obs_idx = 0;

        shared_obs.segment<3>(current_obs_idx) = estimated_state.p;
        current_obs_idx += 3;

        agi::Matrix<3, 3> R_wb = estimated_state.q().toRotationMatrix();

        // 1. Assemble the shared part of the observation
        obs.head<SHARED_OBS_SIZE>() = assembleSharedObservation(estimated_state,
                                                                prev_actions_raw_[drone_idx]);

        // 2. Get the task-specific part from the drone's current task
        if (drone_current_tasks_[drone_idx]) {
            obs.tail<TASK_SPECIFIC_OBS_SIZE>() = drone_current_tasks_[drone_idx]->getTaskSpecificObservation(
                    estimated_state,
                    drone_idx
            );
        }

        return obs;
    }

    // Assembles the shared observation from quadstate and previous action
    agi::Vector<SHARED_OBS_SIZE> AgiSimBatch::assembleSharedObservation(const agi::QuadState &current_state,
                                                                        const PolicyActionType &prev_raw_action) const {
        agi::Vector<SHARED_OBS_SIZE> shared_obs;
        int current_obs_idx = 0;

        // 1. Position p (3)
        shared_obs.segment<3>(current_obs_idx) = current_state.p;
        current_obs_idx += 3;

        // 2. Rotation Matrix Columns R_tilde (6)
        agi::Matrix<3, 3> R_wb = current_state.q().toRotationMatrix();
        shared_obs.segment<3>(current_obs_idx) = R_wb.col(0);
        current_obs_idx += 3;
        shared_obs.segment<3>(current_obs_idx) = R_wb.col(1);
        current_obs_idx += 3;

        // 3. Linear Velocity v (3)
        shared_obs.segment<3>(current_obs_idx) = current_state.v;
        current_obs_idx += 3;

        // 4. Angular Velocity omega (3)
        const Vector<3> body_rates_obs = R_wb.transpose() * current_state.w;
    	shared_obs.segment<3>(current_obs_idx) = body_rates_obs;
		//shared_obs.segment<3>(current_obs_idx) = current_state.bw;
        current_obs_idx += 3;

        // 5. Previous Action a_prev (4)
        shared_obs.segment<4>(current_obs_idx) = prev_raw_action;

        return shared_obs;
    }

    // some Tasks (e.g. Racing) benefit from computing future rewards from a certain state
    // (e.g. judging how "good" a gate pass state is by computing how much reward the drone got after this gate pass)
    void AgiSimBatch::_processEndOfEpisodeForDrone(int drone_idx) {
        if (per_drone_episode_step_count_[drone_idx] <= 0) {
            return; // Nothing to process
        }

        const auto &episode_rewards = per_drone_episode_rewards_[drone_idx];
        size_t episode_length = episode_rewards.size();
        std::vector<double> future_discounted_rewards(episode_length, 0.0);

        double g_next_step = 0.0;
        const double gamma = reward_params_.DISCOUNT_FACTOR_GAMMA;

        for (int step = episode_length - 1; step >= 0; --step) {
            future_discounted_rewards[step] = episode_rewards[step] + gamma * g_next_step;
            g_next_step = future_discounted_rewards[step];
        }

        drone_current_tasks_[drone_idx]->processEndOfEpisode(drone_idx, future_discounted_rewards);

        per_drone_episode_rewards_[drone_idx].clear();
        per_drone_episode_step_count_[drone_idx] = 0;
    }

    // The reset function starts a new episode
    std::vector <ObservationType> AgiSimBatch::reset(const std::vector<bool> &reset_flags) {
        if (reset_flags.size() != num_drones_) {
            throw std::runtime_error("Reset flags vector size does not match number of drones.");
        }
        std::vector <ObservationType> initial_observations(num_drones_);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_drones_; ++i) {
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            std::mt19937 &local_rng = rngs_[thread_id];

            if (reset_flags[i]) {
                // --- 1. Process End of Previous Episode ---
                if (per_drone_episode_step_count_[i] > 0) {
                    _processEndOfEpisodeForDrone(i);
                } else {
                    per_drone_episode_rewards_[i].clear();
                }

                // --- 2. Reset Simulator Physics ---
                simulators_[i].reset(true);
                simulators_[i].setCommand(Command(0.0, 0.0, Vector<3>::Zero()));

                // --- 3. Assign a Task for the New Episode ---
                if (evalMode_) {
                	// In evaluation mode, always assign the first task (assumed to be Racing).
                	drone_current_tasks_[i] = available_tasks_[0];
            	} else {
                	// For training, use a weighted distribution to make racing twice as likely.
                	std::vector<double> task_weights;
                	for (const auto& task : available_tasks_) {
                    	if (task->getTaskID() == 0) { // Racing Task ID is 0
                        	task_weights.push_back(2.0); // Weight of 2 for a 2/3 chance
                    	} else {
                        	task_weights.push_back(1.0); // Weight of 1 for a 1/3 chance
                    	}
                	}

                	std::discrete_distribution<> task_dist(task_weights.begin(), task_weights.end());
                	int selected_task_index = task_dist(rngs_[omp_get_thread_num()]);
                	drone_current_tasks_[i] = available_tasks_[selected_task_index];
            	}

                // --- 4. DELEGATE Reset to the Task ---
                // The task is now fully responsible for all of its reset logic.
                QuadState base_initial_state = drone_current_tasks_[i]->resetDrone(i, local_rng, evalMode_);

                // --- 5. Set Final State in Simulator ---
                simulators_[i].setState(base_initial_state);

                prev_actions_raw_[i].setZero();
                prev_actions_scaled_[i].setZero();

                vio_drift_offsets_[i].setZero();
            }

            // --- 7. Per tf-agents definition the reset function also returns a new observation
            simulators_[i].getState(&current_states_[i]);
            initial_observations[i] = scaleAndClipObservation(assembleObservation(i), i);
        }
        return initial_observations;
    }

    // In short, the step function gets an action from the agent, applies it in the simulation, computes a reward and returns an observation.
    std::vector <StepResultSimple> AgiSimBatch::step(const std::vector <PolicyActionType> &current_actions_raw) {
        std::vector <StepResultSimple> results(num_drones_);
        previous_states_ = current_states_;

        std::vector <ActionScaledType, Eigen::aligned_allocator<ActionScaledType>> scaled_actions_this_step(
                num_drones_);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_drones_; ++i) {

            // --- 1. Get previous state and scale current action ---
            const QuadState &prev_drone_state_snapshot = previous_states_[i];
            const ActionScaledType &prev_drone_action_s_snapshot = prev_actions_scaled_[i];
            const PolicyActionType &current_drone_action_raw = current_actions_raw[i];

            ActionScaledType current_drone_action_s;
            current_drone_action_s = scaleAction(current_drone_action_raw);


            scaled_actions_this_step[i] = current_drone_action_s;

            // --- 2. Run the physics simulation ---
            bool sim_success = false;
            Command cmd(prev_drone_state_snapshot.t, current_drone_action_s(0), current_drone_action_s.tail(3));
            simulators_[i].setCommand(cmd);
            sim_success = simulators_[i].run(sim_dt_);

            QuadState current_drone_state_after_sim;
            simulators_[i].getState(&current_drone_state_after_sim);
            current_states_[i] = current_drone_state_after_sim;

            // --- 3. Check for generic termination conditions ---
            bool done_from_sim_status = !sim_success || isDone(i, current_drone_state_after_sim);

            // --- 4. Delegate task-specific logic ---
            TaskStepResult task_result = drone_current_tasks_[i]->onStep(i, per_drone_episode_step_count_[i],
                                                                         prev_drone_state_snapshot,
                                                                         current_drone_state_after_sim,
                                                                         vio_drift_offsets_[i],
                                                                         gate_reset_pos_std_,
                                                                         gate_reset_att_std_);


            // The episode ends if the simulation fails, the drone is out of bounds, OR the task reports a terminal failure.
            bool final_done =
                    done_from_sim_status ||
                    task_result.missed_goal ||
                    task_result.lap_completed;

            // --- 5. Delegate reward calculation to the task ---
            double reward;
            reward = drone_current_tasks_[i]->calculateReward(
                    prev_drone_state_snapshot,
                    prev_drone_action_s_snapshot,
                    current_drone_action_s,
                    current_drone_state_after_sim,
                    task_result,
                    done_from_sim_status,
                    i
            );

            // --- 6. Assemble the next observation ---
            ObservationType final_observation;
            ObservationType obs_raw = assembleObservation(i);
            final_observation = scaleAndClipObservation(obs_raw, i);

            // --- 7. Store results and handle episode data ---
            results[i].observation = final_observation;
            results[i].reward = reward;
            results[i].done = final_done;
            results[i].time = current_drone_state_after_sim.t;
            results[i].success = task_result.lap_completed;
			results[i].task_id = drone_current_tasks_[i]->getTaskID();

            per_drone_episode_rewards_[i].push_back(reward);
            per_drone_episode_step_count_[i]++;

            if (final_done) {
                _processEndOfEpisodeForDrone(i);
            }
        }

        // --- 8. Update state for the next step ---
        prev_actions_raw_.assign(current_actions_raw.begin(), current_actions_raw.end());
        prev_actions_scaled_.assign(scaled_actions_this_step.begin(), scaled_actions_this_step.end());

        return results;
    }

    // This checks various termination conditions.
    bool AgiSimBatch::isDone(int i, const QuadState &state) const {
		const RacingTask* racing_task = dynamic_cast<const RacingTask*>(drone_current_tasks_[i].get());



        return drone_current_tasks_[i]->onIsDone(state, evalMode_);
    }

}
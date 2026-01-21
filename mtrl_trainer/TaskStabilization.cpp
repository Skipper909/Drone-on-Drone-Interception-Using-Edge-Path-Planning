#include "TaskStabilization.hpp"
#include <cmath> // For std::abs, std::sqrt
#include <iostream>

// Use M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace agi;
namespace py = pybind11;

namespace agi_env {

    StabilizationTask::StabilizationTask() = default;

    void StabilizationTask::initialize(const RewardParams& reward_params,
                                       const std::vector<std::vector<std::vector<double>>>& track_layout,
                                       int num_drones) {
        std::cout << "Initializing StabilizationTask..." << std::endl;
        reward_params_ = reward_params;
        num_drones_ = num_drones;
        drone_states_.resize(num_drones_);

    }

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> StabilizationTask::getTaskSpecificObservation(
            const agi::QuadState& current_state,
            int drone_idx) const {

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> task_obs;
        task_obs.setZero();

        task_obs.segment<3>(0) = drone_states_[drone_idx].current_acceleration;
        task_obs(3) = current_state.p.z() - target_height_z_;

        return task_obs;
    }

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> StabilizationTask::scaleAndClipTaskSpecificObservation(
            const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const {

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> obs_scaled = task_obs_raw;

        auto scale_func = [](double val, double min_b, double max_b) {
            val = std::max(min_b, std::min(max_b, val));

            if (std::abs(max_b - min_b) < 1e-6) {
                return 0.0;
            }
            return 2.0 * (val - min_b) / (max_b - min_b) - 1.0;
        };

        // --- 1. Scale Acceleration (indices 0-2) ---
        for (int i = 0; i < 3; ++i) {
            obs_scaled(i) = scale_func(task_obs_raw(i),
                                       reward_params_.OBS_ACC_MIN(i),
                                       reward_params_.OBS_ACC_MAX(i));
        }

        // --- 2. Scale Target Height z_d (index 3) ---
        obs_scaled(3) = scale_func(task_obs_raw(3),
                                   reward_params_.OBS_POS_MIN(2), // Index 2 for the Z-axis
                                   reward_params_.OBS_POS_MAX(2));

        return obs_scaled;
    }

    agi::QuadState StabilizationTask::resetDrone(int drone_idx, std::mt19937& rng, bool evalMode_) {
        // The drone is initialized with a random position, orientation, and velocities.

		if(evalMode_) {
			velocity_curriculum_level_ = 9999;
		}

        QuadState initial_state;
        initial_state.setZero();

        std::uniform_real_distribution<double> pos_dist(-25.0, 25.0);
        std::uniform_real_distribution<double> pos_z_dist(-4.0, 4.0);
        std::uniform_real_distribution<double> att_dist(-M_PI, M_PI);
        std::uniform_real_distribution<double> rate_dist(-0.1, 0.1);

        initial_state.p = Vector<3>(pos_dist(rng), pos_dist(rng), pos_z_dist(rng) + target_height_z_);
        // Ensure it starts high enough to not crash instantly
        //initial_state.p.z() = std::max(2.0, initial_state.p.z());

        Vector<3> euler_angles(att_dist(rng), att_dist(rng), att_dist(rng));
        initial_state.q(
            Eigen::AngleAxisd(euler_angles.z(), Vector<3>::UnitZ()) *
            Eigen::AngleAxisd(euler_angles.y(), Vector<3>::UnitY()) *
            Eigen::AngleAxisd(euler_angles.x(), Vector<3>::UnitX())
        );
		initial_state.q().normalize();

        double current_vel_bound = initial_vel_bound_ * std::pow(1.0 + curriculum_increase_factor_, velocity_curriculum_level_);
        current_vel_bound = std::min(current_vel_bound, max_vel_bound_);

        double current_vel_z_bound = initial_vel_z_bound_ * std::pow(1.0 + curriculum_increase_factor_, velocity_curriculum_level_);
        current_vel_z_bound = std::min(current_vel_z_bound, max_vel_z_bound_);

        std::uniform_real_distribution<double> vel_dist(-current_vel_bound, current_vel_bound);
        std::uniform_real_distribution<double> vel_z_dist(-current_vel_z_bound, current_vel_z_bound);

        initial_state.v = Vector<3>(vel_dist(rng), vel_dist(rng), vel_z_dist(rng));
        initial_state.w = Vector<3>(rate_dist(rng), rate_dist(rng), 0.0);

        drone_states_[drone_idx].prev_velocity = initial_state.v;

        return initial_state;
    }

    TaskStepResult StabilizationTask::onStep(int drone_idx, int step_count,
                                  const agi::QuadState& prev_state,
                                  const agi::QuadState& current_state,
                                  agi::Vector<6>& vio_drift, // Pass drift by reference
                                      double gate_reset_pos_std,
                                      double gate_reset_att_std) {
        TaskStepResult result;

        task_interaction_count_++;
        if (task_interaction_count_ > 0 && (task_interaction_count_ % curriculum_update_interval_ == 0)) {
            velocity_curriculum_level_++;
            std::cout << "\n*** STABILIZATION TASK - CURRICULUM UPDATE: New Velocity Level: "
                      << velocity_curriculum_level_ << " ***\n" << std::endl;
        }

        const double sim_dt = 0.01; // TODO This should come from sim_params


		drone_states_[drone_idx].current_acceleration = current_state.a;

        result.lap_completed = false;
        result.missed_goal = false;
        result.passed_goal = false;

        //TODO: define in params
        const double hover_linear_velocity_threshold = 0.5;

    // Add a reasonable threshold to ensure the drone is not spinning.
    const double hover_angular_velocity_threshold = 1.25; // rad/s is a good starting point

    // A true hover requires both linear AND angular velocities to be near zero.

	const Matrix<3, 3> R_t = current_state.q().toRotationMatrix();
	const Vector<3> body_rates_manual = R_t.transpose() * current_state.w;

    if (current_state.v.norm() < hover_linear_velocity_threshold && body_rates_manual.norm() < hover_angular_velocity_threshold) {

        // Using 'passed_goal' is more semantically correct for stabilization success.
        result.passed_goal = true;
    }

        return result;
    }

    double StabilizationTask::calculateReward(const agi::QuadState& prev_state,
                                              const ActionScaledType& prev_action_s,
                                              const ActionScaledType& current_action_s,
                                              const agi::QuadState& current_state,
                                              const TaskStepResult& task_result,
                                              bool done_from_sim_status,
                                              int drone_idx) const {

const double beta1 = -7e-4; // height
        const double beta2 = -3e-4;
        const double beta3 = -6e-4; // velocity
        const double beta4 = -5e-5; // body rate
		const double beta_yaw_rate = -3e-4;
        const double beta5 = -1.5e-4; // action smoothness
        const double beta6 = 10.0;  // success

        // 1. Height Reward: Penalizes deviation from target height z_d.
		double height_error = current_state.p.z() - target_height_z_;
        double r_height = beta1 * abs(height_error);

        // 2. Attitude Reward: Penalizes non-level orientation.
const QuadState hover_state = current_state.getHoverState();
// angularDistance gives the shortest angle (in radians) between two quaternions.
// This cleanly measures tilt error (pitch/roll) regardless of the current yaw.
const double tilt_error = current_state.q().angularDistance(hover_state.q());

double r_attitude = beta2 * abs(tilt_error);


        // 3. Velocity Penalty
        double r_velocity = beta3 * 3 * current_state.v.norm();
		//double r_velocity = beta3  * 12 * Vector<2>(current_state.v.x(), current_state.v.y()).norm();


        // 4. Body Rate Penalty
		const Matrix<3, 3> R_t = current_state.q().toRotationMatrix();
		const Vector<3> body_rates_manual = R_t.transpose() * current_state.w;

        double r_body_rate = beta4 * body_rates_manual.norm();
		double r_body_rate_action = beta4 * 55.5 * current_action_s.tail(3).norm();

		double r_yaw_rate = beta_yaw_rate * abs(body_rates_manual.z());

        // 5. Action Smoothness Penalty
        double r_action = beta5 * 2 * (current_action_s - prev_action_s).norm();

        // 6. Success Reward
        double r_success = task_result.passed_goal ? beta6 : 0.0;

		double r_outofbounds = current_state.p.z() < reward_params_.OBS_POS_MIN(2) / 2 || current_state.p.z() > reward_params_.OBS_POS_MAX(2) / 2 || current_state.p.x() < reward_params_.OBS_POS_MIN(0) / 2 || current_state.p.x() > reward_params_.OBS_POS_MAX(0) / 2 || current_state.p.y() < reward_params_.OBS_POS_MIN(1) / 2 || current_state.p.y() > reward_params_.OBS_POS_MAX(1) / 2 ? -4.0 : 0.0;

        double total_reward = r_velocity + r_attitude + r_body_rate + r_body_rate_action + r_action + r_success;

        return total_reward;
    }

    void StabilizationTask::processEndOfEpisode(int drone_idx, const std::vector<double>& episode_rewards) {
        // This task does not require special end-of-episode processing
    }

    bool StabilizationTask::onIsDone(const QuadState &state, bool evalMode_) {
        if (state.t > reward_params_.MAX_SIM_TIME_STABILIZATION) {
            return true;
        }

        return false;
    }

} // namespace agi_env
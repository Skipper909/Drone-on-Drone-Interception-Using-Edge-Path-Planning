#include "TaskRacing.hpp"
#include <cmath> // For M_PI, std::cos, std::sin
#include "agilib/math/types.hpp" // For agi::Quaternion
#include <stdexcept> // For std::out_of_range
#include <iostream>
#include <algorithm> // for std::min_element, std::max_element

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace agi;
namespace py = pybind11;

namespace agi_env {

    RacingTask::RacingTask() = default;

    void RacingTask::initialize(const RewardParams& reward_params,
                                const std::vector<std::vector<std::vector<double>>>& track_layout,
                                int num_drones) {

        std::cout << "Initializing RacingTask..." << std::endl;
        reward_params_ = reward_params;
        num_drones_ = num_drones;
        num_gates_ = track_layout.size();

        drone_states_.resize(num_drones_);
        max_laps_ = reward_params_.MAX_LAPS;
        drone_lap_counts_.assign(num_drones_, 0);

        std::random_device rd;
        rng_.seed(rd());

        // --- Resize all gate-related data structures ---
        gates_corners_world_.assign(num_gates_, {});
        gate_centers_world_.assign(num_gates_, Vector<3>::Zero());
        gate_normals_world_.assign(num_gates_, Vector<3>(0.0, 0.0, 1.0)); // Default normal

        // --- Loop through the provided layout and populate the vectors ---
        for (int i = 0; i < num_gates_; ++i) {
            if (track_layout[i].size() != 4) {
                throw std::runtime_error("RacingTask init: Gate " + std::to_string(i) + " does not have 4 corners.");
            }

            gates_corners_world_[i].resize(4);

            Vector<3> center_sum = Vector<3>::Zero();
            for (int j = 0; j < 4; ++j) {
                if (track_layout[i][j].size() != 3) {
                    throw std::runtime_error("RacingTask init: A corner of gate " + std::to_string(i) + " does not have x, y, z");
                }
                gates_corners_world_[i][j] = Vector<3>(track_layout[i][j].data());
                center_sum += gates_corners_world_[i][j];
            }
            gate_centers_world_[i] = center_sum / 4.0;

            // Calculate the gate normal vector
            Vector<3> v1 = gates_corners_world_[i][1] - gates_corners_world_[i][0];
            Vector<3> v2 = gates_corners_world_[i][3] - gates_corners_world_[i][0];
            Vector<3> normal = v1.cross(v2);
            Scalar norm_mag = normal.norm();

            if (norm_mag > 1e-6) {
                gate_normals_world_[i] = normal / norm_mag;
            } else {
                std::cerr << "WARNING: RacingTask init: Gate " << i
                << " appears degenerate. Setting normal to Z-axis." << std::endl;
            }
        }

        successful_gate_pass_states_.assign(num_gates_, {});
        gate_pass_states_mutexes_ = std::vector<std::mutex>(num_gates_);

        std::cout << "RacingTask: Track layout processed with " << num_gates_ << " gates." << std::endl;
    }

    double RacingTask::getStates() const {
        double avg_sum = 0.0;
        int total_states_counted = 0;
        bool gate_0_printed_this_call = false;
        double overall_avg_sum_scores = 0.0;
        int overall_total_states_counted = 0;

        for (int i = 0; i < successful_gate_pass_states_.size(); i++) {
            const auto &current_gate_buffer = successful_gate_pass_states_[i]; // Read directly (const ref)
            std::cout << "[Gate " << i << "] Stored Entries: " << current_gate_buffer.size();

            if (!current_gate_buffer.empty()) {
                double sum_of_scores_for_this_gate = 0.0;
                for (const auto &entry: current_gate_buffer) {
                    sum_of_scores_for_this_gate += entry.second;
                    overall_avg_sum_scores += entry.second;
                    overall_total_states_counted++;
                }

                double average_score_for_this_gate =
                        sum_of_scores_for_this_gate / static_cast<double>(current_gate_buffer.size());

                std::cout << ", Avg Score: " << std::fixed << std::setprecision(6) << average_score_for_this_gate
                          << std::endl;

            } else {
                std::cout << " (Buffer Empty)" << std::endl;
            }
        }

        std::cout << "-- " << use_spawns << " --" << std::endl << std::flush;

        double final_avg = 0.0;
        double expected_total_capacity = successful_gate_pass_states_.size() * reward_params_.MAX_STORED_STATES_PER_GATE;
        if (expected_total_capacity > 0) {
            final_avg = overall_avg_sum_scores / expected_total_capacity;
        }

        return final_avg;
    }

    agi::Vector<3> RacingTask::getGateCenter(int gate_idx) const {
        if (gate_idx < 0) {
            return reward_params_.START_POS;
        } else if (gate_idx >= num_gates_) {
            if (num_gates_ == 0) return reward_params_.START_POS;
            return gate_centers_world_[num_gates_ - 1];
        } else {
            return gate_centers_world_[gate_idx];
        }
    }

    inline bool isPointInRectangle(double u, double v, double half_width, double half_height) {
        return (std::abs(u) <= half_width) && (std::abs(v) <= half_height);
    }

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> RacingTask::scaleAndClipTaskSpecificObservation(
            const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const {

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> obs_scaled = task_obs_raw;

        auto scale_func = [](double val, double min_b, double max_b) {
            val = std::max(min_b, std::min(max_b, val));
            if (std::abs(max_b - min_b) < 1e-6) {
                return 0.0;
            }
            return 2.0 * (val - min_b) / (max_b - min_b) - 1.0;
        };

        // Scale delta_p1 (first 12 elements of the task-specific vector)
        for (int i = 0; i < 12; ++i) {
            obs_scaled(i) = scale_func(task_obs_raw(i),
                                       reward_params_.OBS_REL_CORNER_WORLD_MIN,
                                       reward_params_.OBS_REL_CORNER_WORLD_MAX);
        }

        // Scale delta_p2 (next 12 elements of the task-specific vector)
        for (int i = 0; i < 12; ++i) {
            obs_scaled(12 + i) = scale_func(task_obs_raw(12 + i),
                                            reward_params_.OBS_REL_CORNER_WORLD_MIN,
                                            reward_params_.OBS_REL_CORNER_WORLD_MAX);
        }

        return obs_scaled;
    }

    agi::QuadState RacingTask::resetDrone(int drone_idx, std::mt19937& rng, bool evalMode_) {
        // --- 1. Decide on the start position from all equally likely options ---
        const int total_start_options = 1 + num_gates_; // 1 global start + N gates
        std::uniform_int_distribution<int> option_dist(0, total_start_options - 1);
        int chosen_option = option_dist(rng);

        QuadState base_initial_state;
        base_initial_state.setZero();
        int gate_to_start_at = -1;

        bool is_global_start = chosen_option == num_gates_ || evalMode_;
        bool has_stored_states = false;

        drone_lap_counts_[drone_idx] = 0;

        // --- 2. Set internal task state AND generate the physical state ---
        if (!is_global_start) {
            // --- A. Task-Specific Start (at a gate) ---
            gate_to_start_at = chosen_option;

            std::vector<std::pair<QuadState, double>> candidate_pass_states_with_scores;

           if (gate_to_start_at >= 0 && gate_to_start_at < successful_gate_pass_states_.size()) {
                 std::lock_guard<std::mutex> lock(gate_pass_states_mutexes_[gate_to_start_at]);
                 if (!successful_gate_pass_states_[gate_to_start_at].empty()) {
                    // This assignment is now type-correct
                    candidate_pass_states_with_scores = successful_gate_pass_states_[gate_to_start_at];
                    has_stored_states = true;
                 }
           }

            // Set the drone's target to be the gate AFTER the start gate
            drone_states_[drone_idx].target_gate_idx = (gate_to_start_at + 1) % num_gates_;

            if (use_spawns && has_stored_states) {
                std::uniform_int_distribution <size_t> random_state_dist(0, candidate_pass_states_with_scores.size() - 1);
                base_initial_state = candidate_pass_states_with_scores[random_state_dist(rng)].first;


            } else {
                // --- Generate physical state at the chosen gate ---
                base_initial_state.p = gate_centers_world_[gate_to_start_at];

                Vector < 3 > x_body_dir_gate_normal = gate_normals_world_[gate_to_start_at];
                x_body_dir_gate_normal.normalize();
                Vector < 3 > z_world_up(0.0, 0.0, 1.0);
                Vector < 3 > y_body_dir_initial = z_world_up.cross(x_body_dir_gate_normal).normalized();
                if (y_body_dir_initial.squaredNorm() < 1e-6) {
                    y_body_dir_initial = (x_body_dir_gate_normal.z() > 0.99) ? Vector < 3 > (0, 1, 0) : Vector < 3 >
                                                                                                        (0, -1, 0);
                }
                Vector < 3 > z_body_dir_initial = x_body_dir_gate_normal.cross(y_body_dir_initial).normalized();
                y_body_dir_initial = z_body_dir_initial.cross(x_body_dir_gate_normal).normalized();
                Matrix<3, 3> R_align_to_normal;
                R_align_to_normal.col(0) = x_body_dir_gate_normal;
                R_align_to_normal.col(1) = y_body_dir_initial;
                R_align_to_normal.col(2) = z_body_dir_initial;
                const double pitch_tilt_radians = reward_params_.INITIAL_RESET_PITCH_TILT_DEGREES * M_PI / 180.0;
                Matrix<3, 3> R_pitch_tilt = Eigen::AngleAxisd(pitch_tilt_radians,
                                                              Vector < 3 > ::UnitY()).toRotationMatrix();
                Matrix<3, 3> R_final_body_to_world = R_align_to_normal * R_pitch_tilt;
                base_initial_state.q(Quaternion(R_final_body_to_world));
                Vector < 3 > drone_forward_direction_world = R_final_body_to_world.col(0);
                base_initial_state.v = drone_forward_direction_world * reward_params_.INITIAL_RESET_SPEED;

            }

        } else {
            // --- B. Global Start ---
            drone_states_[drone_idx].target_gate_idx = 0;
            base_initial_state.p = reward_params_.START_POS;
            base_initial_state.q(Quaternion::Identity());
            base_initial_state.v.setZero();
            base_initial_state.bw.setZero();
            has_stored_states = false;
        }

        // --- 3. Finalize Per-Drone State for the Episode ---
        drone_states_[drone_idx].episode_gate_passes.clear();
        if (num_gates_ > 0) {
            const auto& target_center = gate_centers_world_[drone_states_[drone_idx].target_gate_idx];
            const auto& start_pos = (gate_to_start_at == -1) ? reward_params_.START_POS : gate_centers_world_[gate_to_start_at];
            drone_states_[drone_idx].prev_gate_distance = (start_pos - target_center).norm();
        } else {
            drone_states_[drone_idx].prev_gate_distance = std::numeric_limits<double>::infinity();
        }

        // --- 4. APPLY PERTURBATIONS ---
        QuadState perturbed_initial_state = base_initial_state;

        std::uniform_real_distribution<double> pos_dist(-reward_params_.RESET_POS_PERTURB_BOUND, reward_params_.RESET_POS_PERTURB_BOUND);
        perturbed_initial_state.p += Vector<3>(pos_dist(rng), pos_dist(rng), pos_dist(rng));

        // Use different yaw bounds for global start vs gate start
        double yaw_perturb_bound = is_global_start ? reward_params_.RESET_ATT_YAW_GLOBAL_START_PERTURB_BOUND : reward_params_.RESET_ATT_YAW_GATE_START_PERTURB_BOUND;
        std::uniform_real_distribution<double> yaw_dist(-yaw_perturb_bound, yaw_perturb_bound);
        std::uniform_real_distribution<double> rp_dist(-reward_params_.RESET_ATT_RP_GATE_START_PERTURB_BOUND, reward_params_.RESET_ATT_RP_GATE_START_PERTURB_BOUND);

        Vector<3> euler_pert(is_global_start ? 0.0 : rp_dist(rng), is_global_start ? 0.0 : rp_dist(rng), yaw_dist(rng));
        Matrix<3, 3> R_pert = (Eigen::AngleAxisd(euler_pert.z(), Vector<3>::UnitZ()) *
                               Eigen::AngleAxisd(euler_pert.y(), Vector<3>::UnitY()) *
                               Eigen::AngleAxisd(euler_pert.x(), Vector<3>::UnitX())).toRotationMatrix();

        perturbed_initial_state.q((perturbed_initial_state.q() * Quaternion(R_pert)).normalized());

        // Only perturb linear/angular velocity if starting at a gate
        if (!is_global_start) {
            std::uniform_real_distribution<double> vel_dist(-reward_params_.RESET_VEL_GATE_START_PERTURB_BOUND, reward_params_.RESET_VEL_GATE_START_PERTURB_BOUND);
            perturbed_initial_state.v += Vector<3>(vel_dist(rng), vel_dist(rng), vel_dist(rng));

            std::uniform_real_distribution<double> ome_dist(-reward_params_.RESET_OME_GATE_START_PERTURB_BOUND, reward_params_.RESET_OME_GATE_START_PERTURB_BOUND);
            perturbed_initial_state.w += Vector<3>(ome_dist(rng), ome_dist(rng), ome_dist(rng));
        }

        perturbed_initial_state.p.z() = std::max(0.1, perturbed_initial_state.p.z());

        return perturbed_initial_state;
    }

    TaskStepResult RacingTask::onStep(int drone_idx, int step_count,
                                      const agi::QuadState& prev_state,
                                      const agi::QuadState& current_state,
                                      agi::Vector<6>& vio_drift, // Pass drift by reference
                                      double gate_reset_pos_std,
                                      double gate_reset_att_std) {
        TaskStepResult result;
        if (num_gates_ == 0) return result;

        int target_idx = drone_states_[drone_idx].target_gate_idx;
        GatePassResult pass_result = checkGatePassDetailed(prev_state, current_state, target_idx);
        result.lap_completed = false;

        result.passed_goal = (pass_result == GatePassResult::PassedInside);
        result.missed_goal = (pass_result == GatePassResult::PassedOutside);

        if (result.passed_goal) {
            int next_gate_idx = (target_idx + 1) % num_gates_;

            if (next_gate_idx == 0) {
                drone_lap_counts_[drone_idx]++;
            }

            // TODO only in eval
            if (drone_lap_counts_[drone_idx] >= max_laps_) {
                result.lap_completed = true; // Signal that the race is finished
            }

            // Store gate pass info for end-of-episode processing
            // Note: The step count needs to be managed by AgiSimBatch and passed in if needed for replay buffer
            drone_states_[drone_idx].episode_gate_passes.emplace_back(step_count, target_idx,
                                                                      current_state);
            // Update to next target gate
            drone_states_[drone_idx].target_gate_idx = (target_idx + 1) % num_gates_;

            if (step_count > 0 && (step_count % reward_params_.DRIFT_CORRECTION_INTERVAL == 0)) {

                // Reset the drift to a new, small random value based on the gate detector's uncertainty
                std::normal_distribution<double> pos_reset_dist(0.0, gate_reset_pos_std);
                std::normal_distribution<double> att_reset_dist(0.0, gate_reset_att_std);

                vio_drift.segment<3>(0) = Vector<3>(pos_reset_dist(rng_), pos_reset_dist(rng_), pos_reset_dist(rng_));
                vio_drift.segment<3>(3) = Vector<3>(att_reset_dist(rng_), att_reset_dist(rng_), att_reset_dist(rng_));
            }
        }

        // Update prev_gate_distance for reward calculation
        const auto &next_target_center = gate_centers_world_[drone_states_[drone_idx].target_gate_idx];
        drone_states_[drone_idx].prev_gate_distance = (current_state.p - next_target_center).norm();

        return result;
    }
    double RacingTask::calculateReward(const agi::QuadState& prev_state,
                                       const ActionScaledType& prev_action_s,
                                       const ActionScaledType& current_action_s,
                                       const agi::QuadState& current_state,
                                       const TaskStepResult& task_result,
                                       bool done_from_sim_status,
                                       int drone_idx) const {

        if (num_gates_ == 0) return 0.0;

        // --- Determine the Target Gate for this Step ---
        int current_target_idx = drone_states_[drone_idx].target_gate_idx;
        int previous_target_idx = current_target_idx;
        if (task_result.passed_goal) {
            previous_target_idx = (current_target_idx - 1 + num_gates_) % num_gates_;
        }
        const Vector<3>& P_target = gate_centers_world_[previous_target_idx];

        // --- Calculate Individual Reward/Penalty Components ---

        // 1. Progress Reward
        double dist_at_prev_step = (prev_state.p - P_target).norm();
        double dist_at_curr_step = (current_state.p - P_target).norm();
        double progress = dist_at_prev_step - dist_at_curr_step;
        double progress_reward = reward_params_.PROGRESS_WEIGHT * progress;

        // 2. Perception Reward
        const double CAMERA_TILT_ANGLE_RAD = reward_params_.CAMERA_TILT_ANGLE_DEG * M_PI / 180.0;
        const Vector<3> CAMERA_OPTICAL_AXIS_BODY_FRAME = Vector<3>(std::cos(CAMERA_TILT_ANGLE_RAD), 0.0, std::sin(CAMERA_TILT_ANGLE_RAD));
        Vector<3> drone_to_gate_vec = (P_target - current_state.p).normalized();
        Vector<3> camera_optical_axis_world = current_state.q() * CAMERA_OPTICAL_AXIS_BODY_FRAME;
        double dot_product = std::max(-1.0, std::min(1.0, camera_optical_axis_world.dot(drone_to_gate_vec)));
        double delta_cam_angle = std::acos(dot_product);
        double perception_reward = reward_params_.PERCEPTION_ALPHA2 * std::exp(reward_params_.PERCEPTION_ALPHA3 * std::pow(delta_cam_angle, 4));

        // 3. Action Smoothness Penalty
        ActionType action_diff = current_action_s - prev_action_s;
        double action_penalty = reward_params_.ACTION_SMOOTHNESS_WEIGHT * action_diff.norm();

        // 4. Body Rate Penalty
		const Matrix<3, 3> R_t = current_state.q().toRotationMatrix();
		const Vector<3> body_rates_obs = R_t.transpose() * current_state.w;
        double body_rate_penalty = reward_params_.BODY_RATE_PENALTY_WEIGHT * body_rates_obs.norm();


        // 5. Gate Pass Reward
        double gate_pass_reward = task_result.passed_goal ? reward_params_.GATE_PASS_REWARD : 0.0;

        // 6. Crash/Miss Penalty
        double term_penalty = 0.0;
        if (task_result.missed_goal) {
            double miss_distance = (current_state.p - P_target).norm();
            double penalty_factor = std::min(1.0, miss_distance / reward_params_.GATE_MISS_DISTANCE_THRESHOLD);
            term_penalty = reward_params_.GATE_MISS_BASE_PENALTY + penalty_factor * (reward_params_.GATE_MISS_MAX_PENALTY - reward_params_.GATE_MISS_BASE_PENALTY);
        } else if (done_from_sim_status) {
            term_penalty = reward_params_.TERMINATION_PENALTY;
        }

        // --- Sum all components to get the final reward ---
        double total_reward = progress_reward
                            + perception_reward
                            - action_penalty
                            - body_rate_penalty
                            + gate_pass_reward
                            - term_penalty;

        return total_reward;
    }

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> RacingTask::getTaskSpecificObservation(
            const agi::QuadState &current_state,
            int drone_idx) const {

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> task_obs;
        task_obs.setZero();

        if (num_gates_ > 0) {
            int target_gate_idx = drone_states_[drone_idx].target_gate_idx;

            // delta_p1: relative position of upcoming gate corners in drone's body frame
            task_obs.segment<12>(0) = getRelativeGateCornersForDrone(current_state, target_gate_idx);

            // delta_p2: relative position between the corners of the next and next-next gates
            int next_next_gate_idx = (target_gate_idx + 1) % num_gates_;
            task_obs.segment<12>(12) = getGateToGateRelativeCorners(target_gate_idx, next_next_gate_idx);
        }

        return task_obs;
    }

    Vector12d RacingTask::getRelativeGateCornersForDrone(const QuadState &state, int target_gate_idx) const {
        Vector12d rel_corners_body;
        rel_corners_body.setZero();
        if (target_gate_idx < 0 || target_gate_idx >= num_gates_) { return rel_corners_body; }
        const Matrix<3, 3> R_w_b = state.q().toRotationMatrix().transpose(); // World to Body
        const auto &gate_corners_w = gates_corners_world_[target_gate_idx];
        for (int j = 0; j < 4; ++j) {
            rel_corners_body.segment<3>(j * 3) = R_w_b * (gate_corners_w[j] - state.p);
        }
        return rel_corners_body;
    }

    agi::Vector<12> RacingTask::getGateToGateRelativeCorners(
            int upcoming_gate_idx,    // This is Gate i
            int next_next_gate_idx    // This is Gate i+1
    ) const {

        agi::Vector<12> gate_to_gate_rel_corners;
        gate_to_gate_rel_corners.setZero();

        if (upcoming_gate_idx < 0 || upcoming_gate_idx >= num_gates_ ||
            next_next_gate_idx < 0 || next_next_gate_idx >= num_gates_) {
            return gate_to_gate_rel_corners;
        }

        const auto& upcoming_gate_corners_w = gates_corners_world_[upcoming_gate_idx];
        const auto& next_next_gate_corners_w = gates_corners_world_[next_next_gate_idx];

        for (int j = 0; j < 4; ++j) {
            gate_to_gate_rel_corners.segment<3>(j * 3) = next_next_gate_corners_w[j] - upcoming_gate_corners_w[j];
        }

        return gate_to_gate_rel_corners;
    }

    GatePassResult RacingTask::checkGatePassDetailed(
            const agi::QuadState &prev_state,
            const agi::QuadState &current_state,
            int target_gate_idx) const {

        const Vector<3> &gate_center = gate_centers_world_[target_gate_idx];
        const Vector<3> &gate_normal = gate_normals_world_[target_gate_idx];
        const auto &gate_corners = gates_corners_world_[target_gate_idx]; // Need corners now

        Vector<3> prev_pos_rel_to_center = prev_state.p - gate_center;
        Vector<3> curr_pos_rel_to_center = current_state.p - gate_center;

        Scalar dot_prev = prev_pos_rel_to_center.dot(gate_normal);
        Scalar dot_curr = curr_pos_rel_to_center.dot(gate_normal);

        bool crossed_plane = (dot_prev * dot_curr <= 0.0);


        if (!crossed_plane) {
            return GatePassResult::NotCrossed;
        }

        Vector<3> intersection_point_world;
        Scalar diff_dots = dot_curr - dot_prev;

        if (std::abs(diff_dots) < 1e-9) {
            intersection_point_world = current_state.p;
        } else {
            Scalar t_intersect = -dot_prev / diff_dots;
            t_intersect = std::max(0.0, std::min(1.0, t_intersect));
            intersection_point_world = prev_state.p + t_intersect * (current_state.p - prev_state.p);
        }

        Vector<3> u_vec_world = gate_corners[1] - gate_corners[0]; // Vector along bottom edge
        Vector<3> v_vec_world = gate_corners[3] - gate_corners[0]; // Vector along left edge

        double gate_half_width = u_vec_world.norm() / 2.0;
        double gate_half_height = v_vec_world.norm() / 2.0;

        Vector<3> u_axis_world = u_vec_world.normalized();
        Vector<3> v_axis_world = gate_normal.cross(u_axis_world);

        Vector<3> rel_intersection_world = intersection_point_world - gate_center;
        Scalar u_coord = rel_intersection_world.dot(u_axis_world);
        Scalar v_coord = rel_intersection_world.dot(v_axis_world);

        double margin = reward_params_.GATE_PASS_MARGIN;
        bool is_inside = isPointInRectangle(u_coord, v_coord,
                                            gate_half_width + margin,
                                            gate_half_height + margin);

        if (is_inside) {
            return GatePassResult::PassedInside;
        } else {
            return GatePassResult::PassedOutside;
        }
    }

    void RacingTask::processEndOfEpisode(int drone_idx, const std::vector<double>& future_discounted_rewards) {
        auto& drone_state = drone_states_[drone_idx];

        for (const auto& pass_info_tuple : drone_state.episode_gate_passes) {
            int step_index_of_pass  = std::get<0>(pass_info_tuple);
            int gate_idx_passed     = std::get<1>(pass_info_tuple);
            const QuadState& state_at_pass = std::get<2>(pass_info_tuple);

            if (static_cast<size_t>(step_index_of_pass) < future_discounted_rewards.size()) {
                double quality_score = future_discounted_rewards[step_index_of_pass];

                _storeGatePassIfGood(gate_idx_passed, state_at_pass, quality_score);
            }
        }

        drone_state.episode_gate_passes.clear();
    }

    void RacingTask::_storeGatePassIfGood(int passed_gate_idx,
                                           const QuadState &state_at_pass,
                                           double future_accumulated_reward) {
        if (passed_gate_idx < 0 || passed_gate_idx >= num_gates_) {
            return;
        }

        std::lock_guard <std::mutex> lock(gate_pass_states_mutexes_[passed_gate_idx]);
        auto &buffer = successful_gate_pass_states_[passed_gate_idx];

        if (buffer.size() < reward_params_.MAX_STORED_STATES_PER_GATE) {
            if (future_accumulated_reward > 0.0) {
                buffer.push_back({state_at_pass, future_accumulated_reward});
            }

        } else {
            if (buffer.empty()) return;

            auto worst_element_it = std::min_element(buffer.begin(), buffer.end(),
                                                     [](const std::pair<QuadState, double> &a,
                                                        const std::pair<QuadState, double> &b) {
                                                         return a.second <
                                                                b.second; // Compare by future_accumulated_reward
                                                     });

            if (worst_element_it != buffer.end() && future_accumulated_reward > worst_element_it->second) {
                *worst_element_it = {state_at_pass, future_accumulated_reward};

                if (!use_spawns) {
                    bool thresh_reached = true;
                    for (int i = 0; i < successful_gate_pass_states_.size(); i++) {
                        const auto &current_gate_buffer = successful_gate_pass_states_[i]; // Read directly (const ref)
                        if (!current_gate_buffer.empty()) {
                            double sum_of_scores_for_this_gate = 0.0;
                            for (const auto &entry: current_gate_buffer) {
                                sum_of_scores_for_this_gate += entry.second; // entry.second is the double (score)

                            }

                            double average_score_for_this_gate =
                                    sum_of_scores_for_this_gate / static_cast<double>(current_gate_buffer.size());

                            if (average_score_for_this_gate < reward_params_.GATE_PASS_REWARD_THRESHOLD) {
                                thresh_reached = false;
                            }
                        } else {
                            thresh_reached = false;
                        }
                    }
                    use_spawns = thresh_reached;
                }
            }
        }
    }

    bool RacingTask::onIsDone(const QuadState &state, bool evalMode_) {
        if (state.p.z() < 0.1) return true;

		if (!evalMode_ && state.t > reward_params_.MAX_SIM_TIME_RACING || state.t > 4.0 * reward_params_.MAX_SIM_TIME_RACING) {
            return true;
        }

        return false;
    }

} // namespace agi_env
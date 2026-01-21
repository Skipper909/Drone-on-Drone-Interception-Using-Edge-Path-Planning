#include "TaskInterceptionMoving.hpp"
#include "AgiSimBatch.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace agi_env {

using agi::QuadState;
using agi::Vector;
using agi::Scalar;

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
TaskInterceptionMoving::TaskInterceptionMoving(AgiSimBatch& sim)
    : sim_(sim)
{
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

// -----------------------------------------------------------------------------
// ITask: basic info
// -----------------------------------------------------------------------------
int TaskInterceptionMoving::getTaskID() const {
    // Just needs to be unique vs. your other tasks
    return 3;
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
void TaskInterceptionMoving::initialize(
        const RewardParams& reward_params,
        const std::vector<std::vector<std::vector<double>>>& track_layout,
        int num_drones)
{
    num_drones_ = num_drones;
    drones_.assign(num_drones_, DroneData{});
    prev_dist_m_.assign(num_drones_, 0.0);

    // --- Gate center from first gate ---
    gate_center_world_ = Vector<3>::Zero();
    if (!track_layout.empty()) {
        const auto& gate_corners = track_layout.front();
        int count = 0;
        for (const auto& c : gate_corners) {
            if (c.size() >= 3) {
                gate_center_world_ += Vector<3>(c[0], c[1], c[2]);
                ++count;
            }
        }
        if (count > 0) {
            gate_center_world_ /= static_cast<Scalar>(count);
        }
    }

    // Target starts at gate center by default; velocity is set externally
    for (auto& d : drones_) {
        d.target_start_p = gate_center_world_;
        d.target_v       = Vector<3>::Zero();
    }

    // --- Wire in interception reward params ---
    capture_radius_        = reward_params.INTERCEPTION_CAPTURE_RADIUS;
    max_episode_time_s_    = reward_params.INTERCEPTION_MAX_EPISODE_TIME_S;

    distance_scale_        = reward_params.INTERCEPTION_DISTANCE_SCALE;
    dist_delta_scale_      = reward_params.INTERCEPTION_DIST_DELTA_SCALE;
    vel_closing_scale_     = reward_params.INTERCEPTION_VEL_CLOSING_SCALE;
    time_penalty_per_step_ = reward_params.INTERCEPTION_TIME_PENALTY_PER_STEP;

    success_reward_        = reward_params.INTERCEPTION_SUCCESS_REWARD;
    crash_penalty_         = reward_params.INTERCEPTION_CRASH_PENALTY;
}

// -----------------------------------------------------------------------------
// Configure target velocity (world frame)
// -----------------------------------------------------------------------------
void TaskInterceptionMoving::setTargetVelocityWorld(const Vector<3>& v_world)
{
    for (auto& d : drones_) {
        d.target_v = v_world;
    }
}

// -----------------------------------------------------------------------------
// Helper: target position as function of time
// -----------------------------------------------------------------------------
Vector<3> TaskInterceptionMoving::getTargetPosWorld(const QuadState& state,
                                                    int drone_idx) const
{
    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return gate_center_world_;
    }

    const auto& d = drones_.at(drone_idx);

    // state.t is the simulator's time [s] for this episode
    const double t = state.t;

    return d.target_start_p + d.target_v * t;
}

// -----------------------------------------------------------------------------
// Observations
// -----------------------------------------------------------------------------
Vector<TASK_SPECIFIC_OBS_SIZE>
TaskInterceptionMoving::getTaskSpecificObservation(const QuadState& state,
                                                   int drone_idx) const
{
    Vector<TASK_SPECIFIC_OBS_SIZE> obs;
    obs.setZero();

    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return obs;
    }

    const auto& d = drones_.at(drone_idx);

    // Target position is computed from start + v * t
    Vector<3> target_p = getTargetPosWorld(state, drone_idx);

    // Relative position (world frame) target - drone [m]
    Vector<3> p_rel = target_p - state.p;

    // Relative velocity (world frame): v_target - v_drone
    Vector<3> v_rel = d.target_v - state.v;

    // Simple scaling to put things in ~[-1, 1]
    constexpr double POS_SCALE = 300.0;  // meters
    constexpr double VEL_SCALE = 30.0;   // m/s

    Vector<3> p_rel_scaled = p_rel / POS_SCALE;
    Vector<3> v_rel_scaled = v_rel / VEL_SCALE;

    for (int i = 0; i < 3; ++i) {
        p_rel_scaled[i] = std::max(-1.0, std::min(1.0, static_cast<double>(p_rel_scaled[i])));
        v_rel_scaled[i] = std::max(-1.0, std::min(1.0, static_cast<double>(v_rel_scaled[i])));
    }

    int idx = 0;
    obs.template segment<3>(idx) = p_rel_scaled; idx += 3;
    obs.template segment<3>(idx) = v_rel_scaled; idx += 3;

    // Any remaining task-specific obs stay zero
    return obs;
}

Vector<TASK_SPECIFIC_OBS_SIZE>
TaskInterceptionMoving::scaleAndClipTaskSpecificObservation(
        const Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const
{
    // Already scaled and clipped in getTaskSpecificObservation()
    return task_obs_raw;
}

// -----------------------------------------------------------------------------
// Reset drone and target for a new episode
// -----------------------------------------------------------------------------
QuadState TaskInterceptionMoving::resetDrone(int drone_idx,
                                             std::mt19937& /*rng*/,
                                             bool /*evalMode*/)
{
    QuadState s;
    s.setZero();

    // Drone spawn near origin
    s.p = Vector<3>(0.0, 0.0, 1.5);
    s.v.setZero();
    s.w.setZero();
    s.q() = agi::Quaternion::Identity();
    s.t   = 0.0;

    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return s;
    }

    auto& d = drones_[drone_idx];
    d.target_start_p = gate_center_world_;   // or randomise per episode

    // Reset shaping memory
    prev_dist_m_[drone_idx] = 0.0;

    return s;
}

// -----------------------------------------------------------------------------
// Per-step task logic (termination conditions)
// -----------------------------------------------------------------------------
TaskStepResult TaskInterceptionMoving::onStep(int drone_idx,
                                              int /*step_count*/,
                                              const QuadState& /*prev_state*/,
                                              const QuadState& current_state,
                                              Vector<6>& /*vio_drift*/,
                                              double /*gate_reset_pos_std*/,
                                              double /*gate_reset_att_std*/)
{
    TaskStepResult result;

    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return result;
    }

    Vector<3> target_p = getTargetPosWorld(current_state, drone_idx);
    const double dist  = (target_p - current_state.p).norm();

    // Successful interception: within capture radius in 3D
    if (dist <= capture_radius_) {
        result.missed_goal   = true;   // end episode
        result.lap_completed = true;   // success flag
        return result;
    }

    // Timeout based on sim time
    if (current_state.t >= max_episode_time_s_) {
        result.missed_goal   = true;   // end episode
        result.lap_completed = false;  // no success
        return result;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Reward calculation
// -----------------------------------------------------------------------------
double TaskInterceptionMoving::calculateReward(const QuadState& /*prev_state*/,
                                               const ActionScaledType& /*prev_action_s*/,
                                               const ActionScaledType& /*current_action_s*/,
                                               const QuadState& current_state,
                                               const TaskStepResult& task_result,
                                               bool done_from_sim_status,
                                               int drone_idx) const
{
    const auto& d = drones_.at(drone_idx);

    Vector<3> target_p = getTargetPosWorld(current_state, drone_idx);

    // 3D relative position & distance in world [m]
    const Vector<3> p_rel = target_p - current_state.p;
    const double    dist  = p_rel.norm();

    // --- 1) Progress-based shaping on true 3D distance -----------------------
    double& prev_dist = prev_dist_m_.at(drone_idx);
    if (prev_dist <= 1e-6) {
        prev_dist = dist;
    }

    double delta = prev_dist - dist;   // > 0 if we got closer this step
    prev_dist    = dist;

    double reward = 0.0;

    // Only reward positive progress; do not penalise regress
    if (delta > 0.0) {
        reward += dist_delta_scale_ * delta;
    }

    // --- 2) Distance and closing-speed shaping -------------------------------
    constexpr double MAX_DIST   = 300.0;  // meters
    constexpr double MAX_CLOSE  = 40.0;   // m/s

    double dist_norm = std::min(dist / MAX_DIST, 1.0);

    // Relative velocity (target - drone)
    const Vector<3> v_rel = d.target_v - current_state.v;
    double closing_speed  = 0.0;
    if (dist > 1e-6) {
        // Positive when closing in
        closing_speed = -(p_rel.dot(v_rel) / dist);
    }
    double closing_norm = closing_speed / MAX_CLOSE;
    closing_norm = std::max(-1.0, std::min(1.0, closing_norm));

    // Mild penalty for being far away
    reward -= distance_scale_ * dist_norm;

    // Optional bonus for positive closing speed
    reward += vel_closing_scale_ * closing_norm;

    // Small per-step time penalty
    reward -= time_penalty_per_step_;

    // -------------------------------------------------------------------------
    // 3) Extra near-target shaping with two zones: 25 m and 10 m
    // -------------------------------------------------------------------------
    constexpr double FAR_RADIUS  = 25.0;   // outer shaping zone
    constexpr double NEAR_RADIUS = 10.0;   // inner shaping zone

    // Small bonus when within 25 m
    if (dist < FAR_RADIUS) {
        // 0 at dist = FAR_RADIUS, 1 at dist = 0
        double factor_far = (FAR_RADIUS - dist) / FAR_RADIUS;
        double far_bonus  = 0.10 * success_reward_;   // e.g. 10% of terminal reward
        reward += far_bonus * factor_far;
    }

    // Stronger bonus when within 10 m
    if (dist < NEAR_RADIUS) {
        double factor_near = (NEAR_RADIUS - dist) / NEAR_RADIUS;
        double near_bonus  = 0.25 * success_reward_;  // keep your current 25%
        reward += near_bonus * factor_near;
    }


    // --- 4) Terminal bonuses/penalties ---------------------------------------
    if (task_result.missed_goal && task_result.lap_completed && dist <= capture_radius_) {
        reward += success_reward_;
    }

    if (done_from_sim_status) {
        reward += crash_penalty_;
    }

    return reward;
}

// -----------------------------------------------------------------------------
// End-of-episode hook / isDone
// -----------------------------------------------------------------------------
void TaskInterceptionMoving::processEndOfEpisode(
        int drone_idx,
        const std::vector<double>& /*episode_rewards*/)
{
    if (drone_idx >= 0 && drone_idx < static_cast<int>(prev_dist_m_.size())) {
        prev_dist_m_[drone_idx] = 0.0;
    }
}

bool TaskInterceptionMoving::onIsDone(const QuadState& /*state*/, bool /*evalMode*/)
{
    // We use onStep() and simulator status for termination.
    return false;
}

} // namespace agi_env

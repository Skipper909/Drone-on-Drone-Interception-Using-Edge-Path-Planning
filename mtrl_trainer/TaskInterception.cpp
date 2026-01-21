#include "TaskInterception.hpp"
#include "agilib/math/types.hpp"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace agi;

namespace agi_env {

TaskInterception::TaskInterception(AgiSimBatch& sim)
    : sim_(sim) {
    std::random_device rd;
    rng_.seed(rd());
}

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

void TaskInterception::initialize(const RewardParams& reward_params,
                                  const std::vector<std::vector<std::vector<double>>>& track_layout,
                                  int num_drones) {
    reward_params_ = reward_params;
    num_drones_    = num_drones;

    drones_.clear();
    drones_.resize(num_drones_);

    has_track_target_ = false;

    // Use first gate from track_layout to define target (x,y). z is forced to 100 m.
    if (!track_layout.empty() && !track_layout[0].empty()) {
        const auto& gate0 = track_layout[0];

        double cx = 0.0, cy = 0.0, cz = 0.0;
        int count = 0;
        for (const auto& corner : gate0) {
            if (corner.size() < 3) continue;
            cx += corner[0];
            cy += corner[1];
            cz += corner[2];
            ++count;
        }

        if (count > 0) {
            cx /= count;
            cy /= count;
            cz /= count; // ← now we KEEP this value

            // Use actual gate height from track.json
            base_target_from_track_ = agi::Vector<3>(cx, cy, cz);

            has_track_target_ = true;

            std::cout << "[TaskInterception] Using track.json target at ("
                    << cx << ", " << cy << ", " << cz << ")\n";
        }

    }

    if (!has_track_target_) {
        // Fallback: approx 150 m out in +x, 100 m up
        base_target_from_track_ = agi::Vector<3>(150.0, 0.0, 100.0);
        has_track_target_       = true;

        std::cout << "[TaskInterception] WARNING: track_layout empty; "
                     "falling back to default target at (150, 0, 100)." << std::endl;
    }

    std::cout << "[TaskInterception] Initialized for " << num_drones_
              << " drones with interception task.\n";
}

// ----------------------------------------------------------------------------
// Observation helpers
// ----------------------------------------------------------------------------

agi::Vector<TASK_SPECIFIC_OBS_SIZE> TaskInterception::getTaskSpecificObservation(
    const agi::QuadState& current_state,
    int drone_idx) const {

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> task_obs;
    task_obs.setZero();

    if (drone_idx < 0 || drone_idx >= static_cast<int>(drones_.size())) {
        return task_obs;
    }

    const auto& d = drones_[drone_idx];

    // Relative position / velocity to static target
    Vector<3> p_rel = d.target_p - current_state.p;
    Vector<3> v_ego = current_state.v;

    double dist = p_rel.norm();
    double vmag = v_ego.norm();

    // Pack into first 8 entries, rest zero
    task_obs.segment<3>(0) = p_rel;
    task_obs.segment<3>(3) = v_ego;
    task_obs(6)            = dist;
    task_obs(7)            = vmag;

    // Remaining entries stay zero
    return task_obs;
}

agi::Vector<TASK_SPECIFIC_OBS_SIZE> TaskInterception::scaleAndClipTaskSpecificObservation(
    const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const {

    // For now: identity mapping (no additional scaling/clipping).
    // The shared encoder + PPO can learn directly from these magnitudes.
    // You can add explicit scaling here later if you like.
    return task_obs_raw;
}

// ----------------------------------------------------------------------------
// Reset
// ----------------------------------------------------------------------------

agi::QuadState TaskInterception::resetDrone(int drone_idx,
                                            std::mt19937& rng,
                                            bool /*evalMode_*/) {
    (void)rng; // currently unused; we rely on base_target_from_track_

    if (drone_idx < 0 || drone_idx >= static_cast<int>(drones_.size())) {
        QuadState dummy;
        dummy.setZero();
        return dummy;
    }

    auto& d = drones_[drone_idx];

    // 1) Assign target from track.json (center of first gate, z = 100 m)
    d.target_p = base_target_from_track_;

    // 2) Reset interceptor at/near origin, height 1 m, zero v, identity attitude
    QuadState s;
    s.setZero();
    s.p = Vector<3>(0.0, 0.0, 1.0);
    s.v = Vector<3>::Zero();
    s.q() = Quaternion::Identity();
    s.q().normalize();
    s.t = 0.0; // time will be overwritten by sim

    // 3) Initialize previous distance for progress-based reward
    d.prev_dist = (d.target_p - s.p).norm();

    return s;
}

// ----------------------------------------------------------------------------
// Step
// ----------------------------------------------------------------------------

TaskStepResult TaskInterception::onStep(int drone_idx,
                                        int /*step_count*/,
                                        const agi::QuadState& prev_state,
                                        const agi::QuadState& current_state,
                                        agi::Vector<6>& /*vio_drift*/,
                                        double /*gate_reset_pos_std*/,
                                        double /*gate_reset_att_std*/) {
    TaskStepResult result;

    if (drone_idx < 0 || drone_idx >= static_cast<int>(drones_.size())) {
        return result;
    }

    const auto& d = drones_[drone_idx];

    Vector<3> p_rel_prev = d.target_p - prev_state.p;
    Vector<3> p_rel_cur  = d.target_p - current_state.p;

    double dist_prev = p_rel_prev.norm();
    double dist_cur  = p_rel_cur.norm();

    // If we cross into the capture sphere, mark passed_goal
    if (dist_cur < capture_radius_ && dist_prev >= capture_radius_) {
        result.passed_goal   = true;
        result.lap_completed = false; // no multi-lap notion here
        result.missed_goal   = false;
    }

    return result;
}

// ----------------------------------------------------------------------------
// Reward
// ----------------------------------------------------------------------------

double TaskInterception::calculateReward(const agi::QuadState& /*prev_state*/,
                                         const ActionScaledType& /*prev_action_s*/,
                                         const ActionScaledType& /*current_action_s*/,
                                         const agi::QuadState& current_state,
                                         const TaskStepResult& task_result,
                                         bool done_from_sim_status,
                                         int drone_idx) const {
    if (drone_idx < 0 || drone_idx >= static_cast<int>(drones_.size())) {
        return 0.0;
    }

    // We want to both read and update per-drone state (prev_dist) in a const method.
    const auto& d_const = drones_[drone_idx];
    auto& d             = const_cast<InterceptDroneState&>(d_const);

    Vector<3> p_rel = d.target_p - current_state.p;
    Vector<3> v_ego = current_state.v;

    double dist = p_rel.norm();
    double vmag = v_ego.norm();

    // LOS closing speed (positive when moving toward target)
    double closing_speed = 0.0;
    if (dist > 1e-6) {
        closing_speed = -p_rel.normalized().dot(v_ego);
    }

    // Distance progress (previous minus current)
    double dist_progress = 0.0;
    if (d.prev_dist > 0.0 && dist > 0.0) {
        dist_progress = d.prev_dist - dist;   // >0 if we are getting closer
    }

    // Angle alignment (cos(theta) between LOS and velocity)
    double angle_reward = 0.0;
    if (vmag > 1e-3 && dist > 1e-3) {
        Vector<3> u     = p_rel.normalized();
        Vector<3> v_hat = v_ego / vmag;
        double cos_angle = u.dot(v_hat);      // +1 toward, -1 away
        angle_reward     = angle_align_scale_ * cos_angle;
    }

    // Phase-dependent weights
    double w_closing = vel_closing_scale_;
    double w_rates   = accel_penalty_scale_;

    if (dist > far_dist_thresh_) {
        // Far: emphasize closing, less penalty on rates
        w_closing *= 1.2;
        w_rates   *= 0.5;
    } else if (dist < near_dist_thresh_) {
        // Near: less “charge”, more control smoothness
        w_closing *= 0.5;
        w_rates   *= 2.0;
    }

    double reward = 0.0;

    // 1) Progress toward target
    reward += k_dist_progress_ * dist_progress;

    // 2) LOS closing speed
    reward += w_closing * closing_speed;

    // 3) Attitude / rate penalty
    reward += w_rates * current_state.w.squaredNorm();

    // 4) Angle alignment
    reward += angle_reward;

    // 5) Capture event
    if (task_result.passed_goal) {
        reward += success_reward_;
    }

    // 6) Crash / sim termination
    if (done_from_sim_status) {
        reward += crash_penalty_;
    }

    // NOTE: timeout_penalty_ could be applied via sim status if you distinguish timeout vs crash.

    // Update prev_dist for next step
    d.prev_dist = dist;

    return reward;
}

// ----------------------------------------------------------------------------
// Episode end / done
// ----------------------------------------------------------------------------

void TaskInterception::processEndOfEpisode(int /*drone_idx*/,
                                           const std::vector<double>& /*episode_rewards*/) {
    // No per-episode bookkeeping yet; hook for logging if desired
}

bool TaskInterception::onIsDone(const agi::QuadState& state, bool evalMode_) {
    // End episode if drone hits the ground
    if (state.p.z() < 0.1) {
        return true;
    }

    // Time-based termination
    double max_t = max_episode_time_;
    // Similar to RacingTask: allow longer in eval if desired
    if (!evalMode_) {
        if (state.t > max_t) return true;
    } else {
        if (state.t > 4.0 * max_t) return true;
    }

    return false;
}

} // namespace agi_env

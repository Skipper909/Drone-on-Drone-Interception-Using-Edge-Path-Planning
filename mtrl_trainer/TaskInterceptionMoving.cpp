#include "TaskInterceptionMoving.hpp"
#include "AgiSimBatch.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace agi_env {

static inline double computeAdaptiveTcaMax(const agi::Vector<3>& p_rel,
                                           const agi::Vector<3>& v_rel,
                                           double max_episode_time_s)
{
    // Adaptive horizon: scale with distance and relative speed.
    // Keeps closest-approach features/rewards well-conditioned across engagement regimes.
    const double dist  = p_rel.norm();
    const double speed = v_rel.norm();

    // Closing speed along line-of-sight (positive when closing).
    double closing = 0.0;
    if (dist > 1e-6) {
        closing = -(p_rel.dot(v_rel)) / dist;
    }

    // Effective speed term (avoid division by tiny values).
    const double eff_speed = std::max(0.5, std::max(speed, closing));

    // Nominal horizon: "time to cover current separation" + small buffer.
    double horizon = 0.5 + dist / eff_speed;

    // Clamp to practical range.
    horizon = std::clamp(horizon, 0.25, std::max(0.25, max_episode_time_s));
    return horizon;
}

using agi::QuadState;
using agi::Vector;
using agi::Scalar;

// -----------------------------------------------------------------------------
// Toggle reward variant here.
//
// 0 = "Current + predictive hint" (recommended default)
// 1 = "Advanced predictive" (d_ca and t_ca are more central)
//
// Build flag example:
//   add_compile_definitions(INTERCEPT_USE_ADVANCED_PREDICTIVE_REWARD=1)
// or:
//   target_compile_definitions(your_target PRIVATE INTERCEPT_USE_ADVANCED_PREDICTIVE_REWARD=1)
// -----------------------------------------------------------------------------
#ifndef INTERCEPT_USE_ADVANCED_PREDICTIVE_REWARD
#define INTERCEPT_USE_ADVANCED_PREDICTIVE_REWARD 0
#endif

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
int TaskInterceptionMoving::getTaskID() const
{
    return 3; // unique id
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
    prev_dca_m_.assign(num_drones_, 0.0);

    // Gate center from first gate (if available)
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

    for (auto& d : drones_) {
        d.target_p = gate_center_world_;
        d.target_v = Vector<3>::Zero();
        d.elapsed_time_s = 0.0;
    }

    // Load params
    capture_radius_     = reward_params.INTERCEPTION_CAPTURE_RADIUS;   // set this to 2.0 in yaml
    max_episode_time_s_ = reward_params.INTERCEPTION_MAX_EPISODE_TIME_S;

    distance_scale_        = reward_params.INTERCEPTION_DISTANCE_SCALE;
    dist_delta_scale_      = reward_params.INTERCEPTION_DIST_DELTA_SCALE;
    vel_closing_scale_     = reward_params.INTERCEPTION_VEL_CLOSING_SCALE;
    time_penalty_per_step_ = reward_params.INTERCEPTION_TIME_PENALTY_PER_STEP; // POSITIVE

    success_reward_ = reward_params.INTERCEPTION_SUCCESS_REWARD;
    crash_penalty_  = reward_params.INTERCEPTION_CRASH_PENALTY;

    std::cout << "[TaskInterceptionMoving] init:"
              << " capture_radius=" << capture_radius_
              << " max_time_s=" << max_episode_time_s_
              << " success_reward=" << success_reward_
              << " dist_delta_scale=" << dist_delta_scale_
              << " distance_scale=" << distance_scale_
              << " time_penalty=" << time_penalty_per_step_
              << std::endl;
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
// Target position at time t
// -----------------------------------------------------------------------------
Vector<3> TaskInterceptionMoving::getTargetPosWorld(const QuadState& state, int drone_idx) const
{
    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return gate_center_world_;
    }
    const auto& d = drones_.at(drone_idx);
    return d.target_p + d.target_v * state.t;
}

// -----------------------------------------------------------------------------
// Observations
// -----------------------------------------------------------------------------
Vector<TASK_SPECIFIC_OBS_SIZE>
TaskInterceptionMoving::getTaskSpecificObservation(const QuadState& state, int drone_idx) const
{
    Vector<TASK_SPECIFIC_OBS_SIZE> obs;
    obs.setZero();

    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return obs;
    }

    const auto& d = drones_.at(drone_idx);

    const Vector<3> target_p = getTargetPosWorld(state, drone_idx);
    const Vector<3> p_rel    = target_p - state.p;     // [m]
    const Vector<3> v_rel    = d.target_v - state.v;   // [m/s]

    // Scale to roughly [-1,1]
    constexpr double POS_SCALE = 300.0;
    constexpr double VEL_SCALE = 30.0;

    Vector<3> p_rel_scaled = p_rel / POS_SCALE;
    Vector<3> v_rel_scaled = v_rel / VEL_SCALE;

    for (int i = 0; i < 3; ++i) {
        p_rel_scaled[i] = std::max(-1.0, std::min(1.0, static_cast<double>(p_rel_scaled[i])));
        v_rel_scaled[i] = std::max(-1.0, std::min(1.0, static_cast<double>(v_rel_scaled[i])));
    }

    // Predict closest approach (observation hints)
    double t_ca = 0.0;
    double d_ca = p_rel.norm();
    Vector<3> p_rel_ca = p_rel;

    const double TCA_MAX = std::max(1e-3, static_cast<double>(max_episode_time_s_));
    const double v_rel_norm_sq = v_rel.squaredNorm();
    if (v_rel_norm_sq > 1e-6) {
        t_ca = -p_rel.dot(v_rel) / v_rel_norm_sq;
        if (t_ca < 0.0)          t_ca = 0.0;
        else if (t_ca > TCA_MAX) t_ca = TCA_MAX;

        p_rel_ca = p_rel + v_rel * t_ca;
        d_ca = p_rel_ca.norm();
    }

    const double t_ca_norm = std::max(0.0, std::min(1.0, t_ca / TCA_MAX));
    const double d_ca_norm = std::max(0.0, std::min(1.0, d_ca / POS_SCALE));

    Vector<3> p_rel_ca_scaled = p_rel_ca / POS_SCALE;
    for (int i = 0; i < 3; ++i) {
        p_rel_ca_scaled[i] = std::max(-1.0, std::min(1.0, static_cast<double>(p_rel_ca_scaled[i])));
    }

    // Layout (if capacity allows):
    //   [0:3]  p_rel_scaled
    //   [3:6]  v_rel_scaled
    //   [6]    t_ca_norm
    //   [7]    d_ca_norm
    //   [8:11] p_rel_ca_scaled
    if constexpr (TASK_SPECIFIC_OBS_SIZE >= 3)  obs.template segment<3>(0) = p_rel_scaled;
    if constexpr (TASK_SPECIFIC_OBS_SIZE >= 6)  obs.template segment<3>(3) = v_rel_scaled;
    if constexpr (TASK_SPECIFIC_OBS_SIZE >= 7)  obs[6] = static_cast<Scalar>(t_ca_norm);
    if constexpr (TASK_SPECIFIC_OBS_SIZE >= 8)  obs[7] = static_cast<Scalar>(d_ca_norm);
    if constexpr (TASK_SPECIFIC_OBS_SIZE >= 11) obs.template segment<3>(8) = p_rel_ca_scaled;

    return obs;
}

Vector<TASK_SPECIFIC_OBS_SIZE>
TaskInterceptionMoving::scaleAndClipTaskSpecificObservation(const Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const
{
    return task_obs_raw; // already scaled in getTaskSpecificObservation()
}

// -----------------------------------------------------------------------------
// Reset
// -----------------------------------------------------------------------------
QuadState TaskInterceptionMoving::resetDrone(int drone_idx, std::mt19937& rng, bool evalMode)
{
    QuadState s;
    s.setZero();

    s.p = Vector<3>(0.0, 0.0, 1.5);
    s.v.setZero();
    s.w.setZero();
    s.q() = agi::Quaternion::Identity();
    s.t = 0.0;

    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return s;
    }

    auto& d = drones_[drone_idx];

    if (evalMode) {
        d.target_p = gate_center_world_;
        // keep d.target_v from setTargetVelocityWorld(...)
    } else {
        // Training: randomize target start
        std::uniform_real_distribution<double> x_dist(120.0, 180.0);
        std::uniform_real_distribution<double> y_dist(-40.0, 40.0);
        std::uniform_real_distribution<double> z_dist(80.0, 120.0);
        d.target_p = Vector<3>(x_dist(rng), y_dist(rng), z_dist(rng));

        // Training: mild velocity jitter around configured base (robustness)
        Vector<3> base_v = d.target_v;
        if (base_v.norm() < 1e-3) {
            base_v = Vector<3>(-13.0, 0.0, 0.0); // was (-1.0, 0.0, 0.0)
        }

        std::uniform_real_distribution<double> speed_scale_dist(0.7, 1.3);
        std::uniform_real_distribution<double> yaw_jitter_deg_dist(-15.0, 15.0);
        std::uniform_real_distribution<double> vz_jitter_dist(-0.2, 0.2);

        const double speed_scale = speed_scale_dist(rng);
        const double yaw = (yaw_jitter_deg_dist(rng) * M_PI / 180.0);

        const double c = std::cos(yaw);
        const double sY = std::sin(yaw);

        Vector<3> v = base_v;
        const double vx = v.x();
        const double vy = v.y();
        v.x() = c * vx - sY * vy;
        v.y() = sY * vx + c * vy;

        v *= speed_scale;
        v.z() += vz_jitter_dist(rng);

        d.target_v = v;
    }

    d.elapsed_time_s = 0.0;

    prev_dist_m_[drone_idx] = 0.0;
    prev_dca_m_[drone_idx]  = 0.0;

    return s;
}

// -----------------------------------------------------------------------------
// Step / termination
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

    const Vector<3> target_p = getTargetPosWorld(current_state, drone_idx);
    const double dist = (target_p - current_state.p).norm();

    if (dist <= capture_radius_) {
        result.missed_goal   = true;
        result.lap_completed = true;
        return result;
    }

    if (current_state.t >= max_episode_time_s_) {
        result.missed_goal   = true;
        result.lap_completed = false;
        return result;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Predict helper
// -----------------------------------------------------------------------------
static inline void predictClosestApproach(const Vector<3>& p_rel,
                                          const Vector<3>& v_rel,
                                          double tca_max,
                                          double& t_ca_out,
                                          double& d_ca_out)
{
    double t_ca = 0.0;
    double d_ca = p_rel.norm();

    const double v_rel_norm_sq = v_rel.squaredNorm();
    if (v_rel_norm_sq > 1e-6) {
        t_ca = -p_rel.dot(v_rel) / v_rel_norm_sq;
        if (t_ca < 0.0)          t_ca = 0.0;
        else if (t_ca > tca_max) t_ca = tca_max;

        const Vector<3> p_rel_ca = p_rel + v_rel * t_ca;
        d_ca = p_rel_ca.norm();
    }

    t_ca_out = t_ca;
    d_ca_out = d_ca;
}

// -----------------------------------------------------------------------------
// Reward
// -----------------------------------------------------------------------------
double TaskInterceptionMoving::calculateReward(const QuadState& /*prev_state*/,
                                               const ActionScaledType& /*prev_action_s*/,
                                               const ActionScaledType& /*current_action_s*/,
                                               const QuadState& current_state,
                                               const TaskStepResult& task_result,
                                               bool done_from_sim_status,
                                               int drone_idx) const
{
    if (drone_idx < 0 || drone_idx >= num_drones_) {
        return 0.0;
    }

    const auto& d = drones_.at(drone_idx);

    const Vector<3> target_p = getTargetPosWorld(current_state, drone_idx);
    const Vector<3> p_rel    = target_p - current_state.p;
    const Vector<3> v_rel    = d.target_v - current_state.v;

    const double dist = p_rel.norm();

    // Predict closest approach
    const double TCA_MAX = std::max(1e-3, static_cast<double>(max_episode_time_s_));
    double t_ca = 0.0, d_ca = dist;
    predictClosestApproach(p_rel, v_rel, TCA_MAX, t_ca, d_ca);

    // Dist progress
    double& prev_dist = prev_dist_m_.at(drone_idx);
    if (prev_dist <= 1e-6) prev_dist = dist;
    const double delta_dist = prev_dist - dist;
    prev_dist = dist;

    // DCA progress
    double& prev_dca = prev_dca_m_.at(drone_idx);
    if (prev_dca <= 1e-6) prev_dca = d_ca;
    const double delta_dca = prev_dca - d_ca;
    prev_dca = d_ca;

    double reward = 0.0;

    // Close-in progress gain: amplifies *progress* rewards as we get closer.
    // This avoids a local optimum where the agent "parks" near ~3 m.
    const double close_gain = 1.0 + 6.0 / (dist + 1.0);

#if INTERCEPT_USE_ADVANCED_PREDICTIVE_REWARD
    // Advanced: mix progress in dist + d_ca, penalize d_ca and t_ca
    constexpr double ALPHA = 0.5;
    double effective_progress = 0.0;
    if (delta_dist > 0.0) effective_progress += ALPHA * delta_dist;
    if (delta_dca  > 0.0) effective_progress += (1.0 - ALPHA) * delta_dca;

    if (effective_progress > 0.0) {
        reward += dist_delta_scale_ * effective_progress * close_gain;
    }

    constexpr double MAX_DIST = 300.0;
    const double dca_norm = std::min(d_ca / MAX_DIST, 1.0);
    reward -= distance_scale_ * dca_norm;

    const double tca_norm = std::max(0.0, std::min(1.0, t_ca / TCA_MAX));
    reward -= 0.5 * distance_scale_ * tca_norm;

    reward -= time_penalty_per_step_;

#else
    // Current: standard dist progress + small predictive hint on d_ca
    if (delta_dist > 0.0) {
        reward += dist_delta_scale_ * delta_dist * close_gain;
    }
    if (delta_dca > 0.0) {
        reward += (0.25 * dist_delta_scale_) * delta_dca * close_gain;
    }

    constexpr double MAX_DIST = 300.0;
    const double dist_norm = std::min(dist / MAX_DIST, 1.0);
    reward -= distance_scale_ * dist_norm;

    reward -= time_penalty_per_step_;
#endif

    // Optional closing-speed shaping
    if (vel_closing_scale_ != 0.0) {
        double closing_speed = 0.0;
        if (dist > 1e-6) {
            closing_speed = -(p_rel.dot(v_rel) / dist); // positive when closing
        }
        constexpr double MAX_CLOSE = 40.0;
        double closing_norm = closing_speed / MAX_CLOSE;
        closing_norm = std::max(-1.0, std::min(1.0, closing_norm));
        reward += vel_closing_scale_ * closing_norm;
    }

        // Anti-loiter close-in: inside 10 m, require actual progress (otherwise pay a penalty).
    // IMPORTANT: do NOT add per-step "being close" bonuses; they create a stable loitering optimum.
    constexpr double ANTI_R = 10.0;
    constexpr double ANTI_EPS = 0.002;  // meters per step
    if (dist < ANTI_R) {
        if (delta_dist < ANTI_EPS && delta_dca < ANTI_EPS) {
            reward -= 1.0;
        }
    }

// Terminal success bonus (only if truly within capture radius)
    if (task_result.missed_goal && task_result.lap_completed && dist <= capture_radius_) {
        reward += success_reward_;
    }

    // Crash penalty (sim status)
    if (done_from_sim_status) {
        reward += crash_penalty_;
    }

    return reward;
}

// -----------------------------------------------------------------------------
// Episode end
// -----------------------------------------------------------------------------
void TaskInterceptionMoving::processEndOfEpisode(int drone_idx,
                                                 const std::vector<double>& /*episode_rewards*/)
{
    if (drone_idx >= 0 && drone_idx < static_cast<int>(prev_dist_m_.size())) {
        prev_dist_m_[drone_idx] = 0.0;
        prev_dca_m_[drone_idx]  = 0.0;
    }
}

bool TaskInterceptionMoving::onIsDone(const QuadState& /*state*/, bool /*evalMode*/)
{
    return false; // handled by onStep() + sim status
}

}  // namespace agi_env
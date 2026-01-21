#include "TaskTerminalPursuit.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace agi_env {

TaskTerminalPursuit::TaskTerminalPursuit() = default;

int TaskTerminalPursuit::getTaskID() const {
    // Keep stable: InterceptionMoving can stay 0 in your main env,
    // TerminalPursuit is 1 in the terminal env.
    return 1;
}

void TaskTerminalPursuit::initialize(const RewardParams &reward_params,
                                     const std::vector<std::vector<std::vector<double>>> &track_layout,
                                     int num_drones) {
    (void)track_layout;
    reward_params_ = &reward_params;
    num_drones_ = num_drones;
    target_positions_.assign(num_drones_, agi::Vector<3>::Zero());
    target_velocities_.assign(num_drones_, agi::Vector<3>::Zero());
}

void TaskTerminalPursuit::sampleEngagementBubble(std::mt19937 &rng,
                                                 agi::Vector<3> &rel_pos_out_world,
                                                 agi::Vector<3> &target_vel_out_world,
                                                 const agi::Vector<3> &drone_vel_world) const {
    std::uniform_real_distribution<double> r_dist(r_min_close_, r_max_close_);
    std::uniform_real_distribution<double> az(-M_PI, M_PI);
    std::uniform_real_distribution<double> el_deg(-max_elev_deg_, max_elev_deg_);
    std::normal_distribution<double> dv_dist(target_speed_delta_mean_, target_speed_delta_std_);

    const double r = r_dist(rng);
    const double phi = az(rng);
    const double theta = el_deg(rng) * (M_PI / 180.0);

    rel_pos_out_world(0) = r * std::cos(theta) * std::cos(phi);
    rel_pos_out_world(1) = r * std::cos(theta) * std::sin(phi);
    rel_pos_out_world(2) = r * std::sin(theta);

    // Target velocity: "close-range evasive" as a perturbation around drone velocity
    // so the endgame is mostly about *correction under momentum*.
    agi::Vector<3> dv = agi::Vector<3>::Zero();
    dv(0) = dv_dist(rng);
    dv(1) = dv_dist(rng);
    dv(2) = 0.25 * dv_dist(rng);

    target_vel_out_world = drone_vel_world + dv;
}

agi::QuadState TaskTerminalPursuit::resetDrone(int drone_idx,
                                               std::mt19937 &rng,
                                               bool evalMode_) {
    (void)evalMode_;

    agi::QuadState s;
    s.p.setZero();
    s.p(2) = start_alt_m_;

    // Random yaw
    std::uniform_real_distribution<double> yaw_dist(-M_PI, M_PI);
    const double yaw = yaw_dist(rng);

    // QuadState q appears to be [x, y, z, w] (see SerializableQuadState in AgiSimBatch.hpp)
    //s.q.setZero();
    //s.q(2) = std::sin(0.5 * yaw); // z
   // s.q(3) = std::cos(0.5 * yaw); // w

    // Start moving forward in yaw direction
    std::normal_distribution<double> v_dist(drone_speed_mean_, drone_speed_std_);
    const double vmag = std::max(0.0, v_dist(rng));
    s.v.setZero();
    s.v(0) = vmag * std::cos(yaw);
    s.v(1) = vmag * std::sin(yaw);
    s.v(2) = 0.0;

    s.w.setZero();
    s.t = 0.0;

    // Spawn target in a close bubble around the drone, with motion correlated to drone motion.
    agi::Vector<3> rel_pos, tgt_vel;
    sampleEngagementBubble(rng, rel_pos, tgt_vel, s.v);

    target_positions_[drone_idx] = s.p + rel_pos;
    target_velocities_[drone_idx] = tgt_vel;

    return s;
}

agi::Vector<TASK_SPECIFIC_OBS_SIZE>
TaskTerminalPursuit::getTaskSpecificObservation(const agi::QuadState &current_state,
                                                int drone_idx) const {
    agi::Vector<TASK_SPECIFIC_OBS_SIZE> obs;
    obs.setZero();

    const agi::Vector<3> rel_pos = target_positions_[drone_idx] - current_state.p;
    const agi::Vector<3> rel_vel = target_velocities_[drone_idx] - current_state.v;

    // Pack:
    // 0..2  rel_pos
    // 3..5  rel_vel
    // 6..8  target_vel
    // 9     distance
    obs.segment<3>(0) = rel_pos;
    obs.segment<3>(3) = rel_vel;
    obs.segment<3>(6) = target_velocities_[drone_idx];
    obs(9) = rel_pos.norm();

    // Remaining dims left zero for compatibility with 24-dim task obs.
    return obs;
}

agi::Vector<TASK_SPECIFIC_OBS_SIZE>
TaskTerminalPursuit::scaleAndClipTaskSpecificObservation(
    const agi::Vector<TASK_SPECIFIC_OBS_SIZE> &task_obs_raw) const {
    // Keep simple and stable: mild clipping only.
    agi::Vector<TASK_SPECIFIC_OBS_SIZE> x = task_obs_raw;
    for (int i = 0; i < TASK_SPECIFIC_OBS_SIZE; ++i) {
        x(i) = std::max(-50.0, std::min(50.0, x(i)));
    }
    return x;
}

bool TaskTerminalPursuit::isCapture(const agi::Vector<3> &rel_pos_world) const {
    return rel_pos_world.norm() <= capture_radius_;
}

bool TaskTerminalPursuit::isEscape(const agi::Vector<3> &rel_pos_world) const {
    return rel_pos_world.norm() >= fail_radius_;
}

double TaskTerminalPursuit::computeReward(const agi::Vector<3> &rel_pos_world,
                                          const agi::Vector<3> &rel_vel_world,
                                          const agi::QuadState &quad_state) const {
    const double d = std::max(1e-6, rel_pos_world.norm());
    const double closing = rel_pos_world.dot(rel_vel_world) / d; // negative when closing

    // Facing alignment proxy: use velocity direction vs LOS (works even if yaw isn't directly used).
    double align = 0.0;
    const double vnorm = quad_state.v.norm();
    if (vnorm > 1e-6) {
        const agi::Vector<3> vhat = quad_state.v / vnorm;
        const agi::Vector<3> loshat = rel_pos_world / d;
        align = vhat.dot(loshat); // [-1,1]
    }

    const double omega = quad_state.w.norm();

    double r = 0.0;
    r += w_dist_lin_ * d;
    r += w_dist_quad_ * d * d;
    r += w_inv_ / (d + 0.10);          // strong incentive below ~1 m
    r += w_closing_ * closing;
    r += w_align_ * align;
    r += w_time_;
    r += w_omega_ * omega;

    return r;
}

TaskStepResult TaskTerminalPursuit::onStep(int drone_idx, int step_count,
                                          const agi::QuadState &prev_state,
                                          const agi::QuadState &current_state,
                                          agi::Vector<6> &vio_drift,
                                          double gate_reset_pos_std,
                                          double gate_reset_att_std) {
    (void)prev_state;
    (void)vio_drift;
    (void)gate_reset_pos_std;
    (void)gate_reset_att_std;

    // Propagate target (simple kinematics)
    if (drone_idx >= 0 && drone_idx < (int)target_positions_.size()) {
        // Approx dt from sim time delta (more robust than assuming a fixed dt)
        const double dt = std::max(0.0, current_state.t - prev_state.t);
        target_positions_[drone_idx] += target_velocities_[drone_idx] * dt;
    }

    TaskStepResult out;

    const agi::Vector<3> rel_pos = target_positions_[drone_idx] - current_state.p;

    if (isCapture(rel_pos)) {
        out.lap_completed = true; // "success" flag used by AgiSimBatch
        return out;
    }

    if (isEscape(rel_pos) || step_count >= max_steps_) {
        out.missed_goal = true;   // "fail" flag used by AgiSimBatch
        return out;
    }

    return out;
}

double TaskTerminalPursuit::calculateReward(const agi::QuadState &prev_state,
                                           const ActionScaledType &prev_action_s,
                                           const ActionScaledType &current_action_s,
                                           const agi::QuadState &current_state,
                                           const TaskStepResult &task_result,
                                           bool done_from_sim_status,
                                           int drone_idx) const {
    (void)prev_state;
    (void)prev_action_s;
    (void)current_action_s;

    if (done_from_sim_status) {
        return R_fail_;
    }
    if (task_result.lap_completed) {
        return R_capture_;
    }
    if (task_result.missed_goal) {
        return R_fail_;
    }

    const agi::Vector<3> rel_pos = target_positions_[drone_idx] - current_state.p;
    const agi::Vector<3> rel_vel = target_velocities_[drone_idx] - current_state.v;
    return computeReward(rel_pos, rel_vel, current_state);
}

void TaskTerminalPursuit::processEndOfEpisode(int drone_idx,
                                             const std::vector<double> &episode_rewards) {
    (void)drone_idx;
    (void)episode_rewards;
}

bool TaskTerminalPursuit::onIsDone(const agi::QuadState &state, bool evalMode_) {
    (void)state;
    (void)evalMode_;
    return false; // termination is handled via missed_goal/lap_completed in onStep()
}

} // namespace agi_env

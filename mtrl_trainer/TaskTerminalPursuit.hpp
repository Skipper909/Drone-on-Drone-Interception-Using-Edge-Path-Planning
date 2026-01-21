#pragma once

#include "ITask.hpp"
#include <random>
#include <vector>

namespace agi_env {

class TaskTerminalPursuit final : public ITask {
public:
    TaskTerminalPursuit();

    int getTaskID() const override;

    void initialize(const RewardParams &reward_params,
                    const std::vector<std::vector<std::vector<double>>> &track_layout,
                    int num_drones) override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    getTaskSpecificObservation(const agi::QuadState &current_state,
                               int drone_idx) const override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    scaleAndClipTaskSpecificObservation(
        const agi::Vector<TASK_SPECIFIC_OBS_SIZE> &task_obs_raw) const override;

    agi::QuadState resetDrone(int drone_idx,
                              std::mt19937 &rng,
                              bool evalMode_) override;

    TaskStepResult onStep(int drone_idx, int step_count,
                          const agi::QuadState &prev_state,
                          const agi::QuadState &current_state,
                          agi::Vector<6> &vio_drift,
                          double gate_reset_pos_std,
                          double gate_reset_att_std) override;

    double calculateReward(const agi::QuadState &prev_state,
                           const ActionScaledType &prev_action_s,
                           const ActionScaledType &current_action_s,
                           const agi::QuadState &current_state,
                           const TaskStepResult &task_result,
                           bool done_from_sim_status,
                           int drone_idx) const override;

    void processEndOfEpisode(int drone_idx,
                             const std::vector<double> &episode_rewards) override;

    bool onIsDone(const agi::QuadState &state,
                  bool evalMode_) override;

private:
    void sampleEngagementBubble(std::mt19937 &rng,
                                agi::Vector<3> &rel_pos_out_world,
                                agi::Vector<3> &target_vel_out_world,
                                const agi::Vector<3> &drone_vel_world) const;

    double computeReward(const agi::Vector<3> &rel_pos_world,
                         const agi::Vector<3> &rel_vel_world,
                         const agi::QuadState &quad_state) const;

    bool isCapture(const agi::Vector<3> &rel_pos_world) const;
    bool isEscape(const agi::Vector<3> &rel_pos_world) const;

private:
    // Close-range regime
    double r_min_close_{1.0};
    double r_max_close_{5.0};
    double max_elev_deg_{35.0};

    // Moving start (in-air)
    double start_alt_m_{10.0};
    double drone_speed_mean_{10.0};
    double drone_speed_std_{2.0};
    double target_speed_delta_mean_{0.0};
    double target_speed_delta_std_{2.0};

    // Terminal thresholds
    double capture_radius_{1.0}; // success if dist <= 1.0 m
    double fail_radius_{8.0};    // fail if dist >= 8.0 m
    int max_steps_{400};         // ~8 s at dt=0.02

    // Reward shaping
    double w_dist_lin_{-0.5};
    double w_dist_quad_{-0.15};
    double w_inv_{+1.5};         // strong near-target incentive
    double w_closing_{-0.25};    // negative -> reward closing (closing speed negative)
    double w_align_{+0.05};
    double w_time_{-0.01};
    double w_omega_{-0.01};

    double R_capture_{+50.0};
    double R_fail_{-50.0};

    const RewardParams *reward_params_{nullptr};
    int num_drones_{0};

    std::vector<agi::Vector<3>> target_positions_;
    std::vector<agi::Vector<3>> target_velocities_;
};

} // namespace agi_env

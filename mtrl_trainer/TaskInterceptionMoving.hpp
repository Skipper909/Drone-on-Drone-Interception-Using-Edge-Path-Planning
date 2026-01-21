#pragma once

#include "ITask.hpp"

#include <random>
#include <vector>

namespace agi_env {

class AgiSimBatch;  // forward declaration

/**
 * Interception task against a moving target.
 *
 * Target model:
 *   target(t) = target_start_p + target_v_world * t
 *
 * Policy observes:
 *   - instantaneous relative position p_rel (world)
 *   - instantaneous relative velocity v_rel (world)
 *   - predictive hints (optional fields): t_ca, d_ca, p_rel at closest approach
 *
 * Termination:
 *   - capture: 3D distance <= capture_radius_
 *   - timeout: state.t >= max_episode_time_s_
 */
class TaskInterceptionMoving final : public ITask {
public:
    explicit TaskInterceptionMoving(AgiSimBatch& sim);

    int getTaskID() const override;

    void initialize(const RewardParams& reward_params,
                    const std::vector<std::vector<std::vector<double>>>& track_layout,
                    int num_drones) override;

    // Configure target velocity (world frame). Training can jitter around this.
    void setTargetVelocityWorld(const agi::Vector<3>& v_world);

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    getTaskSpecificObservation(const agi::QuadState& state, int drone_idx) const override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    scaleAndClipTaskSpecificObservation(const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const override;

    agi::QuadState resetDrone(int drone_idx, std::mt19937& rng, bool evalMode) override;

    TaskStepResult onStep(int drone_idx,
                          int step_count,
                          const agi::QuadState& prev_state,
                          const agi::QuadState& current_state,
                          agi::Vector<6>& vio_drift,
                          double gate_reset_pos_std,
                          double gate_reset_att_std) override;

    double calculateReward(const agi::QuadState& prev_state,
                           const ActionScaledType& prev_action_s,
                           const ActionScaledType& current_action_s,
                           const agi::QuadState& current_state,
                           const TaskStepResult& task_result,
                           bool done_from_sim_status,
                           int drone_idx) const override;

    void processEndOfEpisode(int drone_idx, const std::vector<double>& episode_rewards) override;

    bool onIsDone(const agi::QuadState& state, bool evalMode) override;

private:
    struct DroneData {
        agi::Vector<3> target_p{agi::Vector<3>::Zero()};  // start position [m] (world)
        agi::Vector<3> target_v{agi::Vector<3>::Zero()};  // velocity [m/s] (world)
        double elapsed_time_s{0.0};
    };

    agi::Vector<3> getTargetPosWorld(const agi::QuadState& state, int drone_idx) const;

private:
    AgiSimBatch& sim_;
    int num_drones_{0};

    std::vector<DroneData> drones_;
    agi::Vector<3> gate_center_world_{agi::Vector<3>::Zero()};

    // Termination params
    double capture_radius_{1.0};       // [m] (3D)
    double max_episode_time_s_{25.0};  // [s]

    // Reward scales (loaded from RewardParams)
    double distance_scale_{0.4};
    double dist_delta_scale_{5.0};
    double vel_closing_scale_{0.0};
    double time_penalty_per_step_{0.02};   // IMPORTANT: treat as POSITIVE penalty; code subtracts it.

    double success_reward_{800.0};
    double crash_penalty_{-200.0};

    // Curriculum (loaded from RewardParams). If not present in the loader, this will remain 0.
    // Convention used here:
    //   0-1 = easy, 2 = medium, >=3 = hard (original ranges).
    // Per-drone shaping memory
    mutable std::vector<double> prev_dist_m_;  // previous true distance [m]
    mutable std::vector<double> prev_dca_m_;   // previous predicted closest-approach distance [m]

    mutable std::mt19937 rng_;
};

}  // namespace agi_env
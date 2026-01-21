#pragma once

#include "ITask.hpp"
#include <random>
#include <vector>

namespace agi_env {

class AgiSimBatch;  // forward declaration

class TaskInterceptionMoving : public ITask {
public:
    explicit TaskInterceptionMoving(AgiSimBatch& sim);

    // --- ITask interface implementation ---
    int getTaskID() const override;

    void initialize(const RewardParams& reward_params,
                    const std::vector<std::vector<std::vector<double>>>& track_layout,
                    int num_drones) override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    getTaskSpecificObservation(const agi::QuadState& state,
                               int drone_idx) const override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE>
    scaleAndClipTaskSpecificObservation(
            const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const override;

    agi::QuadState resetDrone(int drone_idx,
                              std::mt19937& rng,
                              bool evalMode) override;

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

    void processEndOfEpisode(int drone_idx,
                             const std::vector<double>& episode_rewards) override;

    bool onIsDone(const agi::QuadState& state, bool evalMode) override;

    // Configure target velocity (shared for all drones, world frame)
    void setTargetVelocityWorld(const agi::Vector<3>& v_world);

private:
    struct DroneData {
        agi::Vector<3> target_p{agi::Vector<3>::Zero()};
        agi::Vector<3> target_v{agi::Vector<3>::Zero()};
        double elapsed_time_s{0.0};
    };

    // Target position as function of time: start + v * t  (world frame)
    agi::Vector<3> getTargetPosWorld(const agi::QuadState& state,
                                     int drone_idx) const;

    AgiSimBatch&           sim_;
    int                    num_drones_{0};
    std::vector<DroneData> drones_;

    // Center of first gate in world coords (default target start)
    agi::Vector<3> gate_center_world_{agi::Vector<3>::Zero()};

    // Reward / termination parameters
    double capture_radius_{2.50}; //stable 3
    double max_episode_time_s_{25.0};

    double distance_scale_{0.4};
    double dist_delta_scale_{5.0}; //was 2.5
    double vel_closing_scale_{0.0}; // was 0.5
    double time_penalty_per_step_{0.02}; //was 2

    double success_reward_{800.0};
    double crash_penalty_{-200.0};

    // Per-drone previous 3D distance (for progress-based shaping)
    mutable std::vector<double> prev_dist_m_;

    mutable std::mt19937 rng_;
};

} // namespace agi_env

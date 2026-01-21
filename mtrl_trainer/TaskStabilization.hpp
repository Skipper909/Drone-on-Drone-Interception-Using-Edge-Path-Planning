#ifndef TASK_STABILIZATION_HPP
#define TASK_STABILIZATION_HPP

#include "ITask.hpp"
#include "AgiSimBatch.hpp" // For RewardParams, assuming it's defined here or in a related header

namespace agi_env {

class StabilizationTask : public ITask {
public:
    StabilizationTask();
    virtual ~StabilizationTask() = default;

	int getTaskID() const override { return 1; }

    void initialize(const RewardParams& reward_params,
                    const std::vector<std::vector<std::vector<double>>>& track_layout,
                    int num_drones) override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> getTaskSpecificObservation(
        const agi::QuadState& current_state,
        int drone_idx) const override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> scaleAndClipTaskSpecificObservation(
        const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const override;


    agi::QuadState resetDrone(int drone_idx, std::mt19937& rng, bool evalMode_) override;

    TaskStepResult onStep(int drone_idx, int step_count,
                              const agi::QuadState& prev_state,
                              const agi::QuadState& current_state,
                              agi::Vector<6>& vio_drift, // Pass drift by reference
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

    bool onIsDone(const agi::QuadState &state, bool evalMode_) override;

private:
    RewardParams reward_params_;
    int num_drones_{0};

    struct DroneTaskState {
        agi::Vector<3> prev_velocity{agi::Vector<3>::Zero()};
        agi::Vector<3> current_acceleration{agi::Vector<3>::Zero()};
    };
    std::vector<DroneTaskState> drone_states_;

    double target_height_z_{5.0};

    long long task_interaction_count_{0};
    int velocity_curriculum_level_{0};
    const long long curriculum_update_interval_{500000};
    const double initial_vel_bound_{2.0};
    const double initial_vel_z_bound_{0.5};
    const double max_vel_bound_{24.0};
    const double max_vel_z_bound_{4.0};
    const double curriculum_increase_factor_{0.10};
};

} // namespace agi_env

#endif // TASK_STABILIZATION_HPP
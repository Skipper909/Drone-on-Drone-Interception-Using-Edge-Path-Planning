#ifndef TASK_INTERCEPTION_HPP
#define TASK_INTERCEPTION_HPP

#include "ITask.hpp"
#include "AgiSimBatch.hpp"
#include <random>

namespace agi_env {

class TaskInterception : public ITask {
public:
    // Keep this ctor signature, since you already had it
    explicit TaskInterception(AgiSimBatch& sim);

    ~TaskInterception() override = default;

    // Task ID: 0 = racing, 1 = stabilization, 2 = interception
    int getTaskID() const override { return 2; }

    // --- Initialization ---
    void initialize(const RewardParams& reward_params,
                    const std::vector<std::vector<std::vector<double>>>& track_layout,
                    int num_drones) override;

    // --- Observations ---
    agi::Vector<TASK_SPECIFIC_OBS_SIZE> getTaskSpecificObservation(
        const agi::QuadState& current_state,
        int drone_idx) const override;

    agi::Vector<TASK_SPECIFIC_OBS_SIZE> scaleAndClipTaskSpecificObservation(
        const agi::Vector<TASK_SPECIFIC_OBS_SIZE>& task_obs_raw) const override;

    // --- Episode reset ---
    agi::QuadState resetDrone(int drone_idx, std::mt19937& rng, bool evalMode_) override;

    // --- Per-step logic ---
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

    bool onIsDone(const agi::QuadState& state, bool evalMode_) override;

private:
    struct InterceptDroneState {
        agi::Vector<3> target_p = agi::Vector<3>::Zero();  // static world target
        double prev_dist        = 0.0;                     // for distance-progress reward
    };

    AgiSimBatch& sim_;                        // kept for future use (moving target / evader)
    RewardParams reward_params_{};            // copy of reward params
    std::vector<InterceptDroneState> drones_; // per-drone interception state

    int num_drones_{0};

    // Target derived from track.json (center of first gate, z forced to 100 m)
    agi::Vector<3> base_target_from_track_ = agi::Vector<3>::Zero();
    bool has_track_target_{false};

    // Hyperparameters for interception
    double capture_radius_      = 1.0;    // [m]
    double max_episode_time_    = 12.0;   // [s]
    double success_reward_      = 5.0;
    double crash_penalty_       = -2.0;
    double timeout_penalty_     = -0.5;

    // New reward shaping parameters
    double k_dist_progress_     = 1.0;    // reward per meter of distance improvement
    double vel_closing_scale_   = 0.02;   // reward on closing speed
    double accel_penalty_scale_ = -0.001; // penalty on body rates
    double angle_align_scale_   = 0.05;   // reward on pointing velocity toward target

    // Distance-based phase thresholds [m]
    double near_dist_thresh_    = 20.0;
    double far_dist_thresh_     = 80.0;

    // Spawn distribution (still available if you want some randomness)
    double min_target_radius_   = 20.0;   // [m] distance from origin (unused in default logic)
    double max_target_radius_   = 200.0;  // [m]

    mutable std::mt19937 rng_;
};

} // namespace agi_env

#endif // TASK_INTERCEPTION_HPP

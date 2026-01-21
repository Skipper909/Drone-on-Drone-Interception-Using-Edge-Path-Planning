#pragma once

#include "AgiSimBatch.hpp"          // re-use RewardParams, StepResultSimple, ObservationType, etc.
#include <memory>

namespace agi_env {

class AgiSimBatchTerminal {
public:
    AgiSimBatchTerminal(int num_drones,
                        const std::string &sim_config_path,
                        const std::string &agi_param_dir,
                        const std::string &sim_base_dir,
                        const std::vector<std::vector<std::vector<double>>> &track_layout_world);

    ~AgiSimBatchTerminal() = default;

    std::vector<StepResultSimple> step(const std::vector<PolicyActionType> &actions_raw);
    std::vector<Observation43Type> reset(const std::vector<bool> &reset_flags);

    double getStates() const;

    void setEval();

    void setNoiseIntensity(double vio_pos_drift_std, double vio_att_drift_std_deg,
                           double gate_reset_pos_std, double gate_reset_att_std_deg);

private:
    ObservationType assembleObservation(int drone_idx);
    agi::Vector<SHARED_OBS_SIZE> assembleSharedObservation(const agi::QuadState &current_state,
                                                           const PolicyActionType &prev_raw_action) const;
    ObservationType scaleAndClipObservation(const ObservationType &obs_raw, int drone_idx) const;

    void _processEndOfEpisodeForDrone(int drone_idx);

    ActionScaledType scaleAction(const PolicyActionType &raw_action) const;
    bool isDone(int i, const QuadState &state) const;

private:
    int num_drones_;
    const SimulatorParams sim_params_;
    const double sim_dt_;
    std::vector<agi::QuadrotorSimulator> simulators_;

    std::vector<agi::QuadState, Eigen::aligned_allocator<agi::QuadState>> current_states_;
    std::vector<agi::QuadState, Eigen::aligned_allocator<agi::QuadState>> previous_states_;
    std::vector<PolicyActionType, Eigen::aligned_allocator<PolicyActionType>> prev_actions_raw_;
    std::vector<ActionScaledType, Eigen::aligned_allocator<ActionScaledType>> prev_actions_scaled_;

    std::shared_ptr<ITask> task_; // always terminal pursuit

    std::vector<std::vector<double>> per_drone_episode_rewards_;
    std::vector<int> per_drone_episode_step_count_;

    std::vector<std::mt19937> rngs_;
    const RewardParams reward_params_;

    bool evalMode_{false};

    std::vector<Vector<6>> vio_drift_offsets_;
    double vio_pos_drift_std_{0.0};
    double vio_att_drift_std_{0.0};
    double gate_reset_pos_std_{0.0};
    double gate_reset_att_std_{0.0};
};

} // namespace agi_env

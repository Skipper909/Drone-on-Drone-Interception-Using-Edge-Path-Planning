#ifndef RACING_TASK_HPP
#define RACING_TASK_HPP

#include "ITask.hpp"
#include "AgiSimBatch.hpp"
#include <mutex>

namespace agi_env {

    class RacingTask : public ITask {
    public:
        RacingTask();

        ~RacingTask() override = default;

		int getTaskID() const override { return 0; }

        void initialize(const RewardParams &reward_params,
                        const std::vector <std::vector<std::vector<double>>>& track_layout,
                        int num_drones) override;

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> getTaskSpecificObservation(
                const agi::QuadState &current_state,
                int drone_idx) const override;

        agi::Vector<TASK_SPECIFIC_OBS_SIZE> scaleAndClipTaskSpecificObservation(
                const agi::Vector<TASK_SPECIFIC_OBS_SIZE> &task_obs_raw) const override;


        agi::QuadState resetDrone(int drone_idx, std::mt19937 &rng, bool evalMode_) override;

        TaskStepResult onStep(int drone_idx, int step_count,
                              const agi::QuadState &prev_state,
                              const agi::QuadState &current_state,
                              agi::Vector<6> &vio_drift, // Pass drift by reference
                              double gate_reset_pos_std,
                              double gate_reset_att_std) override;

        double calculateReward(const agi::QuadState &prev_state,
                               const ActionScaledType &prev_action_s,
                               const ActionScaledType &current_action_s,
                               const agi::QuadState &current_state,
                               const TaskStepResult &task_result,
                               bool done_from_sim_status,
                               int drone_idx) const override;


        void processEndOfEpisode(int drone_idx, const std::vector<double> &episode_rewards) override;

        bool onIsDone(const agi::QuadState &state, bool evalMode_) override;

        double getStates() const;

    private:
        agi::Vector<12> getRelativeGateCornersForDrone(const agi::QuadState &state, int target_gate_idx) const;

        agi::Vector<12> getGateToGateRelativeCorners(int upcoming_gate_idx, int next_next_gate_idx) const;

        agi::Vector<3> getGateCenter(int gate_idx) const;

        GatePassResult checkGatePassDetailed(const agi::QuadState &prev_state,
                                             const agi::QuadState &current_state,
                                             int target_gate_idx) const;

        void _storeGatePassIfGood(int passed_gate_idx, const agi::QuadState &state_at_pass,
                                  double future_accumulated_reward);

        agi::Vector<3> getGateCenter(int gate_idx);

        struct DroneTaskState {
            int target_gate_idx{0};
            double prev_gate_distance{std::numeric_limits<double>::infinity()};
            std::vector <std::tuple<int, int, agi::QuadState>> episode_gate_passes;
        };
        std::vector <DroneTaskState> drone_states_;

        RewardParams reward_params_;
        int num_drones_{0};
        int num_gates_{0};
        std::vector <agi::Vector<3>> gate_centers_world_;
        std::vector <agi::Vector<3>> gate_normals_world_;
        std::vector <std::vector<agi::Vector<3>>> gates_corners_world_;

        std::vector <std::vector<std::pair < agi::QuadState, double>>>
        successful_gate_pass_states_;
        std::vector <std::mutex> gate_pass_states_mutexes_;

        bool use_spawns = false;

        int max_laps_;
        std::vector<int> drone_lap_counts_;
        std::mt19937 rng_;
    };

} // namespace agi_env

#endif // RACING_TASK_HPP
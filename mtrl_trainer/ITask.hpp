#ifndef ITASK_HPP
#define ITASK_HPP

#include "agilib/types/quad_state.hpp"
#include "agilib/math/types.hpp"
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <Eigen/StdVector>


// Forward declarations from AgiSimBatch.hpp
struct RewardParams;
struct StepResultSimple;
using PolicyActionType = agi::Vector<4>;
using ActionScaledType = agi::Vector<4>;

// Define observation vector sizes for clarity, based on the paper
static constexpr int SHARED_OBS_SIZE = 19;
static constexpr int TASK_SPECIFIC_OBS_SIZE = 24;
static constexpr int TOTAL_OBS_SIZE = SHARED_OBS_SIZE + TASK_SPECIFIC_OBS_SIZE; // 43
using ObservationType = agi::Vector<TOTAL_OBS_SIZE>;


namespace agi_env {
    struct TaskStepResult {
        bool passed_goal{false};
        bool missed_goal{false};
        bool lap_completed{false};
        // NEW: generic task termination flag for non-racing tasks
        bool done{false};
    };


    class ITask {
    public:
        virtual ~ITask() = default;

		virtual int getTaskID() const = 0;

        // --- Initialization ---
        virtual void initialize(const RewardParams &reward_params,
                                const std::vector <std::vector<std::vector<double>>>& track_layout,
                                int num_drones) = 0;

        virtual agi::Vector<TASK_SPECIFIC_OBS_SIZE> getTaskSpecificObservation(
                const agi::QuadState &current_state,
                int drone_idx) const = 0;

        virtual agi::Vector<TASK_SPECIFIC_OBS_SIZE> scaleAndClipTaskSpecificObservation(
                const agi::Vector<TASK_SPECIFIC_OBS_SIZE> &task_obs_raw) const = 0;

        virtual agi::QuadState resetDrone(int drone_idx, std::mt19937 &rng, bool evalMode_) = 0;

        virtual TaskStepResult onStep(int drone_idx, int step_count,
                                      const agi::QuadState &prev_state,
                                      const agi::QuadState &current_state,
                                      agi::Vector<6> &vio_drift, // Pass drift by reference
                                      double gate_reset_pos_std,
                                      double gate_reset_att_std) = 0;

        virtual double calculateReward(const agi::QuadState &prev_state,
                                       const ActionScaledType &prev_action_s,
                                       const ActionScaledType &current_action_s,
                                       const agi::QuadState &current_state,
                                       const TaskStepResult &task_result,
                                       bool done_from_sim_status,
                                       int drone_idx) const = 0;

        virtual void processEndOfEpisode(int drone_idx, const std::vector<double> &episode_rewards) = 0;

        virtual bool onIsDone(const agi::QuadState &state, bool evalMode_) = 0;
    };

} // namespace agi_env

#endif // ITASK_HPP
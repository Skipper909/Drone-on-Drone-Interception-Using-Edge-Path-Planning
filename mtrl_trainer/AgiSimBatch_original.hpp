#ifndef AGISIM_BATCH_HPP
#define AGISIM_BATCH_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <tuple> // For std::tuple

#include <mutex>
#include <memory>

#include <random> // For random number generation
#include <optional> // For storing optional states

#include <yaml-cpp/yaml.h>

#include <Eigen/StdVector>

// Headers from standalone library needed for declaration
#include "agilib/math/types.hpp"          // For agi::Vector, agi::Matrix used by QuadState
#include "agilib/simulator/simulator_params.hpp" // For agi::SimulatorParams member
#include "agilib/simulator/model_body_drag.hpp"
#include "agilib/simulator/model_init.hpp"
#include "agilib/simulator/model_motor.hpp"
#include "agilib/simulator/model_propeller_bem.hpp"
#include "agilib/simulator/model_rigid_body.hpp"
#include "agilib/simulator/model_thrust_torque_simple.hpp"
#include "agilib/simulator/quadrotor_simulator.hpp"
#include "agilib/simulator/simulator_params.hpp"
#include "agilib/types/quad_state.hpp"    // For agi::QuadState member/method args

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // For automatic std::vector conversions
#include <pybind11/eigen.h> // If agi::Vector3d etc are Eigen types

#include "ITask.hpp"

// Using namespace for library (optional)
using namespace agi;
namespace py = pybind11;

// Define types used in the public interface

using ActionType = agi::Vector<4>; // Represents 4 motor commands

using ActionScaledType = agi::Vector<4>;   // T, wx, wy, wz (Scaled for Simulator)
using PolicyActionType = agi::Vector<4>;   // Assumed raw action from policy [-1, 1] range typically
using Vector12d = agi::Vector<12>;         // Relative Gate Corners (x,y,z * 4)
using Observation43Type = agi::Vector<43>; // Final Observation type (R15 + RelCorners + PrevRawAction)


struct StepResultSimple {
    Observation43Type observation;
    double reward;
    bool done;
    double time;
    bool success;
	int task_id;
};

template<int N>
inline agi::Vector<N> loadVector(const YAML::Node& node) {
    agi::Vector<N> v;
    std::vector<double> vec = node.as<std::vector<double>>();
    if (vec.size() != N) {
        throw std::runtime_error("Incorrect vector size in YAML config");
    }
    for (int i = 0; i < N; ++i) {
        v(i) = vec[i];
    }
    return v;
}

struct RewardParams {
    // --- Members are now uninitialized ---
    double PROGRESS_WEIGHT;
    double ACTION_SMOOTHNESS_WEIGHT;
    double BODY_RATE_PENALTY_WEIGHT;
    double GATE_PASS_REWARD;
    double TERMINATION_PENALTY;
    double MAX_SIM_TIME_RACING;
	double MAX_SIM_TIME_STABILIZATION;
    double GATE_MISS_BASE_PENALTY;
    double GATE_MISS_MAX_PENALTY;
    double GATE_MISS_DISTANCE_THRESHOLD;
    double MIN_THRUST_CMD;
    double MAX_THRUST_CMD;
    double MAX_RATE_CMD;
    PolicyActionType POLICY_ACTION_MIN;
    PolicyActionType POLICY_ACTION_MAX;
    Vector<3> OBS_POS_MIN;
    Vector<3> OBS_POS_MAX;
    Vector<3> OBS_VEL_MIN;
    Vector<3> OBS_VEL_MAX;
    Vector<3> OBS_ANG_VEL_MIN;
    Vector<3> OBS_ANG_VEL_MAX;
    Vector<3> OBS_ACC_MIN;
    Vector<3> OBS_ACC_MAX;
    double PERCEPTION_ALPHA2;
    double PERCEPTION_ALPHA3;
    double CAMERA_TILT_ANGLE_DEG;
    double OBS_REL_CORNER_MIN;
    double OBS_REL_CORNER_MAX;
    double OBS_REL_CORNER_WORLD_MIN;
    double OBS_REL_CORNER_WORLD_MAX;
    double GATE_PASS_MARGIN;
    double RESET_POS_PERTURB_BOUND;
    double RESET_VEL_GATE_START_PERTURB_BOUND;
    double RESET_ATT_RP_GATE_START_PERTURB_BOUND;
    double RESET_ATT_YAW_GATE_START_PERTURB_BOUND;
    double RESET_ATT_YAW_GLOBAL_START_PERTURB_BOUND;
    double RESET_OME_GATE_START_PERTURB_BOUND;
    double DISCOUNT_FACTOR_GAMMA;
    double INITIAL_RESET_PITCH_TILT_DEGREES;
    double INITIAL_RESET_SPEED;
    double GATE_PASS_REWARD_THRESHOLD;
    Vector<3> START_POS;
    int MAX_LAPS;
    int DRIFT_CORRECTION_INTERVAL;
    int MAX_STORED_STATES_PER_GATE;

    // --- NEW: Constructor to load from file ---
    RewardParams(const std::string& filepath) {
        YAML::Node config = YAML::LoadFile(filepath);

        PROGRESS_WEIGHT = config["PROGRESS_WEIGHT"].as<double>();
        ACTION_SMOOTHNESS_WEIGHT = config["ACTION_SMOOTHNESS_WEIGHT"].as<double>();
        BODY_RATE_PENALTY_WEIGHT = config["BODY_RATE_PENALTY_WEIGHT"].as<double>();
        GATE_PASS_REWARD = config["GATE_PASS_REWARD"].as<double>();
        TERMINATION_PENALTY = config["TERMINATION_PENALTY"].as<double>();
        MAX_SIM_TIME_RACING = config["MAX_SIM_TIME_RACING"].as<double>();
		MAX_SIM_TIME_STABILIZATION = config["MAX_SIM_TIME_STABILIZATION"].as<double>();
        GATE_MISS_BASE_PENALTY = config["GATE_MISS_BASE_PENALTY"].as<double>();
        GATE_MISS_MAX_PENALTY = config["GATE_MISS_MAX_PENALTY"].as<double>();
        GATE_MISS_DISTANCE_THRESHOLD = config["GATE_MISS_DISTANCE_THRESHOLD"].as<double>();
        MIN_THRUST_CMD = config["MIN_THRUST_CMD"].as<double>();
        MAX_THRUST_CMD = config["MAX_THRUST_CMD"].as<double>();
        MAX_RATE_CMD = config["MAX_RATE_CMD"].as<double>();
        PERCEPTION_ALPHA2 = config["PERCEPTION_ALPHA2"].as<double>();
        PERCEPTION_ALPHA3 = config["PERCEPTION_ALPHA3"].as<double>();
        CAMERA_TILT_ANGLE_DEG = config["CAMERA_TILT_ANGLE_DEG"].as<double>();
        OBS_REL_CORNER_MIN = config["OBS_REL_CORNER_MIN"].as<double>();
        OBS_REL_CORNER_MAX = config["OBS_REL_CORNER_MAX"].as<double>();
        OBS_REL_CORNER_WORLD_MIN = config["OBS_REL_CORNER_WORLD_MIN"].as<double>();
        OBS_REL_CORNER_WORLD_MAX = config["OBS_REL_CORNER_WORLD_MAX"].as<double>();
        GATE_PASS_MARGIN = config["GATE_PASS_MARGIN"].as<double>();
        RESET_POS_PERTURB_BOUND = config["RESET_POS_PERTURB_BOUND"].as<double>();
        RESET_VEL_GATE_START_PERTURB_BOUND = config["RESET_VEL_GATE_START_PERTURB_BOUND"].as<double>();
        RESET_ATT_RP_GATE_START_PERTURB_BOUND = config["RESET_ATT_RP_GATE_START_PERTURB_BOUND"].as<double>();
        RESET_ATT_YAW_GATE_START_PERTURB_BOUND = config["RESET_ATT_YAW_GATE_START_PERTURB_BOUND"].as<double>();
        RESET_ATT_YAW_GLOBAL_START_PERTURB_BOUND = config["RESET_ATT_YAW_GLOBAL_START_PERTURB_BOUND"].as<double>();
        RESET_OME_GATE_START_PERTURB_BOUND = config["RESET_OME_GATE_START_PERTURB_BOUND"].as<double>();
        DISCOUNT_FACTOR_GAMMA = config["DISCOUNT_FACTOR_GAMMA"].as<double>();
        INITIAL_RESET_PITCH_TILT_DEGREES = config["INITIAL_RESET_PITCH_TILT_DEGREES"].as<double>();
        INITIAL_RESET_SPEED = config["INITIAL_RESET_SPEED"].as<double>();
        GATE_PASS_REWARD_THRESHOLD = config["GATE_PASS_REWARD_THRESHOLD"].as<double>();
        MAX_LAPS = config["MAX_LAPS"].as<int>();
        MAX_STORED_STATES_PER_GATE = config["MAX_STORED_STATES_PER_GATE"].as<int>();
        DRIFT_CORRECTION_INTERVAL = config["DRIFT_CORRECTION_INTERVAL"].as<int>();

        // Load vector types using the helper
        POLICY_ACTION_MIN = loadVector<4>(config["POLICY_ACTION_MIN"]);
        POLICY_ACTION_MAX = loadVector<4>(config["POLICY_ACTION_MAX"]);
        OBS_POS_MIN = loadVector<3>(config["OBS_POS_MIN"]);
        OBS_POS_MAX = loadVector<3>(config["OBS_POS_MAX"]);
        OBS_VEL_MIN = loadVector<3>(config["OBS_VEL_MIN"]);
        OBS_VEL_MAX = loadVector<3>(config["OBS_VEL_MAX"]);
        OBS_ANG_VEL_MIN = loadVector<3>(config["OBS_ANG_VEL_MIN"]);
        OBS_ANG_VEL_MAX = loadVector<3>(config["OBS_ANG_VEL_MAX"]);
        OBS_ACC_MIN = loadVector<3>(config["OBS_ACC_MIN"]);
        OBS_ACC_MAX = loadVector<3>(config["OBS_ACC_MAX"]);
        START_POS = loadVector<3>(config["START_POS"]);
    }

    // Add a default constructor for cases where you might need it
    RewardParams() = default;
};


namespace agi_env {

    enum class GatePassResult {
        NotCrossed,
        PassedInside,
        PassedOutside
    };

    struct SerializableQuadState {
        std::vector<double> p_vec; // {x, y, z}
        std::vector<double> q_vec; // {x, y, z, w}
        std::vector<double> v_vec; // {x, y, z}
        std::vector<double> w_vec; // {x, y, z}
    };

    using GatePassStatesDataset = std::vector <std::vector<SerializableQuadState>>;

//---------------------------------------------------------------------
// AgiSim Batch Environment Class Declaration
//---------------------------------------------------------------------
    class AgiSimBatch {
    public:
        AgiSimBatch(int num_drones,
                    const std::string &sim_config_path,
                    const std::string &agi_param_dir,
                    const std::string &sim_base_dir,
                    const std::vector <std::vector<std::vector < double>>

        >& track_layout
        );

        ~AgiSimBatch() = default;

        std::vector <StepResultSimple> step(const std::vector <PolicyActionType> &actions_raw);

        std::vector <Observation43Type> reset(const std::vector<bool> &reset_flags);

        double getStates() const;

        void enableRollYaw();

        void progressRollYawInclusion(double);

        void logAndResetTimings();

        void setEval();

        void setNoiseIntensity(double vio_pos_drift_std, double vio_att_drift_std_deg,
                               double gate_reset_pos_std, double gate_reset_att_std_deg);

    private:
        int num_drones_;
        const SimulatorParams sim_params_;
        const double sim_dt_;
        std::vector <agi::QuadrotorSimulator> simulators_;

        std::vector <agi::QuadState, Eigen::aligned_allocator<agi::QuadState>> current_states_;
        std::vector <agi::QuadState, Eigen::aligned_allocator<agi::QuadState>> previous_states_;
        std::vector <PolicyActionType, Eigen::aligned_allocator<PolicyActionType>> prev_actions_raw_;
        std::vector <ActionScaledType, Eigen::aligned_allocator<ActionScaledType>> prev_actions_scaled_;

        std::vector <std::shared_ptr<ITask>> available_tasks_;
        std::vector <std::shared_ptr<ITask>> drone_current_tasks_;

        std::vector <std::vector<double>> per_drone_episode_rewards_;
        std::vector<int> per_drone_episode_step_count_;

        ObservationType assembleObservation(int drone_idx); // New orchestrator method
        agi::Vector<SHARED_OBS_SIZE> assembleSharedObservation(const agi::QuadState &current_state,
                                                               const PolicyActionType &prev_raw_action) const;

        ObservationType scaleAndClipObservation(const ObservationType &obs_raw, int drone_idx) const;

        void _processEndOfEpisodeForDrone(int drone_idx);

        ActionScaledType scaleAction(const PolicyActionType &raw_action) const;

        bool isDone(int i, const QuadState &state) const;

        QuadState perturbState(QuadState raw_state, int drone_idx);

        std::vector <std::mt19937> rngs_;
        const RewardParams reward_params_; // Tasks might need access

        bool evalMode_;

        std::vector <Vector<6>> vio_drift_offsets_;
        double vio_pos_drift_std_{0.0};
        double vio_att_drift_std_{0.0};
        double gate_reset_pos_std_{0.0};
        double gate_reset_att_std_{0.0};
    };

} // namespace agi_env


#endif // AGISIM_BATCH_HPP
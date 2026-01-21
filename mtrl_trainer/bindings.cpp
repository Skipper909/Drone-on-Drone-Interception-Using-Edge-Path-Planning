// File: bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Automatic conversion for std::vector
#include <pybind11/eigen.h>     // Automatic conversion for Eigen types (like agi::Vector)

// Include pybind11/functional.h if needed for callbacks, not required for guard
// #include <pybind11/functional.h>

#include "AgiSimBatch.hpp"      // Include your environment header

// Use the same namespace as your class
using namespace agi_env;
namespace py = pybind11;

// PYBIND11_MODULE defines the entry point for the Python module
// 'csim_env' is the name you will use to import the module in Python (e.g., import csim_env)
// 'm' is the py::module_ object representing the module
PYBIND11_MODULE(csim_env, m
) {
m.

doc() = "Pybind11 bindings for the AgiSimBatch C++ environment"; // Optional module docstring

    py::class_<agi_env::SerializableQuadState>(m, "SerializableQuadState")
            .def(py::init<>()) // Allows creating instances from Python if needed
            .def_readwrite("p_vec", &agi_env::SerializableQuadState::p_vec)
            .def_readwrite("q_vec", &agi_env::SerializableQuadState::q_vec)
            .def_readwrite("v_vec", &agi_env::SerializableQuadState::v_vec)
            .def_readwrite("w_vec", &agi_env::SerializableQuadState::w_vec)
            // Add other members of SerializableQuadState here if any (e.g., .def_readwrite("t", &agi_env::SerializableQuadState::t); )
            .def("__repr__", [](const agi_env::SerializableQuadState &sqs) {
                // Optional: for a nicer printout in Python
                std::string p_str = "p_vec=[";
                for(size_t i=0; i<sqs.p_vec.size(); ++i) p_str += std::to_string(sqs.p_vec[i]) + (i==sqs.p_vec.size()-1 ? "" : ", ");
                p_str += "]";
                // Similar for q_vec, v_vec, w_vec if desired
                return "<SerializableQuadState: " + p_str + ">";
            });




// Bind the StepResultSimple struct so Python can receive it
py::class_<StepResultSimple>(m,
"StepResultSimple")
// Expose members if needed, often handled automatically by vector bindings below
// Can make members readable/writeable from Python if desired:
.def_readonly("observation", &StepResultSimple::observation)

.def_readonly("reward", &StepResultSimple::reward) // <<< Expose reward
.def_readonly("done", &StepResultSimple::done)
.def_readonly("time", &StepResultSimple::time) // Readonly is often safer for results
.def_readonly("success", &StepResultSimple::success)
.def_readonly("task_id", &StepResultSimple::task_id)
.def("__repr__", // Optional: nice string representation
[](
const StepResultSimple &r
) {
// Improved repr (avoiding large observation print)
std::string obs_repr = "<ObservationType len=" + std::to_string(r.observation.size()) + ">";
return "<StepResultSimple obs=" + obs_repr + " done=" +
std::to_string(r
.done) + " time=" +
std::to_string(r
.time) +">";
});


py::class_<AgiSimBatch>(m, "AgiSimBatch")
.def(py::init<int,
        const std::string&,
        const std::string&,
        const std::string&,
        const std::vector<std::vector<std::vector<double>>>&>(),
py::arg("num_drones"),
py::arg("sim_config_path"),
py::arg("agi_param_dir"),
py::arg("sim_base_dir"),
py::arg("track_layout_world")
)

.def("setEval", &AgiSimBatch::setEval)
.def("getStates", &AgiSimBatch::getStates)
// Ensure step binding returns StepResultSimple which now includes reward
.def("step", &AgiSimBatch::step,
py::arg("actions_raw"), // Argument is now raw policy actions
py::call_guard<py::gil_scoped_release>())


.def("setNoiseIntensity", &AgiSimBatch::setNoiseIntensity,
             "Sets the parameters for the VIO drift and gate reset simulation.",
             py::arg("vio_pos_drift_std"),
             py::arg("vio_att_drift_std_deg"),
             py::arg("gate_reset_pos_std"),
             py::arg("gate_reset_att_std_deg"))

/*
    .def("get_successful_gate_pass_states_data", &agi_env::AgiSimBatch::getSuccessfulGatePassStatesData,
                 "Retrieves the stored successful gate pass states and scores as Python lists/dicts.")

    .def("set_successful_gate_pass_states_data", &agi_env::AgiSimBatch::setSuccessfulGatePassStatesData,
         py::arg("data_to_load"),
         "Sets the successful gate pass states from Python lists/dicts data.")
*/



// --- Update reset signature ---
// It now returns std::vector<Observation31Type>
.def("reset", &AgiSimBatch::reset,
py::arg("reset_flags"),
py::call_guard<py::gil_scoped_release>());
}

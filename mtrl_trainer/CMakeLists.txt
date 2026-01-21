# CMakeLists.txt for Python module csim_env

cmake_minimum_required(VERSION 3.16)
project(agisim_batch_pybind_module LANGUAGES CXX) # Renamed project slightly

set(CMAKE_CXX_STANDARD 17) # Match agilib's standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Define Paths ---
# (Copied from your working executable CMakeLists.txt)
set(AGILIB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../agilib/install)
set(AGILIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../agilib)
set(AGILIB_ACADOS_EXTERN_DIR ${AGILIB_SRC_DIR}/externals/acados-src) # Assumed location - VERIFY THIS!
set(AGILIB_ACADOS_EXTERN_LIB_DIR ${AGILIB_ACADOS_EXTERN_DIR}/lib)
set(AGILIB_ACADOS_EXTERN_INC_DIR ${AGILIB_ACADOS_EXTERN_DIR}/include)

# --- Status Messages & Path Verification ---
# (Copied from your working executable CMakeLists.txt)
message(STATUS "Using agilib includes from: ${AGILIB_INSTALL_DIR}/include")
message(STATUS "Using agilib libraries from: ${AGILIB_INSTALL_DIR}/lib")
message(STATUS "Expecting Acados includes in: ${AGILIB_ACADOS_EXTERN_INC_DIR}")
message(STATUS "Expecting Acados libraries in: ${AGILIB_ACADOS_EXTERN_LIB_DIR}")
# (Verification 'if(NOT EXISTS ...)' blocks kept exactly as before)
if(NOT EXISTS ${AGILIB_INSTALL_DIR}/include OR NOT EXISTS ${AGILIB_INSTALL_DIR}/lib)
    message(FATAL_ERROR "agilib install directory (${AGILIB_INSTALL_DIR}) does not contain include/ and lib/ subdirectories...")
endif()
if(NOT EXISTS ${AGILIB_INSTALL_DIR}/lib/libagilib.a OR NOT EXISTS ${AGILIB_INSTALL_DIR}/lib/libacados_mpc.a)
    message(WARNING "libagilib.a or libacados_mpc.a not found in ${AGILIB_INSTALL_DIR}/lib...")
endif()
if(NOT EXISTS ${AGILIB_ACADOS_EXTERN_LIB_DIR} OR NOT EXISTS ${AGILIB_ACADOS_EXTERN_INC_DIR})
    message(WARNING "Acados external include/lib directory not found at expected location...")
endif()

# --- Find Dependencies ---

# 0. pybind11 (NEW)
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11")

# 1. OpenMP (Manual Setup)
# (Copied from your working executable CMakeLists.txt)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    message(STATUS "Manually setting OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP flags not determined for compiler ID: ${CMAKE_CXX_COMPILER_ID}.")
    set(OpenMP_CXX_FLAGS "")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(RELEASE_CXX_FLAGS
        "-O3 -mtune=native -ffast-math -flto -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " ${RELEASE_CXX_FLAGS}")


add_compile_options("$<$<CONFIG:Release>:${OpenMP_CXX_FLAGS}>")

# 2. yaml-cpp
# (Copied from your working executable CMakeLists.txt)
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp")

# 3. Eigen3
# (Copied from your working executable CMakeLists.txt)
find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3")

# 4. Acados Libraries (Manual find)
# (Copied from your working executable CMakeLists.txt)
# Make sure NAMES and PATHS are correct!
find_library(ACADOS_LIB NAMES acados libacados PATHS ${AGILIB_ACADOS_EXTERN_LIB_DIR} NO_DEFAULT_PATH)
find_library(HPIPM_LIB NAMES hpipm libhpipm PATHS ${AGILIB_ACADOS_EXTERN_LIB_DIR} NO_DEFAULT_PATH)
find_library(BLASFEO_LIB NAMES blasfeo libblasfeo PATHS ${AGILIB_ACADOS_EXTERN_LIB_DIR} NO_DEFAULT_PATH)
# Add find_library calls for other required Acados libs...
# (Verification message kept as before)
if (NOT ACADOS_LIB OR NOT HPIPM_LIB OR NOT BLASFEO_LIB) # Add others if needed
    message(WARNING "Could not find all expected Acados core library files...")
endif()

# 5. Threads
# (Copied from your working executable CMakeLists.txt)
find_package(Threads REQUIRED)
message(STATUS "Found Threads")

# 6. Math library
# (Copied from your working executable CMakeLists.txt)
find_library(MATH_LIB NAMES m)


# --- Build Python Module Target --- ## CHANGED ##
# Use pybind11_add_module instead of add_executable
# 'csim_env' becomes the Python import name (import csim_env)
# List ONLY the sources needed for the module (NO main function)
pybind11_add_module(csim_env # Target name is also the Python module name
        bindings.cpp      # Your pybind11 binding code
        AgiSimBatch2.cpp   # The class implementation
        TaskRacing.cpp
        TaskStabilization.cpp
        TaskInterception.cpp
        TaskInterceptionMoving.cpp
        TaskTerminalPursuit.cpp
)
# Set C++ standard for the module target
set_target_properties(csim_env PROPERTIES CXX_STANDARD 17)


# --- Include Directories --- ## CHANGED ## Target name updated
# Apply includes to the new 'csim_env' target
target_include_directories(csim_env PRIVATE
        # Add pybind11 includes
        ${pybind11_INCLUDE_DIRS}

        # Keep previous include directories
        ${AGILIB_INSTALL_DIR}/include      # Agilib installed headers
        ${Eigen3_INCLUDE_DIRS}             # Eigen headers
        ${YAML_CPP_INCLUDE_DIRS}         # yaml-cpp headers
        ${AGILIB_ACADOS_EXTERN_INC_DIR}    # Main Acados includes from externals
        ${AGILIB_ACADOS_EXTERN_INC_DIR}/blasfeo/include # If needed
        ${AGILIB_ACADOS_EXTERN_INC_DIR}/hpipm/include   # If needed
)

set(TSAN_FLAGS
        -fsanitize=thread
        -fno-omit-frame-pointer
        -g)

# --- Link Dependencies --- ## CHANGED ## Target name updated
# Apply linking to the new 'csim_env' target
target_link_libraries(csim_env PRIVATE
        # --- Link agilib static libraries by full path ---
        ${AGILIB_INSTALL_DIR}/lib/libagilib.a
        ${AGILIB_INSTALL_DIR}/lib/libacados_mpc.a

        # --- Link agilib's other dependencies manually ---
        yaml-cpp      # Use target
        Eigen3::Eigen       # Use target
        Threads::Threads
        ${MATH_LIB}         # Link libm

        # Link Acados libs found by find_library
        ${ACADOS_LIB}
        ${HPIPM_LIB}
        ${BLASFEO_LIB}
        # Add others found...

        # --- Add OpenMP flag/library for linking ---
        ${OpenMP_CXX_FLAGS}
)



# --- Compiler Flags --- ## CHANGED ## Target name updated
# Apply flags to the new 'csim_env' target
target_compile_options(csim_env PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-g>
        ${OpenMP_CXX_FLAGS}
)

set_property(TARGET csim_env PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

message(STATUS "Configuration successful for Python module 'csim_env'.")
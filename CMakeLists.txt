cmake_minimum_required(VERSION 3.14) # Keep 3.8 if needed, but 3.14+ is better
project(agilib VERSION 0.1.0)

message(STATUS "==================   !Agilicious! (Standalone Build v2)   ===================")

################################################################################
# Options
################################################################################

option(BUILD_TESTS "Build the tests" ON)
option(BUILD_BENCH "Build the benchmark" OFF)
option(ENABLE_FAST "Build with optimizations for speed" ON)
option(UNSAFE_MATH "Build with -funsafe-math-optimizations..." ON)
option(ENABLE_PROFILING "Build for profiling" OFF)
option(ENABLE_PARALLEL "Build using OpenMP parallelization" OFF) # Default OFF is safer unless needed internally
option(EIGEN_FROM_SYSTEM "Use the system-provided Eigen" ON)
option(DEBUG_LOGGING "Enable detailed logging" OFF)
set(EIGEN_ALTERNATIVE "" CACHE STRING "Path to alternative Eigen, autodownload if blank")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Added: For standard install paths ---
include(GNUInstallDirs)

################################################################################
# Finding Dependencies
################################################################################

# find_package(ament_cmake REQUIRED) # REMOVED

# --- Eigen ---
if(EIGEN_FROM_SYSTEM)
    find_package(Eigen3 3.3.4 QUIET) # Check specific version first
    if (Eigen3_FOUND)
        message(STATUS "Using system provided Eigen ${Eigen3_VERSION}")
    else()
        find_package(Eigen3 REQUIRED) # Find any version if specific one not found
        message(STATUS "Using system provided Eigen ${Eigen3_VERSION}")
    endif()
    # Simplified: Assuming find_package works or errors out. Removed download logic.
    if(NOT Eigen3_FOUND)
        message(FATAL_ERROR "System Eigen3 required but not found.")
    endif()
elseif(EIGEN_ALTERNATIVE STREQUAL "")
    message(FATAL_ERROR "EIGEN_FROM_SYSTEM is OFF but EIGEN_ALTERNATIVE is blank. Provide path or enable system Eigen.")
    # Restore include(cmake/eigen.cmake) for download logic if needed
else()
    set(EIGEN3_INCLUDE_DIRS ${EIGEN_ALTERNATIVE})
    message(STATUS "Using alternative Eigen from: ${EIGEN3_INCLUDE_DIRS}")
    # Manually mark Eigen3 as found and create an INTERFACE target for consistency
    set(Eigen3_FOUND TRUE)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIRS})
    endif()
endif()

# --- ccache (Optional) ---
# (Kept as is)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
    message(STATUS "Build time could be improved with ccache! (sudo apt install ccache)")
endif()

# --- Acados ---
# Keep existing logic. Assumes cmake/acados.cmake defines the 'acados_agi' target
# and potentially variables like ACADOS_INCLUDE_DIRS, ACADOS_LIBRARIES.
include(cmake/acados.cmake)
if(NOT TARGET acados_agi)
    message(WARNING "cmake/acados.cmake did not define the 'acados_agi' target. Linking might fail.")
endif()
set(ACADOS_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/acados-src/lib") # Keep for reference

# --- OpenMP (Find if enabled) ---
set(AGILIB_ENABLE_OPENMP OFF)
if(ENABLE_PARALLEL)
    find_package(OpenMP)
    # Check CXX support explicitly via the target existence
    if(OpenMP_FOUND AND TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP CXX support, enabling parallelization.")
        set(AGILIB_ENABLE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP requested but CXX support not fully found/determined. Parallelization disabled.")
        set(ENABLE_PARALLEL OFF) # Disable if support incomplete
    endif()
endif()

if(ENABLE_PARALLEL) # <--- Outer IF starts
    find_package(OpenMP)
    # Check CXX support explicitly via the target existence
    if(OpenMP_FOUND AND TARGET OpenMP::OpenMP_CXX) # <--- Inner IF starts - THIS IS LIKELY LINE 91
        message(STATUS "Found OpenMP CXX support, enabling parallelization.") # <--- ERROR REPORTED HERE
        set(AGILIB_ENABLE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP requested but CXX support not fully found/determined. Parallelization disabled.")
        set(ENABLE_PARALLEL OFF) # Disable if support incomplete
    endif() # <--- Inner IF ends
endif() # <--- Outer IF ends

################################################################################
# Setup Compilation
################################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1048576)

if(NOT AGILIB_ENABLE_OPENMP) # Check flag set earlier
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# ... (Profiling logic kept as is) ...
# ... (Architecture flags kept as is) ...
# ... (Optimization flags kept as is) ...
# ... (Unsafe math logic kept as is) ...
# ... (Debug logging logic kept as is) ...
# ... (Build type default kept as is) ...
# ... (Flag combination kept as is) ...

################################################################################
# Specify Build Resources
################################################################################

# --- Main Library Sources ---
# Exclude mpc directory first, then glob remaining src
file(GLOB_RECURSE MPC_SOURCES_TO_EXCLUDE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/controller/mpc/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/controller/mpc/acados/*.c"
)
file(GLOB_RECURSE ALL_SRC_SOURCES # Get all potentially relevant source files
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" # If any C files needed outside acados
)
# Remove the excluded MPC sources from the main list
list(REMOVE_ITEM ALL_SRC_SOURCES ${MPC_SOURCES_TO_EXCLUDE})
set(SOURCES ${ALL_SRC_SOURCES}) # Final list for agilib target


# --- ACADOS Sources ---
# Use the same GLOB patterns used for exclusion
# !! IMPORTANT: Verify these paths match your actual source layout !!
message(STATUS "Searching for ACADOS sources relative to ${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB_RECURSE ACADOS_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/controller/mpc/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/controller/mpc/acados/*.c"
)
# Print found files for debugging
message(STATUS "ACADOS_SOURCES found: ${ACADOS_SOURCES}")


# --- Build acados_mpc Target ---
# Define target variable for later conditional use
set(ACADOS_MPC_TARGET "")
if(ACADOS_SOURCES) # Check if GLOB found any files
    message(STATUS "Building acados_mpc target.")
    set(ACADOS_MPC_TARGET acados_mpc) # Set target name variable
    add_library(${ACADOS_MPC_TARGET} STATIC ${ACADOS_SOURCES}) # Build as STATIC

    # Link acados_agi (must be defined by cmake/acados.cmake)
    target_link_libraries(${ACADOS_MPC_TARGET} PRIVATE acados_agi)

    # Compile options
    target_compile_options(${ACADOS_MPC_TARGET} PRIVATE -Wno-error -Wno-register)

    # Include directories (Using generator expressions for build vs install)
    target_include_directories(${ACADOS_MPC_TARGET} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # agilib headers
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>      # Installed agilib headers
            $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>             # Eigen
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/agilib/controller/mpc/acados/> # Specific path
            $<BUILD_INTERFACE:${ACADOS_INCLUDE_DIRS}>             # Core Acados includes (if variable set)
    )

    # Apply OpenMP flags if enabled
    if(AGILIB_ENABLE_OPENMP)
        target_compile_options(${ACADOS_MPC_TARGET} PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>) # Get flags from target
        target_link_libraries(${ACADOS_MPC_TARGET} PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # FATAL_ERROR because original CMake links it publicly from agilib
    message(FATAL_ERROR "Could not find any ACADOS sources matching patterns in src/controller/mpc/. "
            "Cannot build required 'acados_mpc' target. Check paths in CMakeLists.txt.")
endif()

# ... (TEST_SOURCES, BENCH_SOURCES globbing kept as is) ...

################################################################################
# Build the main agilib Library
################################################################################

add_library(agilib STATIC ${SOURCES}) # Use sources excluding MPC

# Include directories
target_include_directories(agilib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${ACADOS_INCLUDE_DIRS}> # Core Acados includes (if needed by agilib headers)
)

# Link dependencies
target_link_libraries(agilib PUBLIC
        acados_agi          # Defined by cmake/acados.cmake
        ${ACADOS_MPC_TARGET} # Use the variable holding the target name (acados_mpc)
        Eigen3::Eigen       # Use imported target
)

# Add OpenMP if enabled
if(AGILIB_ENABLE_OPENMP)
    target_compile_options(agilib PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>)
    target_link_libraries(agilib PRIVATE OpenMP::OpenMP_CXX)
endif()

# ... (target_compile_options for warnings kept as is) ...

################################################################################
# Build Tests (if enabled)
################################################################################

if(BUILD_TESTS AND TARGET GTest::gtest AND TARGET GTest::gtest_main)
    add_executable(tests ${TEST_SOURCES})
    # Tests link agilib privately, plus GTest libs
    target_link_libraries(tests PRIVATE agilib GTest::gtest GTest::gtest_main)
    if(AGILIB_ENABLE_OPENMP)
        target_compile_options(tests PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>)
        target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    include(GoogleTest)
    gtest_discover_tests(tests)
endif()

################################################################################
# Build Benchmarks (if enabled)
################################################################################

if(BUILD_BENCH AND TARGET benchmark::benchmark AND TARGET benchmark::benchmark_main)
    add_executable(benchmarks ${BENCH_SOURCES})
    target_link_libraries(benchmarks PRIVATE agilib benchmark::benchmark benchmark::benchmark_main)
    if(AGILIB_ENABLE_OPENMP)
        target_compile_options(benchmarks PRIVATE $<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>)
        target_link_libraries(benchmarks PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

################################################################################
# Installation Rules (Replaces ament_package)
################################################################################

message(STATUS "Configuring installation...")
# Install static libraries (.a) into the standard library directory
# The ACADOS_MPC_TARGET variable will be "acados_mpc" if built, empty otherwise.
# CMake handles empty target names gracefully in install(TARGETS...).
install(TARGETS agilib ${ACADOS_MPC_TARGET}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # e.g., <prefix>/lib
)

# Install public headers from the 'include' directory into '<prefix>/include'
# Preserves structure like include/agilib/...
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # e.g., <prefix>/include
        FILES_MATCHING PATTERN "*.hpp"
)

# --- Optional: Generate and Install CMake Package Config Files ---
# (Commented out section kept as is - enable if needed and create template file)
# include(CMakePackageConfigHelpers)
# include(GNUInstallDirs)
# ... etc ...

message(STATUS "agilib SOURCES: ${SOURCES}")
message(STATUS "===========   Done! Smells agilicious! (Standalone Build v2)   ==============")

# ament_package() # REMOVED
# Autonomous Drone Racing with Multi Task Reinforcement Learning (MTRL)

This project focuses on training an autonomous drone to navigate race tracks using reinforcement learning. 
Out of the box, it can utilize the AgiLib library for drone dynamics and a custom C++ simulator environment, with policy training performed using
TF-Agents in Python. To use Agilib, access to the Agilicious repository has to be obtained separately. 
With some customization, other simulators can instead be integrated into the training process.

## Prerequisites

* Ideally Ubuntu, this needs to be adapted significantly to run on other systems
* CMake (version 3.10 or higher recommended)
* Anaconda or Miniconda for Python environment management
* Git
* (Access to Agilicous source)

## Building the Project

AgiLib provides the core quadrotor dynamics and simulation components.

1. **Compiler and C++ prerequisites:**
   ```bash
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt-get update
   sudo apt-get install -y gcc-9 g++-9
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
   
   sudo apt-get install -y clang-10
   CC /usr/bin/clang-10
   CXX /usr/bin/clang++-10
   
   sudo apt-get install -y libyaml-cpp-dev libeigen3-dev libgoogle-glog-dev ccache tmux  net-tools iputils-ping nano wget usbutils htop gdb psmisc screen
   ```

2. **Create a main conda environment:**
   ```bash
   conda create -n tfa python=3.9 -y
   conda activate tfa
   pip install pybind11
   ```

3. **Download agilib from the Agilicous repository and copy it into the root folder at the same level as /mtrl_trainer**


4. **Copy the custom CMakeLists.txt from this project's root folder into /agilib**


5. **Build and install agilib:**
   ```bash
   cd agilib
   rm -rf build install
   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Debug
   cmake --build build -j $(nproc)
   cmake --install build
   ```

4.  **Build the MTRL trainer:**
   ```bash
  # make sure conda env tfa is active for this

   cd mtrl_trainer
   rm -rf build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
  cmake --build build -j $(nproc)

   ```

4.  **install rest of python dependencies (this list is incomplete):**
   ```bash
  export TF_USE_LEGACY_KERAS=1
# `--force-reinstall helps guarantee the right versions.
  pip install --user --force-reinstall tf-nightly
  pip install --user --force-reinstall tf-keras-nightly
  pip install --user --force-reinstall tfp-nightly
  pip install --user --force-reinstall dm-reverb-nightly
  pip install --user --upgrade tf-agents-nightly

  pip install scipy
  pip install numpy==1.26.4

   ```

4.  **Fixing paths:**
   ```bash
  # open train.py and fix the parameter paths at the top. 
  # Should be replaced with relative paths in the future.
   ```

4.  **Running the training:**
   ```bash
  cd mtrl_trainer
  conda activate tfa && export TF_USE_LEGACY_KERAS=1

  python3 train.py
  
  # fix any errors and add missing python dependencies
   ```
# AscendOps — flash_attn_minimal

English | [简体中文](./README.md)

## Project Overview
AscendOps is a lightweight, high-performance operator development project template. It integrates PyTorch, PyBind11, and the Ascend CANN toolchain, and provides a complete toolchain from operator kernel coding and compilation to Python wrapping.

## Key Features
🚀 Out-of-the-Box: A complete Ascend NPU operator development environment configuration is preset, so you can start development right after cloning.

🧩 Minimalist Design: The code structure is clear and intuitive, focusing on the core operator development workflow.

⚡ High Performance: Based on the AscendC programming model, it fully exploits the hardware capabilities of Ascend NPUs.

📦 One-Click Deployment: It integrates the setuptools build system and supports one-click compilation and installation.

🔌 PyTorch Integration: It seamlessly integrates PyTorch tensor operations and supports automatic differentiation and a unified GPU/NPU interface.

## Core Components
1. `csrc/flash_attn_minimal/torch_interface.cpp`: operator kernel implementation
2. `csrc/flash_attn_minimal/CMakeLists.txt`: operator CMake configuration

## Environment Requirements
*   AI processor: Ascend 950PR&950DT products
*   Python: 3.8+
*   PyTorch: 2.6.0+
*   PyTorchAdapter 7.1.0+
*   CMake: 3.18.4+
*   CANN Ascend Toolkit

## Environment Preparation

1. **Install the community edition CANN toolkit package**

    Download the CANN development kit package `Ascend-cann-toolkit_{version}_linux-{arch}.run` according to your actual environment.

    Then install the CANN development kit package. For details, see the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit).

    ```bash
    ./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install-path}
    ```

    - {version}: path of the CANN version package.
    - {arch}: system architecture.
    - {install-path}: specified installation path.

2. **Configure environment variables**

	Choose an appropriate command according to your actual scenario.

    ```bash
   # Default-path installation, using the root user as an example (for a non-root user, replace /usr/local with ${HOME})
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # Custom-path installation
   source ${install-path}/ascend-toolkit/set_env.sh
    ```

3. **Install the torch and torch_npu packages**

    Download the corresponding torch package according to your actual environment. Common wheel file names are as follows:

   - x86_64 Linux version: `torch-${torch_version}+cpu-${python_version}-linux_x86_64.whl`
   - ARM Linux (older versions usually do not have the `+cpu` suffix): `torch-${torch_version}-${python_version}-linux_aarch64.whl`

    Note: The version numbers and the `+cpu` suffix in the preceding examples are for illustration only. For different operating systems and architectures (especially ARM Linux), the wheel names may not contain the `+cpu` suffix. Use the actual package names available on the official PyTorch download page or through `pip`.

    Download link: [official website](https://download.pytorch.org/whl/torch)

    The installation command is as follows (replace `<torch_whl>` with the actual downloaded file name):

    ```sh
    pip3 install <torch_whl>
    ```

    Install the torch-npu package that matches the torch version according to your actual environment, for example: `torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`.

    You can download and install it directly using the pip command (replace `<torch_npu_whl>` with the actual downloaded file name):

    ```sh
    pip3 install <torch_npu_whl>
    ```

    - ${torch_version}: version number of the torch package.
    - ${python_version}: version number of python.
    - ${arch}: CPU architecture, such as aarch64 and x86_64.

## Installation Steps

1. Enter the directory and install dependencies
    ```sh
    pip3 install -r requirements.txt
    ```

2. Build the .whl package from source code
    ```sh
    python3 -m build --wheel -n
    ```

3. Install the built .whl package
    ```sh
    pip3 install dist/ascend_ops-*.whl
    ```

    To reinstall, run the following command to overwrite the installed version:
    ```sh
    pip3 install dist/ascend_ops-*.whl --force-reinstall --no-deps
    ```

4. (Optional) It is recommended to run the following command to clear the compilation cache before rebuilding
    ```sh
    python3 setup.py clean
    ```

## Development Mode Build

This command sets up a development environment that takes effect immediately. After execution, source code modifications take effect directly, omitting the process of building and installing a complete .whl package. This mode is suitable for scenarios where you need to modify and verify an operator multiple times:
  ```sh
  pip3 install --no-build-isolation -e .
  ```

## Usage Example
1. On-device execution

After installation, you can use NPU operators in the same way as ordinary PyTorch operations. Taking the flash_attn_minimal operator as an example, you can find and execute the script in the `ascend_ops/csrc/flash_attn_minimal/test` directory:

```bash
cd ./ascend_ops/csrc/flash_attn_minimal/test/
python3 test_fa.py
```

The execution succeeds if you see the following output:
```bash
compare CPU Result vs NPU Result: True
```

Operator performance profiling is also supported. You can collect profiling data quickly with the following command:
```bash
msprof --application="python3 test_fa.py"
```


## Further Reading
The following materials help you gain a deeper understanding of operator development and implementation, so that you can implement operators with better performance.

- [Simplified FA Operator Practice](./ascend_ops/csrc/flash_attn_minimal/README_en.md).

# Environment Deployment

Before performing [Operator Invocation](../invocation/quick_op_invocation.md) or [Operator Development](../develop/aicore_develop_guide.md) based on this project, complete the basic environment setup by referring to the following steps.

Note that the compilation and runtime scenarios mentioned in this document have the following meanings. Install as needed:

- Compilation scenario: For scenarios where only compilation without running this project is needed, only install prerequisite dependencies and the CANN toolkit package.
- Runtime scenario: For scenarios where this project is run (compile and run, or pure run), in addition to installing prerequisite dependencies and the CANN toolkit package, also install driver and firmware, and the CANN ops package.

## Prerequisites

Before using this project, ensure the following basic dependencies, NPU driver, and firmware have been installed.

1. **Install Dependencies**

   The dependencies used for source code compilation of this project are as follows. Note the version requirements.

   - python >= 3.7.0 (recommended version <= 3.10)
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz (optional, improves packaging speed after installation, recommended version >= 2.4)
   - dos2unix
   - gawk
   - make
   - googletest (only required when executing UT, recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

   The above dependency packages can be installed through the project root directory install\_deps.sh. The command is as follows. If encountering unsupported systems, refer to this file for self-adaptation.

   ```bash
   bash install_deps.sh
   ```

2. **Install Driver and Firmware (Runtime Dependency)**

   When running operators, driver and firmware must be installed. If only compiling operators, skip this operation.

   Click [Download Link](https://www.hiascend.com/hardware/firmware-drivers/community), and based on the actual product model and environment architecture, obtain the corresponding `Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run` and `Ascend-hdk-<chip_type>-npu-firmware_<version>.run` packages.

   For installation guidance, refer to [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum).

## Environment Preparation (Choose One of Three)

This project provides multiple ways to deploy CANN packages. Choose as needed.

- WebIDE and Docker environment: Provides simplified environment setup, **default installation of the latest commercially released CANN software package** (currently CANN 8.5.0).
- Manual CANN package installation: If you want to experience manual CANN package installation or the latest master branch capabilities, manual installation is recommended.

### Using WebIDE Environment

For users without an environment, you can directly use the WebIDE development platform, the "**Operator One-Stop Development Platform**". This platform provides an online directly runnable Ascend environment with essential software packages pre-installed, requiring no manual installation. For more information about the development platform, refer to [LINK](https://gitcode.com/opdevtools/plugin_release).

1. Enter the open-source project, click the "`Cloud Development`" button, and log in with a certified Huawei Cloud account. If not registered or certified, follow the page prompts for registration and certification.

 <img src="../figures/cloudIDE.PNG" alt="Cloud Platform"  width="750px" height="90px"> 

2. Follow the page prompts to create and start the cloud development environment. Click "`Connect > WebIDE`" to enter the operator one-stop development platform. The open-source project resources are in the `/mnt/workspace` directory by default.

   <!-- <img src="../figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px"> -->

### Using Docker Deployment

> **Note:**
>
> - Docker images are an efficient deployment method, currently only applicable to Atlas A2 series products, and currently only adapted for Ubuntu operating system.
> - The image file is relatively large, and downloading requires some time.

#### 1. Download Image

1. Log in to the host machine as root user. Ensure Docker engine (version 1.11.2 or above) is installed on the host.
2. Pull the image pre-integrated with the CANN software package and `ops-transformer` required dependencies from the [Ascend Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). The command is as follows, select based on actual architecture:

    ```bash
    # Sample: Pull ARM architecture CANN development image
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    # Sample: Pull X86 architecture CANN development image
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    ```

#### 2. Run Docker

After pulling the image, start the container with specific parameters so the container can access the host's Ascend devices.

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

| Parameter | Description | Notes |
| :--- | :--- | :--- |
| `--name cann_container` | Specify a name for the container for management. | Can be customized. |
| `--device /dev/davinci0` | Core: Map the host NPU device card to the container. Multiple NPU device cards can be specified. | Must be adjusted according to actual situation: `davinci0` corresponds to the 0th NPU card in the system. Execute `npu-smi info` on the host first, and modify this number based on the device number shown in the output (such as `NPU 0`, `NPU 1`).|
| `--device /dev/davinci_manager` | Map NPU device management interface. | - |
| `--device /dev/devmm_svm` | Map device memory management interface. | - |
| `--device /dev/hisi_hdc` | Map host-device communication interface. | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | Mount device container management interface (DCMI) related tools and libraries. | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mount `npu-smi` tool. | Enables running this command directly inside the container to query NPU status and performance information.|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Map host NPU driver libraries to the container. | - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mount driver version information file. | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mount CANN software installation information file. | - |
| `-it` | Combination of `-i` (interactive) and `-t` (allocate pseudo-terminal) parameters. | - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` | Specify the Docker image to run. | Ensure this image name and tag match exactly with the image pulled through `docker pull`. |
| `bash` | Command executed immediately after container startup. | - |

### Manual CANN Package Installation

#### 1. Download Software Package

Based on the following scenarios, obtain `Ascend-cann-toolkit_${cann_version}_linux-${arch}.run` and `Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run` as needed.

- Scenario 1: If you want to experience the **officially released CANN package** capabilities, visit the [CANN Official Download Center](https://www.hiascend.com/cann/download), select the corresponding version CANN software package (only supports CANN 8.5.0 and subsequent versions). For installation guidance, refer to [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html).

- Scenario 2: If you want to experience the **latest master branch capabilities**, click [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master) to obtain.

Note that the product model and environment architecture must correspond to the real environment. Additionally, the ops package is a runtime dependency. If only compiling operators, this package does not need to be installed.

#### 2. Install Software Package

1. **Install Community CANN Toolkit Package**

    ```bash
    # Ensure the installation package has executable permissions
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # Install command
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```

    - ${cann\_version}: Represents the CANN package version number.
    - ${arch}: Represents the CPU architecture, such as aarch64, x86_64.
    - ${install\_path}: Represents the specified installation path, default installation in the `/usr/local/Ascend` directory.

2. **Install Community CANN Ops Package (Runtime Dependency)**

    When running operators, this package must be installed. If only compiling operators, skip this operation.

    ```bash
    # Ensure the installation package has executable permissions
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # Install command
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```

    - ${soc\_name}: Represents the NPU model name, which is the content remaining after removing "ascend" from ${soc\_version}.
    - ${install\_path}: Represents the specified installation path, which must be the same path as the toolkit package installation. Default installation in the `/usr/local/Ascend` directory.

## Environment Verification

After installing the CANN package or entering the Docker container, verify that the environment and driver are functioning normally.

- **Check NPU Device**:

    ```bash
    # Run npu-smi. If device information is displayed normally, the driver is functioning properly
    npu-smi info
    ```

- **Check CANN Installation**:

    ```bash
    # View CANN Toolkit version information
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

## Environment Variable Configuration

Select the appropriate command to activate environment variables as needed.

```bash
# Default path installation, using root user as example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation
# source ${install_path}/cann/set_env.sh
```

## Source Code Download

Download the project source code through the following command, and install other dependencies. Replace ${tag\_version} with the branch tag name. The correspondence between this source repository and CANN versions can be found in the [release repository](https://gitcode.com/cann/release-management).

```bash
# Download project corresponding branch source code
git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer.git
# Install root directory requirements.txt dependencies
pip3 install -r requirements.txt
```

> [!NOTE] Note
> When using the HTTPS protocol on the gitcode platform, configure and use a personal access token instead of the login password for cloning, pushing, and other operations.  

If your compilation environment cannot access the network and cannot download code through `git` commands, download the source code in a networked environment and manually upload it to the target environment.

- In a networked environment, enter [This Project Homepage](https://gitcode.com/cann/ops-transformer), and complete source code download through the `Download ZIP` or `clone` button following the guidance.
- Connect to the offline environment and upload the source code to your specified directory. If a source code compressed package was downloaded, it also needs to be decompressed.

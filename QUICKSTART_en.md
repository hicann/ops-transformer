# Quick Start: Based on the ops-transformer Repository

## Usage Notice

This guide helps you quickly get started with CANN and the `ops-transformer` operator repository, providing simplified software installation and compilation/execution guidance **based on WebIDE or Docker environments**. Note that WebIDE or Docker environments provide the **latest commercially released CANN software package** by default, which is currently CANN 8.5.0.

> **Note**: If you are manually installing the CANN package or experiencing the latest capabilities of the master branch or other scenarios, you can skip this quick start and refer to the guides below for detailed tutorials. These tutorials provide rich operation methods for different scenarios to meet diverse business requirements.
>
> 1. [Environment Deployment Guide](./docs/en/context/quick_install.md): Environment setup guidance for different scenarios, including Docker installation, manual CANN software package installation, and so on.
> 2. [Compile and Execute Operator Guide](./docs/en/invocation/quick_op_invocation.md): Operator package compilation and verification guidance for different scenarios, such as offline compilation, in-depth understanding of compilation parameters and invocation methods.
> 3. [Operator Development Guide](./docs/en/develop/aicore_develop_guide.md): Guide for custom development of standard operators, learning to create operator projects from scratch and implement Tiling and Kernel.
> 4. [Debugging and Tuning Guide](./docs/en/debug/op_debug_prof.md): Systematic debugging techniques and performance optimization methods for different scenarios.

The basic flow of operator development and contribution is shown in the figure below. You are welcome and encouraged to contribute operators to the community to jointly enrich the project ecosystem.

![Operator Development Contribution Process](./docs/en/figures/Operator_development_contribution_process.png "Operator Development Contribution Process Diagram")

To help you quickly understand the full process of operator development, we will use the **AddExample** operator as the practice object. Its source files are located in `ops-transformer/examples/add_example`. The specific operation steps are as follows:

1. **[Environment Installation](#1-environment-installation-choose-one)**: Set up the operator development and runtime environment.
2. **[Compile and Deploy](#2-compile-and-deploy)**: Compile the custom operator package and deploy the installation to quickly invoke operators.
3. **[Operator Development](#3-operator-development)**: Modify the existing operator Kernel to experience the complete loop of development, compilation, and verification.
4. **[Operator Debugging](#4-operator-debugging)**: Learn operator printing and performance data collection methods.
5. **[Operator Verification](#5-operator-verification)**: Learn how to modify operator example samples to verify the functional correctness of operators under different inputs.

## 1. Environment Installation (Choose One)

### 1. No Environment Scenario: WebIDE Development

For users without an environment, use the WebIDE development platform directly, which is the "**One-stop Operator Development Platform**". This platform provides an online directly runnable Ascend environment with essential software packages already installed, requiring no manual installation. For more information about the development platform, refer to [LINK](https://gitcode.com/opdevtools/plugin_release).

1. Enter the ops-transformer open-source project, click the "`Cloud Development`" button, and log in with a certified Huawei Cloud account. If not registered or certified, follow the page prompts to register and certify.

   <img src="docs/en/figures/cloudIDE.PNG" alt="Cloud Platform" width="750px" height="90px"> 

2. Follow the page prompts to create and start a cloud development environment. Click "`Connect > WebIDE`" to enter the one-stop operator development platform. The open-source project resources are in the `/mnt/workspace` directory by default.

  <!--  <img src="docs/en/figures/webIDE.png" alt="Cloud Platform" width="1000px" height="150px"> -->

3. Check whether the environment is complete.

    In the cloud platform terminal window, execute the following commands to verify the environment and driver status.

    - **Check NPU Device**

        Execute the following command. If driver-related information is returned, the device has been successfully mounted.

        ```bash
        npu-smi info
        ```

    - **Check CANN Version**

        Execute the following command to view the CANN Toolkit version information.

        ```bash
        cat /home/developer/Ascend/ascend-toolkit/latest/opp/version.info
        ```

### 2. Existing Environment Scenario: Docker Installation

#### Prerequisites

* **Docker Environment**: Using Atlas A2 product (910B) as an example, the host machine has already installed the Docker engine (version 1.11.2 or higher).

* **Driver and Firmware**: The host machine has already installed the Ascend NPU [driver and firmware](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) Ascend HDK version 24.1.0 or higher. For installation guidance, refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/latest/softwareinst/instg/instg_0107.html?Mode=PmIns&InstallType=netyum&OS=openEuler).

    > **Note**: Use `npu-smi info` to check the corresponding driver and firmware version.

#### Download Image

Pull the image that has pre-integrated the CANN software package and the dependencies required by `ops-transformer`.

1. Log in to the host machine as the root user.
2. Execute the pull command (select based on your host machine architecture):

    * ARM architecture:

        ```bash
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```

    * X86 architecture:

        ```bash
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```

> **Note**: Under normal network speed, image download time is approximately 5-10 minutes.

#### Docker Run

Run Docker using the following command:

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

The following are parameter descriptions that users need to pay attention to:

| Parameter | Description | Precautions |
| :--- | :--- | :--- |
| `--name cann_container` | Specify a name for the container for easy management. | Can be customized. |
| `--device /dev/davinci0` | Core: Map the host machine NPU device card to the container. Multiple NPU device cards can be specified for mapping. | Must be adjusted according to actual conditions: `davinci0` corresponds to the 0th NPU card in the system. Execute the `npu-smi info` command on the host machine first, and modify this number based on the device number shown in the output (such as `NPU 0`, `NPU 1`).|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Map the host machine NPU driver library to the container. | - |

#### Check Environment

After entering the container, verify the environment and driver status.

- **Check NPU Device**

    Execute the following command. If driver-related information is returned, the device has been successfully mounted.

    ```bash
    npu-smi info
    ```

- **Check CANN Version**

    Execute the following command to view the CANN Toolkit version information.

    ```bash
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

You now have an "out-of-the-box" operator development environment. Next, verify the complete toolchain from source code to runnable operators in this environment.

## 2. Compile and Deploy

The purpose of this stage is to **quickly experience the project standard process** and verify whether the environment can successfully perform operator source code compilation, packaging, installation, and running.

### 1. Obtain Project Source Code

1. Obtain the project source code.

    Docker or WebIDE environments provide the latest commercially released version source code by default. If you need to obtain other version source code, download it through the following command. Replace `${tag_version}` with the target branch tag name. The correspondence between branch tags and CANN versions can be found in the [release repository](https://gitcode.com/cann/release-management).

    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer.git
    ```

    If "`fatal: destination path 'ops-transformer' already exists and is not an empty directory.`" appears, the project source code already exists. Use the `git pull` command to refresh the project code.

2. Enter the project root directory using the following command. Distinguish between Docker and WebIDE scenarios.
    - Docker scenario:

      ```bash
      cd ops-transformer
      ```

    - WebIDE scenario:

      ```bash
      cd /mnt/workspace/ops-transformer
      ```

### 2. Compile the AddExample Operator

Enter the project root directory and compile the specified operator. The general compilation command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

Using the AddExample operator as an example, the compilation command is as follows:

```bash
bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
```

If the following message appears, the compilation is successful.

```bash
Self-extractable archive "cann-ops-transformer-custom-linux.${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install the AddExample Operator Package

```bash
./build_out/cann-ops-transformer-*linux*.run
```

`AddExample` is installed in the `${ASCEND_HOME_PATH}/opp/vendors` path. `${ASCEND_HOME_PATH}` represents the CANN software installation directory.

### 4. Configure Environment Variables

Add the custom operator package path to the environment variables to ensure it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run Operator Sample

The general run command format: `bash build.sh --run_example <operator name> <run mode> <package mode>`.

Using AddExample as an example, it provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

Expected output: Print the addition calculation result of the `AddExample` operator, indicating that the operator has been successfully deployed and correctly executed.

```text
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## 3. Operator Development

The purpose of this stage is to try **modifying the kernel function code** of the successfully running AddExample operator.

### 1. Modify Kernel Implementation

Open the core Kernel implementation file of the AddExample operator `ops-transformer/examples/add_example/op_kernel/add_example.h`, and try changing the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Compile and Verify

Repeat steps 2 through 5 in the [Compile and Deploy](#2-compile-and-deploy) section:

1. **Recompile**:
    Return to the project root directory first. The compilation command is as follows:

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
    ```

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-transformer-*linux*.run
    ```

3. **Reverify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

4. **Success Indicator**: The output result changes to multiplication results.

    ```text
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## 4. Operator Debugging

This stage uses AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and positioning.

### 1. Printing

If the operator encounters execution failure, precision anomalies, or other issues, add printing for problem analysis and positioning.

Modify the code in `examples/add_example/op_kernel/add_example.h`.

* **printf**

  This interface supports printing Scalar type data, such as integers, characters, Boolean, and so on. For detailed introduction, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0005.html) in "Operator Debugging API > printf".
  
  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // Print the current core calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of a specified Tensor, and also supports printing custom additional information, such as the current line number. For detailed introduction, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0005.html) in "Operator Debugging API > DumpTensor".
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Data Collection

After the operator function verification is correct, collect operator performance data through the `msprof` tool.

- **Generate Executable File**

    Invoke the AddExample operator example sample to generate the executable file (test_aclnn_add_example), which is located in the project `ops-transformer/build` directory.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

- **Collect Performance Data**

    Enter the AddExample operator executable file directory `ops-transformer/build/`, and execute the following command:

    ```bash
    msprof --application="./test_aclnn_add_example"
    ```

The collection results are in the project `ops-transformer/build/` directory. After the msprof command execution is completed, it will automatically parse and export the performance data result files. For detailed content, refer to [msprof](https://www.hiascend.com/document/detail/en/mindstudio/latest/TITools/msProf/docs/en/getting_started/quick_start.md).

## 5. Operator Verification

This stage modifies the input data in the AddExample operator example sample to verify the functional correctness of the operator in multiple scenarios.

### 1. Modify Test Input

Find and edit the `AddExample` `ops-transformer/examples/add_example/examples/test_aclnn_add_example.cpp`, and modify the input tensor shape and values.

**Modify Input/Output Data**: Modify the shape information of inputs and outputs, as well as the initialization data, to construct the corresponding input and output tensors.

```c++
int main() {
    // ... Initialization code ...
    
    // === 1. Modify selfX input ===
    // Before modification: shape = {32, 4, 4, 4}, all values are 1
    // After modification: Change input shape to {8, 8, 8, 8}, and fill with different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 * 8
    // Use a loop to fill more distinguishable data, such as an incremental sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill with cyclic values 0-9
    }
    // === 2. Similarly modify selfY and selfZ inputs following the selfX pattern ===
    
    // ... Subsequent execution code ...
}
```

### 2. Recompile and Verify

1. Since only the example test code was modified, there is no need to recompile the operator package.

2. Re-execute the verification command:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

3. Observe whether the operator output results meet expectations.

## 6. Development Contribution

After experiencing the above operations, you have basically completed an operator development. You can contribute the operator to the `experimental` directory of this project. For the contribution process, refer to the [Contribution Guide](CONTRIBUTING_en.md). Any questions during the process can be consulted through Issues.

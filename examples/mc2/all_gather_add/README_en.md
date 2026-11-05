# AllGatherAdd

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |

## Function Description

- Operator function: fuses AllGather communication with Add computation.
- Computation formula:

    $$
    gatherOut=AllGather(a0, a1)
    $$
    $$
    c[i]=gatherOut[i] + b[i]
    $$

- Operator semantics example: (two cards participate in the computation, and the input prefixes are rank0 and rank1)

  - Inputs:
    rank0_a = [1,2,3];
    rank0_b = [4,5,6,7,8,9];
    rank1_a = [10,11,12];
    rank1_b = [13,14,15,16,17,18];

  - Outputs:
    rank0_c = AllGatherAdd(rank0_a, rank1_a, rank0_b)
                = AllGather(rank0_a,rank1_a) + rank0_b
                = [1,2,3,10,11,12] + [4,5,6,7,8,9]
                = [5,7,9,17,19,21]

    rank1_c = AllGatherAdd(rank0_a, rank1_a, rank1_b)
                = AllGather(rank0_a,rank1_a) + rank1_b
                = [1,2,3,10,11,12] + [13,14,15,16,17,18]
                = [14,16,18,26,28,30]

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 380px">
 <col style="width: 120px">
 <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>Input</td>
      <td>Operator input a. Input a of all cards in the communicator first participates in the AllGather computation to generate the intermediate result gatherOut.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Operator input b. Input b of each card in the communicator performs the Add computation with the gatherOut result to generate output c.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Output</td>
      <td>Final output of the operator computation.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gather_out</td>
      <td>Output</td>
      <td>Intermediate output of the AllGather communication part of the operator.</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Attribute</td>
      <td><li>String that identifies the communicator on the Host side, that is, the communicator name.</li><li>It is obtained through the Hccl interface "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>Attribute</td>
      <td>Number of communication data splits, that is, total data volume divided by the volume of a single communication.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rank_size</td>
      <td>Attribute</td>
      <td>Number of cards in the communicator.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

* Currently, this example operator only supports fixed shapes: a(240, 256), b(240 * 2, 256), and a fixed rank_size = 2.
* Empty tensors are not supported for all inputs, and the value range is [-5,5].
* Deterministic computation:
  - The allGatherAdd operator implements deterministic computation by default.

## Invocation Description

Before invoking this operator, ensure that the code repository has been downloaded locally and that the following basic dependencies, NPU drivers, and firmware have been installed.

The dependencies used for compiling the source code of this project are as follows. Refer to the **Prerequisites** and **Environment Preparation** chapters in the [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md) document to complete the download of environment dependencies and the preparation of the CANN package. In the **Environment Preparation** chapter, the **Install the latest CANN ops-math package** part can be skipped.

**Compilation and Execution**

This example operator is compiled and executed in the custom operator package mode.

1. **Compile the custom operator package**

    Enter the project root directory and run the following compilation command:

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # Command execution example: (compile the all_gather_add operator in the 910b environment
    # bash build.sh --pkg --soc=ascend910b --ops=all_gather_add
    ```

    - --soc: \$\{soc\_version\} indicates the NPU model. Use "ascend910b" (default) for Atlas A2 products, and "ascend910_93" for Atlas A3 products.
    - --vendor_name (optional): \$\{vendor\_name\} indicates the name of the custom operator package to build. The default name is custom.
    - --ops: enter the name of this example operator, all_gather_add.

    If the following information is displayed, the compilation succeeds.

    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```

    After the compilation succeeds, the run package is stored in the build_out directory in the project root directory.

2. **Install the custom operator package**

    ```bash
    # First, go to the directory where the custom operator package is generated
    cd build_out/
    # Then install the custom operator package
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    The installation path of the custom operator package is ```${ASCEND_HOME_PATH}/opp/vendors```. \$\{ASCEND\_HOME\_PATH\} has been configured through the environment variable and indicates the installation path of the CANN toolkit package, which is generally \$\{install\_path\}/latest/opp. Note that the custom operator package does not support uninstallation.

    If an error message similar to "environment variable ASCEND_CUSTOM_OPP_PATH=/xxx: is set and has multiple path in it(colon inside), which will cause the custom op installed incorrectly. Please use the	--install-path option to specify an installation path instead." is displayed during the installation of the custom operator package, the ASCEND_CUSTOM_OPP_PATH path has been set in advance in the environment. In this case, unset the environment variable and reinstall the custom operator package.

**Local Verification**

With the build.sh script in the project root directory, you can quickly invoke the operator and UT cases to verify that the project functions properly. For the build parameters, see [Build Parameter Description](../../../docs/zh/install/build.md).

- **Run the operator sample**

    - Currently, this example operator only supports invoking the operator based on the single-operator API execution mode, that is, the [two-phase aclnn interface](../../../docs/zh/context/two_phase_api_en.md) invocation. The aclnn interface of this example is defined in the [op_api](../all_gather_add/op_host/op_api) path.

      To run this sample operator, perform the following steps:

      1. Install the custom operator package in the default path by referring to the **Install the custom operator package** chapter.

      2. Write the operator execution code, and complete the operator invocation by calling the two-phase interfaces aclnnAllGatherAddGetWorkspaceSize and aclnnAllGatherAdd in the [aclnn_all_gather_add.h](../all_gather_add/op_host/op_api/aclnn_all_gather_add.h) file. Note that the operator execution code must be placed in the [examples](../all_gather_add/examples/) path of the operator, and the file name must follow the "test_aclnn_[operator name].cpp" format. The [test project running script](../../../build.sh) matches and runs the corresponding execution file according to the operator name.

      The sample operator execution code of this example: [test_aclnn_all_gather_add.cpp](examples/test_aclnn_all_gather_add.cpp). The code first randomly generates test data of the fixed shapes as inputs, concatenates and adds the input data on the CPU side to simulate the AllGather and Add operations and generate the golden data, then converts the generated test data into Tensors and copies them to the device side, and finally calls the two-phase interfaces in [the aclnn file of this example](../all_gather_add/op_host/op_api/aclnn_all_gather_add.h) in sequence to complete the single-operator API invocation.

      After the custom operator package is installed and the operator execution code is written, run the following command:

        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}]
        # Command execution example:
        # bash build.sh --run_example all_gather_add eager cust
        ```

        - \$\{op\}: operator to be executed, in the lowercase-underscore form, for example, all_gather_add.
        - \$\{mode\}: execution mode. Currently, eager (aclnn invocation) is supported.
        - \$\{pkg_mode\}: package mode. Currently, only cust, that is, the custom operator package, is supported.
        - \$\{vendor\_name\} (the name can be customized): consistent with the setting of the built custom operator package. The default name is custom.

        After the operator execution finishes, the test sample copies the operator execution results to the host side and performs an accuracy comparison with the golden data. The accuracy requirement of this example operator is one thousandth.
        The following is an example of the running result:

        ```
        device_0 aclnnAllGatherAdd execute successfully.
        device_1 aclnnAllGatherAdd execute successfully.
        device_0 aclnnAllGatherAdd golden compare successfully.
        device_1 aclnnAllGatherAdd golden compare successfully.
        ```

- **Precautions**
    - This example operator only accepts tensors of fixed shapes as parameters and only supports execution on 2 Ascend AI processors.

      In the operator execution code [test_aclnn_all_gather_add.cpp](examples/test_aclnn_all_gather_add.cpp), rankId is the logical ID of an Ascend AI processor. The rankId values in the code are set to 0 and 1 so that the operator executes on the Ascend AI processors with IDs 0 and 1. During actual execution, multiple processes may compete for processors 0 and 1, causing the following error:

        ```bash
        [ERROR] HcclCommInit failed. ret = 4
        ```

      In this case, set the [ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0028.html) environment variable to specify the Ascend AI processors that actually execute the operator:

        ```bash
        # View the running status of each Device
        npu-smi info
        # If processors 6 and 7 are idle, specify that only 6 and 7 are visible to the current process:
        export ASCEND_RT_VISIBLE_DEVICES=6,7
        ```

        Then, the current process can only see the two processors 6 and 7, and their logical IDs for the current process are 0 and 1.
        Re-run the operator.

# Introduction

CANN Simulator is a SoC-level chip simulation tool for operator development scenarios, used to analyze precision and performance data (such as instruction execution status) of AI tasks running on the AI simulator at various stages. This tool helps users perform deep performance tuning, enabling R&D personnel to obtain verification results and performance feedback almost consistent with real chips even when chip resources are unavailable or scarce.

# Main Functions

This tool maintains binary compatibility with on-board execution (the same kernel can execute on both the simulator and AI processor). The main uses are as follows:

* Precision Simulation: Output bit-level precision results, assisting users in completing operator precision verification.
* Performance Simulation: Output instruction pipeline diagrams, assisting users in locating operator performance bottleneck issues.

# Preparation Before Use

## Usage Constraints

* Recommended tool environment configuration: CPU 16 cores, memory 32GB or more. 
* All paths mentioned in this document need to ensure the running user has read or read-write permissions.
* For security and minimum privilege considerations, it is recommended to use regular user permissions to execute this tool, avoiding root or other high-privilege accounts.
* This tool depends on the CANN software package. Before use, install the CANN software package first. No driver or firmware installation is required. Execute the CANN set_env.sh environment variable file through the source command. For security, do not modify the environment variables involved in set_env.sh after executing the source command.
* Users should follow the principle of minimum privilege. For example, files input to the tool must not be writable by other users. In some more strictly security-required functional scenarios, ensure that input files are not writable by group users.
* This tool is a development tool and is not recommended for use in production environments.
* The tool's simulation function only supports single-card scenarios and cannot simulate multi-card environments. The code can only be set to card 0. Modifying the visible card number will cause simulation failure.
* The simulation environment only supports AI Core computation-type operators (does not support MC2 and HCCL type operators).
* The CANN Simulator tool is currently in the early experience version stage, only supporting the Ascend950PR chip. It is recommended that the simulator runtime environment be configured with 16-core CPU and 32GB or more memory.
* Currently, ARM environment simulation is not supported.

## Environment Preparation

CANN Simulator is integrated in the CANN toolkit package. Refer to [Environment Deployment](../context/quick_install.md) to complete software package installation.

# Quick Start

Below, [add_examples](../../../examples/add_example/) is used as an example to detail operator simulation.

## Operator Compilation

* Refer to [Operator Invocation](../invocation/quick_op_invocation.md) to complete add_example operator compilation and installation.

```bash
# Note: Enter the project root directory and execute the following compilation command. The command is for reference only; for details, see the operator invocation instructions.
bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example
# Install custom operator package
./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
```

* Refer to [aclnn Invocation](../invocation/op_invocation.md#aclnn-invocation) to complete test_aclnn_add_example.cpp compilation, generating the executable file test_aclnn_add_example.

## Execute Simulation Command

```bash
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
```

The simulation tool execution log files are in the examples/add_example/examples/build/bin/cannsim_* directory. The execution log file is:

```text
cannsim.log
```

From the simulation tool log file, you can see the print information in the sample:

```text
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
```

## View Performance Pipeline

The simulation performance pipeline files are in the project `examples/add_example/examples/build/bin/cannsim_*/report` directory. The pipeline-related file is:

```text
trace_core0.json
```

Enter "chrome://tracing" in the Chrome browser address bar, and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For specific parameter introduction, refer to the "Simulation Result Parsing" chapter.

# Simulation Execution Description

## Command Function

Execute applications in the simulation environment.

## Command Format

cannsim record [options] user_app --user_options

## Parameter Description

Table 1 Simulation Execution Parameter Description

|Parameter|Optional/Required|Description|
| --- | --- | --- |
|-s  or --soc_version  [options] parameter | Required | Specify the target chip version for simulation (such as Ascend950).|
|-o  or --output  [options] parameter | Optional| Path for generated files, can be configured as absolute path or relative path, and the tool execution user needs to have read-write permissions. If no path is specified, data is saved in the current directory by default.|
|-g or --gen-report [options] parameter | Optional | Enable automatic parsing after simulation completion and generate analysis report. Default is no automatic parsing.|
|user_app|Required|Operator executable file.|
|--user_options|Optional|Runtime parameters for the operator executable file.|

## Usage Sample

1. Complete operator development and compilation.
2. Execute simulation command. Refer to the following usage samples:

    ```bash
    # Method 1: Enable simulation and save output to ./output directory, /path/to/app is the operator program
    $ cannsim record /path/to/app -o ./output -s Ascend950

    # Method 2: Enable simulation and generate report for subsequent performance analysis
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. After command completion, a folder named "cannsim_{timestamp}_${user_app}" will be generated in the default path or specified "output" directory. The structure sample is as follows:

    ```text
    ├─cannsim_{timestamp}_${user_app}
    ├── cannsim.log
    ```

4. Users can obtain operator execution results and compare precision. Results are displayed in cannsim.log. The sample is as follows:

    The following output is only an example of Ascend C single operator direct invocation precision comparison results, which may vary slightly due to version differences. Refer to actual output.

    ```text
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. View operator instruction pipeline diagram, refer to simulation result parsing.

# Simulation Result Parsing Description

## Command Function

Generate visualized instruction pipeline diagrams.

## Command Format

cannsim report [options]

## Parameter Description

Table 1 Simulation Result Parsing Parameter Description

|Parameter | Optional/Required | Description|
| --- | --- | --- |
|-e  or --export  [options] parameter | Required | Original result file directory, must be specified as the result directory generated after simulation execution, specified to the cannsim_{timestamp}_${user_app} level, can be configured as absolute path or relative path, and the tool execution user has read-write permissions.|
|-o  or --output  [options] parameter | Optional | Parsing result output directory, can be configured as absolute path or relative path, and the execution user needs to have read-write permissions. If no path is specified, data is saved in the current directory by default. If the generated result file has the same name as an existing file, the original file will be overwritten.|
|-n or --core-id  [options] parameter | Optional | Specify the core ID for generating instruction pipeline. If not specified, core 0 pipeline is generated by default. The configuration format is as follows: To generate pipeline for all cores, configure as 'all'. To specify core ID range, such as '0-1'. To specify a single core ID, such as '5'.|

## Usage Sample

1. Refer to simulation execution to execute operator simulation, compare output samples, and ensure corresponding results execute correctly.
2. Execute simulation result parsing command. Refer to the following execution samples.

    ```bash
    # Generate performance analysis report in the current directory (default only analyzes core 0)
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} 

    # Generate performance analysis reports for core 0, core 1, core 11, core 12 in the specified directory
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} -o /path/to/report -n '0-1, 11-12'
    ```

3. After command execution, corresponding pipeline files will be generated in the output configured directory. The file format is json format. The output result sample is as follows:

    ```text
    trace_core0.json
    trace_core1.json
    ...
    ```

4. Simulation result viewing
    Enter "chrome://tracing" in the Chrome browser address bar, and drag the generated instruction pipeline diagram file (trace.json) to the blank area to open it. Use keyboard shortcuts (W: zoom in, S: zoom out, A: move left, D: move right) to view.
    ![Instruction Pipeline Diagram](../figures/Instruction_pipeline.png)

    Table 2 Key Field Description

    |Field Name|Field Meaning|
    | --- | --- |
    |VECTOR|Vector computation unit.|
    |SCALAR|Scalar computation unit.|
    |Cube|Matrix multiplication computation unit.|
    |MTE1|Data搬运pipeline, data搬运direction: L1 ->{L0A/L0B, UBUF}.|
    |MTE2|Data搬运pipeline, data搬运direction: {DDR/GM, L2} ->{L1, L0A/B, UBUF}.|
    |MTE3|Data搬运pipeline, data搬运direction: UBUF -> {DDR/GM, L2, L1}, L1->{DDR/L2}.|
    |FIXP|Data搬运pipeline, data搬运direction: FIXPIPE L0C -> OUT/L1.|
    |FLOWCTRL|Control flow instructions.|
    |ICACHELOAD|View ICache misses.|

# Query Help Information

## Command Function

Query tool help information.

## Command Format

Query tool help information:

```bash
cannsim --help
```

Query tool record subcommand help information:

```bash
cannsim record --help
```
  
Query tool report subcommand help information:

 ```bash
 cannsim report --help 
 ```

## Parameter Description

None

## Usage Sample

1. Log in to the Host-side server.
2. Execute the following command.

    ```bash
    cannsim --help
    ```

## Output Description

```text
usage: cannsim [-h] {record,report} ...

Command-line tool for performance simulation analysis on Ascend hardware.

positional arguments:
  {record,report}  Available commands
    record         Run user application in AscendOps simulation environment
    report         Generate performance analysis reports

options:
  -h, --help       show this help message and exit
```

# build Parameters

## Overview

**build.sh** is the build script of this project, which is stored in the root directory of the project by default. It is used to automatically compile, link, and configure the source code, and finally generate executable files, library files, or other target files that can be installed or directly run. Specifically, different parameters can be configured in the script to implement multiple functions, including building multiple target libraries (such as **libophost_math.so**), compiling operator packages, and executing unit tests.

## Usage

1. **Configuring environment variables**
   
   Configure environment variables by referring to [Environment Setup](../../zh/invocation/quick_op_invocation.md).
   
   ```bash
   # The following uses the default installation path and the root user as an example.
   source /usr/local/Ascend/set_env.sh
   ```

2. **Build command format**

   The following uses the command for building an operator package as an example. In the command, `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For details about all options, see the "Options" section below. Select appropriate options as required.

## Options

The **build.sh** script supports multiple functions. You can run the following command to view all function parameters.

```bash
bash build.sh --help
```

| Option            | Mandatory (Yes/No) | Description                                                                     |
|-----------------|--------|---------------------------------------------------------------------------|
| -j${n}          | No    | Specifies the number of compilation threads. ${n} indicates the number of threads and the default value is 8 (for example, -j8). If the number of threads exceeds the number of CPU cores, the number of compilation threads is automatically adjusted to the number of CPU cores.               |
| -v              | No    | Views the CMake compilation configuration.                                                           |
| -O${n}          | No    | Specifies the compilation optimization level, which can be O0, O1, O2, and O3 (for example, -O3). ${n} indicates the optimization level.                               |
| -u              | No    | Enables the unit test (UT) compilation mode and compiles all UT targets.                                                 |
| --help, -h       | No    | Prints the help information about the script.                                                              |
| --ops           | No    | Specifies the operator to be compiled, for example, add and add_lora. Use commas (,) to separate multiple operators. This option cannot be used together with --ophost, --opapi, and --opgraph.|
| --soc           | No    | Specifies the NPU model. Only one NPU model can be compiled each time.                                                  |
| --jit           | No    | Specifies whether to compile the binary file of the operator. If this option is configured, the binary file of the operator is not compiled.                                                       |
| --vendor_name   | No    | Specifies the name of the custom operator package. The default value is custom.                                                  |
| --build-type    | No    | Enables the debug mode. The value can be Release (default) or Debug. If the value is Debug, this option cannot be used together with --mssanitizer or --oom.        |
| --cov           | No    | Reserved. You can ignore it.                                                          |
| --noexec        | No    | Compiles only the UT binary files and does not automatically execute the compiled UT executable files.                                           |
| --opkernel      | No    | Compiles the binary kernel.                                                                 |
| --pkg           | No    | Generates an installation package. This option cannot be used together with -u (UT mode), --ophost, --opapi, and --opgraph.                        |
| --asan          | No    | Enables the AddressSanitizer (ASAN) memory check function on the host.                                          |
| --valgrind      | No    | Reserved. You can ignore it.                                                          |
| --make_clean    | No    | Executes basic cleanup (cleaning up compilation products) and exits the script.                                                |
| --ophost        | No    | Compiles the libophost_math.so library. This option cannot be used together with --pkg and --ops.                                 |
| --opapi         | No    | Compiles the libopapi_math.so library. This option cannot be used together with --pkg and --ops.                                  |
| --opgraph       | No    | Compiles the libopgraph_math.so library. This option cannot be used together with --pkg and --ops.                                |
| --ophost_test   | No    | Compiles UTs related to ophost. This option is equivalent to the combination of -u and --ophost.                                         |
| --opapi_test    | No    | Compiles UTs related to opapi. This option is equivalent to the combination of -u and --opapi.                                           |
| --opgraph_test  | No    | Reserved. You can ignore it.                                                          |
| --opkernel_test | No    | Compiles UTs related to opkernel. This option is equivalent to the combination of -u and --opkernel.                                     |
| --run_example   | No    | Compiles the sample of the specified operator and mode, and executes the compiled executable file.                                                |
| --genop         | No    | Creates the initial directory for custom operators of AI Core.                                                      |
| --genop_aicpu   | No    | Creates the initial directory for custom operators of AI CPU.                                                       |
| --experimental  | No    | Compiles the user operator in the experimental directory.                                                  |
| --mssanitizer   | No    | Enables the mssanitizer memory check function on the kernel. This option cannot be used together with --oom.                                                  |
| --oom           | No    | Enables the OOM memory check function on the kernel. This option cannot be used together with --mssanitizer.                                                  |

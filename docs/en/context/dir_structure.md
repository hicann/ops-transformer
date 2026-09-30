# Project Directory

> Some directories listed in this section are optional. The actual deliverables prevail. Especially, the deliverables vary in different scenarios for the **single-operator directory**. The details are as follows:
>
> - If the **op_host** directory is missing, one possible reason is that the **op_host** implementation of another operator is called. For details about the calling logic, see the source code implementation in the **op_api** or **op_graph** directory of the operator. Another possible cause is that the kernel is not implemented using Ascend C. If necessary, you are welcome to contribute to the operator by referring to [Contribution Guide](../../../CONTRIBUTING.md).
> - If the **op_kernel** directory is missing, one possible reason is that the **op_kernel** implementation of another operator is called. For details about the calling logic, see the source code implementation in the **op_api** or **op_graph** directory of the operator. Another possible cause is that the kernel is not implemented using Ascend C. If necessary, you are welcome to contribute to the operator by referring to [Contribution Guide](../../../CONTRIBUTING.md).
> - If the **op_api** directory is missing, the operator does not support aclnn call.
> - If the **op_graph** directory is missing, the operator cannot be called in graph mode.

The full project directory structure is as follows:

```text
├── cmake                                               # Project build directory
│   ├── aclnn_ops_math.h.in                             # aclnn summary header file template
│   └── ...
├── common                                              # Common header files and code of the project
│   ├── CMakeLists.txt
│   ├── inc                                             # Common header file directory
│   └── src                                             # Common code directory
├── experimental                                        # Directory for storing custom operators
│   ├── conversion                                      # (Optional) Directory for storing custom conversion operators
│   │   └── CMakeLists.txt
│   ├── math                                            # (Optional) Directory for storing custom math operators
│   │   └── CMakeLists.txt
│   └── random                                          # (Optional) Directory for storing custom random operators
│       └── CMakeLists.txt
├── ${op_class}                                         # Operator class, such as conversion, math, and random operators
│   ├── ${op_name}                                         # Operator project directory. ${op_name} indicates the operator name consisting of lowercase letters and underscores (_).
│   │   ├── CMakeLists.txt                              # Operator CMakeLists entry
│   │   ├── README.md                                   # Operator introduction
│   │   ├── docs                                        # Operator documentation directory
│   │   │   └── aclnn${OpName}.md                       # Operator aclnn API introduction. ${OpName} indicates the operator name in UpperCamelCase style.
│   │   ├── examples                                    # Operator call example directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Example of calling the operator using aclnn
│   │   │   └── test_geir_${op_name}.cpp                # Example of calling the operator using GEIR
│   │   ├── op_graph                                    # Graph fusion implementation
│   │   │   ├── CMakeLists.txt                          # CMakeLists file on the op_graph side
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file, which implements operator data type inference.
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition, which is used to identify operators in the graph optimization and fusion phases
│   │   │   └── fusion_pass                             # Operator fusion pattern directory
│   │   ├── op_host                                     # Implementation on the host
│   │   │   ├── CMakeLists.txt                          # CMakeLists file on the host
│   │   │   ├── config                                  # (Optional) Binary configuration file. If this file is not configured, it will be automatically generated.
│   │   │   │   ├── ${soc_version}                      # Binary information of the operator configured on the NPU. ${soc_version} indicates the NPU model.
│   │   │   │   │   ├── ${op_name}_binary.json          # Binary configuration file of the operator
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # SimplifiedKey configuration of the operator
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library, which defines basic operator information, such as the name, input, output, and data type.
│   │   │   ├── ${op_name}_infershape.cpp               # (Optional) InferShape implementation, which infers the output shape based on the operator shape. If this file is not configured, the output shape is the same as the input shape.
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # (Optional) Tiling optimization in some sub-scenarios. ${sub_case} indicates the sub-scenario. For example, ${op_name}_tiling_arch35 is used for optimization in the arch35 architecture. If this file does not exist, the operator does not have a specific tiling policy for the corresponding sub-scenario.
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # (Optional) Header file for tiling implementation in the ${sub_case} sub-scenario.
│   │   │   ├── ${op_name}_tiling.cpp                   # (Optional) Tiling implementation file. If this file does not exist, tiling is not implemented in the corresponding scenario. (Tensors are divided into multiple blocks, and parallel computation is performed based on data types.)
│   │   │   ├── ${op_name}_tiling.h                     # (Optional) Header file for tiling implementation
│   │   │   └── op_api                                  # (Optional) Directory for storing the operator aclnn implementation file. If this directory is not configured, it will be automatically generated.
│   │   │       ├── aclnn_${op_name}.cpp                # Operator aclnn API implementation file
│   │   │       ├── aclnn_${op_name}.h                  # Operator aclnn API implementation header file
│   │   │       ├── ${op_name}.cpp                      # Operator l0 API implementation file
│   │   │       ├── ${op_name}.h                        # Operator l0 API implementation header file
│   │   │       └── CMakeLists.txt
│   │   │── op_kernel                                   # Kernel implementation of the AI Core operator on the device
│   │   │   ├── ${sub_case}                             # (Optional) Directory used in the ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file. ${model} indicates the user-defined file name extension, which is usually the tiling template name.
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # (Optional) TilingKey file, which defines the key of the tiling policy and identifies different division modes. If this file is not configured, the operator does not have a corresponding tiling policy.
│   │   │   ├── ${op_name}_tiling_data.h                # (Optional) TilingData file, which stores the configuration related to the tiling policy, such as the block size and parallelism degree. If this file is not configured, the operator does not have a corresponding tiling policy.
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file, which contains the main function and scheduling logic
│   │   │   └── ${op_name}.h                            # Kernel implementation file, which defines the kernel header file and contains the function declaration, structure definition, and logic implementation
│   │   │── op_kernel_aicpu                             # (Optional) Kernel implementation of the AI CPU operator on the device
│   │   │   ├── ${op_name}_aicpu.cpp                    # Kernel entry file, which contains the main function and scheduling logic
│   │   │   └── ${op_name}_aicpu.h                      # Kernel header file, which contains function declarations, structure definitions, and logic implementation
│   │   └── tests                                       # Operator test case directory
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # (Optional) UT test cases. Develop test cases based on the actual situation.
│   │           ├── CMakeLists.txt                      # CMakeLists file for UT cases
│   │           ├── graph_plugin                        # Directory for storing graph_plugin test cases
│   │           │   ├── CMakeLists.txt
│   │           │   └── fusion_pass                     # Directory for storing fusion pattern test cases
│   │           │       └── CMakeLists.txt
│   │           ├── op_host                             # Directory for storing op_host test cases
│   │           │   ├── CMakeLists.txt
│   │           │   ├── ${op_name}_regbase_tiling.h
│   │           │   ├── op_api                          # Directory for storing op_api test cases
│   │           │   │   ├── CMakeLists.txt
│   │           │   │   └── test_aclnn_${op_name}.cpp   # Operator aclnn test case file
│   │           │   ├── test_${op_name}_${sub_case}.cpp # op_host test case file in the ${sub_case} sub-scenario
│   │           │   ├── test_${op_name}.cpp             # op_host test case file
│   │           │   ├── test_${op_name}_infershape.cpp  # Operator InferShape test case file
│   │           │   └── test_${op_name}_tiling.cpp      # Operator tiling test case file
│   │           └── op_kernel                           # op_kernel test case directory
│   │               ├── CMakeLists.txt
│   │               │── test_${op_name}.cpp             # Operator kernel test case file
│   │               └── ${op_name}_data                 # (Optional) Data comparison and generation scripts on which the op_kernel test cases depend. If these scripts are not configured, you need to manually implement them in the corresponding test cases.
│   │                   ├── compare_data.py             # Data script
│   │                   └── gen_data.py                 # Data generation script
│   └── ...
├── docs                                                # Project-related document directory (zh: Chinese; en: English) 
├── examples                                            # End-to-end operator development and call examples
│   ├── add_example                                     # AI Core operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator use example directory
│   │   ├── op_graph                                    # Operator graph construction directory
│   │   ├── op_host                                     # Operator information library, tiling, and InferShape implementation directory
│   │   ├── op_kernel                                   # Operator kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── add_example_aicpu                               # AI CPU operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator use example directory
│   │   ├── op_graph                                    # Operator graph construction directory
│   │   ├── op_host                                     # Operator information library and InferShape implementation
│   │   ├── op_kernel_aicpu                             # Operator kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── CMakeLists.txt
│   ├── fast_kernel_launch_example                       # Lightweight and high-performance operator development project template
│   │   ├── ascend_ops                                  # Example operator implementation directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── README.md                                   # Description of the lightweight and high-performance operator development project
│   │   ├── requirements.txt
│   │   └── setup.py                                    # Build script
│   └── README.md                                       # Project example introduction
├── scripts                                             # Script directory, including configuration files related to custom operators and kernel building
├── tests                                               # Project-level test directory
│   ├── requirements.txt                                # Third-party components on which test cases depend
│   └── ut                                              # UT case project
│       ├── CMakeLists.txt                              # CMakeLists script of the UT project
│       ├── common                                      # Common code used in the UT project
│       ├── op_api                                      # op_api test project
│       ├── op_host                                     # op_host test project
│       └── op_kernel                                   # op_kernel test project
├── CMakeLists.txt                                      # CMakeLists entry of the project
├── CONTRIBUTING.md                                     # Project contribution guide
├── LICENSE                                             # Open source license information of the project
├── OAT.xml                                             # Configuration script, which is used by the code repository tool to check whether the license is standard.
├── README.md                                           # General project introduction
├── SECURITY.md                                         # Project security statement
├── build.sh                                            # Project compilation script
├── classify_rule.yaml                                  # Component classification information
├── install_deps.sh                                     # Script for installing the project dependency package
├── requirements.txt                                    # Third-party dependency package of the project
└── version.info                                        # Project version information
```

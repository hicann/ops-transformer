# Project Documentation

## Directory Description

The key directory structure is as follows:

```text
├── context                            # Common directory, storing documents including basic concepts, project directory introduction, build parameter description, and so on
│   ├── dir_structure.md
│   ├── build.md
│   └── ...
├── debug                              # Operator debugging and tuning document directory
│   ├── op_debug_prof.md
│   ├── ...
├── develop                            # Operator development document directory
│   ├── aicore_develop_guide.md
│   ├── ...
├── figures                            # Image directory
├── invocation                         # Operator invocation document directory (including aclnn invocation, graph mode invocation, and so on)
│   ├── op_invocation.md
│   ├── ...
├── op_api_list.md                     # Full operator interface list (aclnn)
├── op_list.md                         # Full operator list
└── README.md
```

## Document Description

The full project documents are as follows. Obtain the corresponding content as needed.

| Document | Description |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [Operator List](zh/op_list.md) | Introduces the list of all operators included in the project. |
| [aclnn List](zh/op_api_list_en.md) | Introduces all operator APIs included in the project. Operators can be directly invoked through this API. |
| [Environment Deployment](en/context/quick_install.md) | Introduces the basic environment setup process, including the acquisition and installation of software packages and third-party dependencies for different scenarios. |
| [Operator Invocation](en/invocation/quick_op_invocation.md) | Introduces how to compile source code and execute operators, including operator package compilation, operator sample execution, UT execution, and so on for different scenarios. |
| [Operator Development](en/develop/aicore_develop_guide.md) | Introduces how to develop new operators based on this project, including operator prototype definition, Tiling implementation, Kernel implementation, and so on. |
| [Operator Invocation Methods](en/invocation/op_invocation.md) | Introduces multiple operator invocation methods and invocation processes, such as aclnn invocation, graph mode invocation, and so on. |
| [Operator Debugging and Tuning](en/debug/op_debug_prof.md) | Introduces common operator debugging and tuning methods. |

## Appendix

| Document | Description |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](en/context/basic_concepts.md) | Introduces basic concepts and terminology related to the operator domain, such as quantization/sparse, data types, data formats, and so on. |
| [build Parameter Description](en/context/build.md) | Introduces the functionality and parameter meanings of the build.sh script in this project. |

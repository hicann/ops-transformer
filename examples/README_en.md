English | [简体中文](./README.md)

## Introduction

> Description:

> - For details about the AI Core, see the "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" section in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples for AI Core operators. Refer to the corresponding implementations based on your actual requirements.

## Directory Description

```
├── example
│   ├── add_example                # Directory name of the AddExample operator, generally in lowercase with underscores
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Implementations related to the operator information library, Tiling, and InferShape
│   │   ├── op_kernel              # Operator kernel directory
│   │   └── op_kernel_aicpu        # Operator kernel_aicpu directory
│   ├── attention                  # Examples of attention operators
│   │   └── flash_attn
│   │       └── L0_minimal         # Minimal FlashAttention sample, with only op_kernel and no Host Tiling
│   ├── mc2                        # Examples of communication-computation fusion operators
│   │   ├── all_gather_add         # Directory name of the AllGatherAdd operator, generally in lowercase with underscores
│   │   │   └── ...
│   ├── CMakeLists.txt             # Operator compilation configuration file; keep the original file
│   └── README.md                  # Operator description document

```

## Operator Development Samples

|Sample Directory| 	Sample Description	           |Operator Development|Operator Invocation |
|---|------------------|---|---|
| add_example | 	Operator that adds two tensors.	 | For the end-to-end operator development process, see the [AI Core Operator Development Guide](../docs/zh/develop/aicore_develop_guide.md). |For the invocation sample, see [README](add_example/README_en.md)|
| mc2/all_gather_add | 	Performs AllGather collective communication first, and then performs element-wise addition.	 | For the end-to-end operator development process, see the [AI Core Operator Development Guide](./mc2/all_gather_add/docs/AllGatherAdd_design_en.md). |For the invocation sample, see [README](mc2/all_gather_add/README_en.md)|
| mc2/ccu_alltoall | CCU direct-call (`<<<>>>`) AllToAll collective communication + Blaze MXFP8 MatMul dual-stream operator fusion (kernel direct-call sample). | — | For the invocation sample, see [README](mc2/ccu_alltoall/README_en.md)|
| attention/flash_attn/L0_minimal | 	Minimal FlashAttention implementation, with no mask, no combine, and no independent Host Tiling.	 | For the operator implementation details, see [FA_IMPL.md](attention/flash_attn/L0_minimal/ascend_ops/csrc/flash_attn_minimal/FA_IMPL_en.md). |For the invocation sample, see [README](attention/flash_attn/L0_minimal/README_en.md)|

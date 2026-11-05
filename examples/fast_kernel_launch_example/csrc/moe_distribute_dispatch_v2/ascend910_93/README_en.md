# MoeDistributeDispatchV2 (<<<>>> Direct Kernel Launch Version)

English | [简体中文](./README.md)

## Overview

This operator is a direct kernel launch implementation of the MoeDistributeDispatchV2 operator, and uses the <<<>>> invocation method to directly launch the Ascend C kernel.

For the core functions, computation formulas, and communication principles of the Dispatch operator, see [the aclnnMoeDistributeDispatchV3 documentation.

## Product Support

| Product | Supported |
|:-----|:--------:|
| Atlas A3 products | √ |

## Constraints

To fully enable the feature, the following version requirements must be met:
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

The platform supported by this operator is ascend910_93, while the default build environment is ascend910b. To build in the ascend910_93 build environment, run the following command first:
```sh
export NPU_SOC_VERSION=ascend910_93
```

## <<<>>> Direct Kernel Launch Invocation Flow

### Overall Call Chain

```
Python layer (MoeDistributeBuffer)
    │
    ├── Calls updateContext during initialization to obtain the communication context
    │
    └── Calls the npu_moe_distribute_dispatch_v2 method
            │
            ↓
        torch.ops.ascend_ops.MoeDistributeDispatchV2
            │
            ↓
        C++ Extension Layer (npu_moe_distribute_dispatch_v2)
            │
            ├── Input validation (ValidateMoeDistributeDispatchV2Input)
            │
            ├── Output Tensor shape computation and creation
            │
            ├── Tiling data construction (calculate_tilingdata)
            │
            ├── Obtains the NPU Stream
            │
            └── Calls moe_distribute_dispatch_v2_api
                    │
                    ├── Tensor address conversion
                    │
                    ├── TilingKey computation (calculate_tilingkey)
                    │
                    ├── Profiling integration
                    │
                    └── Calls moe_distribute_dispatch_v2_entry
                            │
                            ├── <<<>>> Kernel Launch
                            │       │
                            │       ├── moe_distribute_dispatch_v2_generic
                            │       │
                            │       └── SK mode (moe_distribute_dispatch_v2_generic_sk)
                            │
                            └── TilingKey branch selection (moe_dispatch_switch)
                                    │
                                    ↓
                                Ascend C kernel execution
```

### Directory Structure

```
moe_distribute_dispatch_v2/ascend910_93/
├── moe_distribute_dispatch_v2_torch.cpp      # Main implementation of the C++ Extension
├── moe_distribute_dispatch_v2_torch.h        # Helper functions for obtaining Tensor addresses
├── moe_distribute_dispatch_v2_torch_validate.cpp  # Input parameter validation
├── moe_distribute_dispatch_v2_torch_validate.h
├── moe_distribute_dispatch_v2_entry.cpp      # <<<>>> entry function
├── moe_distribute_dispatch_v2_entry.h
├── op_kernel/                                # Kernel implementation
└── CMakeLists.txt                            # Build configuration
```

## Invocation Flow Details

### 1. Python Layer - MoeDistributeBuffer

The MoeDistributeBuffer class encapsulates the complete logic of the operator invocation and provides users with a concise interface.

#### Initialization Logic

During initialization, MoeDistributeBuffer performs the following operations:

**Step 1: Obtain distributed information** - Obtains the rank_id and world_size of the current process through the torch.distributed API and stores them in member variables.

**Step 2: Obtain the HCCL communicator name** - Obtains the name string of the underlying HCCL communicator through group._get_backend, which is used for the subsequent creation of the communication context in the C++ layer.

**Step 3: Create the communication context** - Calls torch.ops.ascend_ops.updateContext, passing in the communicator name and world_size. In the C++ layer, this interface obtains the Window addresses of all cards through the HCCL API, packages them into an mc2_context Tensor, and stores the Tensor in a member variable for use when the kernel is invoked. The interface also returns the HCCL Buffer Size.

#### HCCL Buffer Size Computation

The get_ccl_buffer_size static method computes the minimum required HCCL Buffer size based on the configuration parameters:

- Selects different data alignment strategies based on the comm_alg type (fullmesh_v1/fullmesh_v2)
- Computes the Dispatch and Combine data usage of a single Token
- Computes the total Buffer requirement based on parameters such as world_size, the number of experts, and TopK
- Returns the Buffer size in MB. You can set the HCCL_BUFFSIZE environment variable according to the returned Buffersize, or set a Buffersize greater than the computed size by yourself.

#### Dispatch Invocation Logic

The npu_moe_distribute_dispatch_v2 method merges the user parameters with the cached context information and calls torch.ops.ascend_ops.MoeDistributeDispatchV2:

- **Passes in the cached context**: mc2_context, group_name, world_size, rank_id, ccl_buffer_size
- **Passes in user parameters**: x, expert_ids, moe_expert_num, quant_mode, comm_alg, and so on
- **Returns outputs**: expand_x, dynamic_scales, expand_idx (assist_info), expert_token_nums, ep_recv_counts, expand_scales

### 2. C++ Extension Layer

#### Torch Interface Registration

The operator signature is registered through TORCH_LIBRARY_FRAGMENT to define the input and output parameter types, and the binding is performed through TORCH_LIBRARY_IMPL.

#### Main Function

The main function npu_moe_distribute_dispatch_v2 executes the following flow:

**Stage 1: Input validation**

The parameters are packed into the MoeDistributeDispatchV2ValidateParams struct, and ValidateMoeDistributeDispatchV2Input is called to perform validation (see the next section for details). After validation, the computed dimension information (bs, h, k) is obtained from validate_params.

**Stage 2: Output Tensor Shape Computation**

The output shapes are computed based on the rank type (shared expert card or MoE expert card):

- Shared expert card: local_moe_expert_num=1, and the number of output rows a is max_bs * max_shared_group_num
- MoE expert card: local_moe_expert_num is moe_expert_num divided by the number of MoE expert cards, and the number of output rows a is global_bs * min(local_moe_expert_num, k)

**Stage 3: Output Tensor Creation**

The output Tensors are created according to the computed Shapes and the configuration:
- expand_x: a rows and h columns. The data type is selected according to the quantization mode (the original type for the non-quantized mode, and INT8 for the quantized mode)
- dynamic_scales: a rows, with the data type Float
- assist_info_forcombine: length of max(bs*k, a*128), with the data type Int (stores triples)
- expert_token_nums: length of local_moe_expert_num, with the data type Long
- ep_recv_counts: length of ep_world_size * local_moe_expert_num, with the data type Int
- expand_scales: a rows, with the data type Float
- new_workspace: an Int Tensor of 16MB, used for inter-kernel synchronization

**Stage 4: Tiling Data Construction**

calculate_tilingdata is called to construct the Tiling data from the input parameters and Tensor information: the passed configuration parameters (ep_world_size, ep_rank_id, moe_expert_num, quant_mode, global_bs, and so on) are directly assigned to the corresponding fields of the struct; the dimension information (bs, h, k) is obtained from the input Tensors; the number of AIV cores and the UB size are obtained from the platform APIs; and the Mask type and smoothScale configuration are determined based on the presence of the optional Tensors.

**Stage 5: Obtain the Stream and Invoke Asynchronously**

The current NPU Stream is obtained, a lambda function is constructed to call moe_distribute_dispatch_v2_api, and the function is executed asynchronously through OpCommand::RunOpApiV2.

#### Input Validation Flow

ValidateMoeDistributeDispatchV2Input performs validation in the following order:

| Step | Validation Item | Constraint |
|-----|---------|------|
| 1 | EP domain parameters | epWorldSize ∈ [2, 768], epRankId ∈ [0, epWorldSize) |
| 2 | Shared expert parameters | The combination of sharedExpertNum and sharedExpertRankNum must be valid |
| 3 | MoE expert parameters | moeExpertNum ∈ (0, 1024], and it must be divisible by the number of MoE expert cards |
| 4 | Quantization mode | Only 0 (non-quantized) and 2 (PerToken dynamic quantization) are supported |
| 5 | Common parameters | expertTokenNumsType ∈ {0,1}, and the number of special experts >= 0 |
| 6 | Tensor dimensions | x must be 2D, and expert_ids must be 2D |
| 7 | Parameter ranges | h ∈ [1024, 8192], bs ∈ (0, 512], k ∈ (0, 16] (upper limit 12 for fullmesh_v2) |
| 8 | global_bs constraint | If it is not 0, it must equal bs*ep_world_size |
| 9 | Data types | x supports BF16/FP16, expert_ids must be Int32, and scales must be Float |

During validation, intermediate parameters such as localMoeExpertNum and zeroComputeExpertNum are computed, and the Mask type (1D or 2D) is determined.

### 3. API Layer

The API layer function moe_distribute_dispatch_v2_api converts the Tensors into the address parameters required by the <<<>>> invocation, integrates Profiling, and calls the underlying Entry function.

#### Tensor Address Conversion

The get_first_tensor_address function obtains the device address pointers of the Tensors:

- For required Tensors (x, expert_ids, mc2_context, and output Tensors): data_ptr is obtained directly.
- For optional Tensors (scales, x_active_mask, expert_scales, performance_info): has_value is checked first. If the value exists, the address is obtained; otherwise, nullptr is passed.

#### TilingKey Computation

calculate_tilingkey computes a 5-digit decimal TilingKey based on the runtime configuration:

| Base value | 10000 |
|-------|-------|
| quantMode | Added to the ones digit (0 or 2) |
| xType is BF16 | Adds 10 (to the tens digit) |
| scales exists | Adds 100 (to the hundreds digit) |
| comm_alg is fullmesh_v2 | Adds 1000 (to the thousands digit) |

#### Profiling Integration

Profiling records detailed information about operator execution, facilitating performance analysis and debugging.

**Working principle**: The `aclprofRangePushEx` and `aclprofRangePop` APIs wrap the <<<>>> invocation to form a profiling range. Within this range, the metadata of the operator is recorded.

**Recorded information**:
- **Operator identifiers**: opName (the operator name string converted to an ID) and opType (the operator type string converted to an ID)
- **Execution configuration**: blockDim (number of AIV cores), kernelType (kernel type, for example, MIX_AIV=5 indicates a mixed AIV kernel), and stream (execution stream)
- **Tensor information**: For each input/output tensor, the type (input 0/output 1), format (data format), dataType (data type), and shape (dimension information)
- **Operator performance**: The time consumed by each execution of the operator, including the detailed breakdown of vector, scalar, mte, and other items


### 4. Entry Layer

The Entry layer contains the <<<>>> invocation entry and the SK adaptation, which are defined in the moe_distribute_dispatch_v2_entry.cpp and moe_distribute_dispatch_v2_entry.h files.

#### <<<>>> Invocation Entry

The moe_distribute_dispatch_v2_entry function uses the <<<>>> syntax to launch the moe_distribute_dispatch_v2_generic kernel, passing in the TilingKey, the number of Blocks (number of AIV cores), the Stream, the addresses of the Tensors, and the TilingData.

#### SK Adaptation

SK (Super Kernel) is an operator fusion mechanism provided by Ascend C. It fuses multiple operators in the same model into one integral operator, avoiding host dispatch overhead.

**SK mode description**:
- Functions marked with `__sk__` are SK kernels and support operator fusion
- The `split_num` template parameter indicates the splitting mode. Four modes (0-3) are supported, corresponding to different inter-core task allocation strategies
- The `SK_BIND` macro binds the different splitting modes of the SK kernel to the main kernel, and the underlying scheduler automatically selects the optimal mode

**Parameter packing**: The `MoeDistributeDispatchV2SkArgs` struct packs all kernel parameters, facilitating passing during the SK invocation. The struct contains fields such as tilingKey, the addresses of all input/output tensors, workspace, mc2Context, and tilingData.

**Execution flow**: When the scheduler selects the SK mode for execution, it calls the `moe_distribute_dispatch_v2_generic_sk` function, parses the parameters from the SkArgs struct, and then executes the same logic as the ordinary kernel (calling `moe_dispatch_switch`).

### 5. Kernel Layer

#### TilingKey Branch Selection

The kernel entry moe_dispatch_switch selects the corresponding template instantiation based on the TilingKey through a switch statement:

- **10000 series**: calls the moe_distribute_dispatch_v2 template (fullmesh_v1 communication algorithm)
- **11000 series**: calls the moe_distribute_dispatch_v2_full_mesh template (fullmesh_v2 communication algorithm)

The template parameters (input type, output type, quantization mode, and whether smoothScale exists) are implicitly determined by the digits of the TilingKey.

#### Overall Kernel Functions

The moe_distribute_dispatch_v2_generic kernel implements the MoE dispatch function based on the HCCL Window mechanism: it sends the Token data to the Window of each target card according to expert_ids, waits for all cards to finish sending, and then reads the data from the Window and writes it into the output Tensors. It supports features such as PerToken dynamic quantization, the Token active Mask, and the Expert active Mask.

For the detailed algorithm principles, communication flows, and data layouts of the kernel, see [aclnnMoeDistributeDispatchV3_en](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3_en.md).

## MC2 Context Description

### Why Context Is Needed

Under the traditional HCCL communication mode, the kernel obtains the Window addresses of all ranks through HCCL APIs such as `GetHcclContext`. This mode depends on HcclOpParam created by HCCL through the framework. In the <<<>>> direct launch scenario, the kernel is invoked directly through the torch interface, so HCCL does not create HcclOpParam through the calling framework. Therefore, the communication context required by dispatch and combine needs to be passed directly into the kernel as parameters.

### Context Creation Flow

See [update_context_en.md](../../update_context/update_context_en.md) for details. The core flow consists of two stages:

1. **CreateHcclContext**: establishes the HCCL communication framework, obtains the communication handle through `HcclCommGetHandleWithName`, and creates the operation resource context through `HcclCreateOpResCtx`.

2. **CreatMc2Context**: fills in the IPC buffer addresses. It traverses all ranks, obtains the IPC buffer address of each rank through `HcclGetHcclBuffer` (local) and `HcclGetRemoteIpcHcclBuf` (remote), and stores the addresses in the `epHcclBuffer` array.

### Use of Context Within the Kernel

Within the kernel, the context addr replaces the `GetBaseWindAddrByRankId()` function to obtain the Window address of the target rank: when the Context mode is used, the address is obtained directly from the `mc2Context_->epHcclBuffer_[rankId]` array according to the peer rank ID.

After obtaining the peer address, the first 1MB is used as the status area address, and the rest is used as the data area address.

## Differences from aclnnMoeDistributeDispatchV3

### Parameter Comparison

| Parameter | aclnn version | <<<>>> version |
|-----|----------|-----------|
| epRankId, epWorldSize, groupEp | Passed in by the user | Automatically obtained by MoeDistributeBuffer |
| tp domain parameters | Supported | Not supported |
| totalWinsizeEp size | The user configures the environment variable by themselves | The get_ccl_buffer_size method can be called for computation, or the user configures it by themselves |
| mc2_context | Not supported and not required | A required parameter. The user does not need to pass it in; the MoeDistributeBuffer class builds it |

### Supported Quantization Modes

Only the following modes are supported: non-quantized (0) and PerToken dynamic quantization (2)

### Supported commAlg Values

Supported values: "" (fullmesh_v1 by default), fullmesh_v1, fullmesh_v2

## Invocation Example

For complete examples, see [test\_combine.py](../../../tests/moe_distribute_combine_v2/test_combine.py) and [test\_combine\_graph.py](../../../tests/moe_distribute_combine_v2/test_combine_graph.py)

The basic flow:

1. Initialize the distributed environment and create the EP communicator
2. Create a MoeDistributeBuffer instance
3. Call npu_moe_distribute_dispatch_v2 to perform dispatch
4. Call npu_moe_distribute_combine_v2 to perform combine (using the Dispatch outputs)

## Build Description

Before building, set the environment variable: export NPU_SOC_VERSION=ascend910_93, source the CANN package environment, and activate conda.

Compilation method: enter the examples\fast_kernel_launch_example directory and run the command: python3 -m build --wheel -n
After compilation, run the command
python3 -m pip install ./dist/*.whl --force-reinstall --no-deps
to install ascend_ops.

For the detailed build process, refer to [README_en.md](../../../../README_en.md).

The shared library is built through CMake and loaded by the PyTorch Extension.

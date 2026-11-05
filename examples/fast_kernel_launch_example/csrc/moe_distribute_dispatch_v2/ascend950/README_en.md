# MoeDistributeDispatchV2 (<<<>>> Direct Kernel Launch Version)

English | [简体中文](./README.md)

## Overview

This operator is a direct kernel launch implementation of the MoeDistributeDispatchV2 operator for the A5 (ascend950), and uses the <<<>>> invocation method to directly launch the Ascend C kernel. The kernel implementation directly references the `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/` headers in the repository (resolved through the repository-root include path, with no local copies).

For the core functions, computation formulas, and communication principles of the Dispatch operator, see the [aclnnMoeDistributeDispatchV3](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3_en.md) documentation.

## Product Support

| Product | Supported |
|:-----|:--------:|
| Atlas A5 products | √ |

## Constraints

To fully enable the feature, the following version requirements must be met:
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

The platform supported by this operator is ascend950, while the default build environment is ascend910b. To build in the ascend950 build environment, run the following command first:
```sh
export NPU_SOC_VERSION=ascend950
```

Current scope of support: non-quantized (quant_mode=0) and perToken dynamic quantization (quant_mode=2, int8 output), BF16/FP16 input, tp_world_size=1, and MTE communication only.

## <<<>>> Direct Kernel Launch Invocation Flow

### Overall Call Chain

```
Python layer (test scripts)
    │
    ├── Initializes the torch.distributed (hccl) communicator
    │
    └── Calls torch.ops.ascend_ops.MoeDistributeDispatchV2
            │
            ↓
        C++ Extension Layer (npu_moe_distribute_dispatch_v2)
            │
            ├── Input validation (ValidateMoeDistributeDispatchV2Input)
            │
            ├── Output Tensor shape computation and creation
            │
            ├── Tiling data construction (calculate_tilingdata_a5, including the mc2 communication tiling)
            │
            ├── Obtains the NPU Stream
            │
            └── Calls moe_distribute_dispatch_v2_api
                    │
                    ├── Tensor address conversion
                    │
                    ├── TilingKey computation (calculate_tilingkey_a5)
                    │
                    ├── Tiling data H2D copy (aclrtMemcpyAsync to tilingGM)
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
                            └── TilingKey branch selection (moe_dispatch_a5_switch)
                                    │
                                    ↓
                                Ascend C kernel execution
```

### Directory Structure

```
moe_distribute_dispatch_v2/ascend950/
├── moe_distribute_dispatch_v2_torch.cpp      # Main implementation of the C++ Extension
├── moe_distribute_dispatch_v2_torch.h        # Helper functions for obtaining Tensor addresses
├── moe_distribute_dispatch_v2_torch_validate.cpp  # Input parameter validation
├── moe_distribute_dispatch_v2_torch_validate.h
├── moe_distribute_dispatch_v2_entry.cpp      # <<<>>> entry function and SK adaptation
├── moe_distribute_dispatch_v2_entry.h
├── mc2_cc_tiling_shim.cpp                    # mc2 communication tiling shim (legacy ABI linking shim)
├── mc2_cc_tiling_shim.h
└── CMakeLists.txt                            # Build configuration
```

## Invocation Flow Details

### 1. Python Layer

The A5 kernel builds the communication context through the updateContext flow (aligned with A3). The Python layer needs to:

- Initialize the EP communicator (backend=hccl) through torch.distributed, and obtain the communicator name
- Compute the HCCL Buffer Size based on the communication parameters and set the HCCL_BUFFSIZE environment variable
- Create a MoeDistributeBuffer instance (which internally completes the HCCL communicator initialization through updateContext)
- Call torch.ops.ascend_ops.MoeDistributeDispatchV2, where mc2_context is filled in automatically by the wrapper class

### 2. C++ Extension Layer

#### Torch Interface Registration

The operator signature is registered through TORCH_LIBRARY_FRAGMENT to define the input and output parameter types, and the binding is performed through TORCH_LIBRARY_IMPL.

#### Main Function

The main function npu_moe_distribute_dispatch_v2 executes the following flow:

**Stage 1: Input validation**

The parameters are packed into the MoeDistributeDispatchV2ValidateParams struct, and ValidateMoeDistributeDispatchV2Input is called to perform validation. After validation, the computed dimension information (bs, h, k) is obtained from validate_params. quant_mode=0 (non-quantized) and quant_mode=2 (perToken dynamic quantization, int8 output) are supported.

**Stage 2: Output Tensor Shape Computation**

The output shapes are computed based on the rank type (shared expert card or MoE expert card):

- Shared expert card: local_moe_expert_num=1, and the number of output rows a is max_bs * max_shared_group_num
- MoE expert card: local_moe_expert_num is moe_expert_num divided by the number of MoE expert cards, and the number of output rows a is global_bs * min(local_moe_expert_num, k)

**Stage 3: Output Tensor Creation**

The output Tensors are created according to the computed Shapes and the configuration: expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, expand_scales, and new_workspace (16MB, inter-kernel synchronization). In addition, tiling_buffer is created (a Byte Tensor of the size of the TilingData struct, used as the H2D target).

**Stage 4: Tiling Data Construction**

calculate_tilingdata_a5 is called to construct `MoeDistributeDispatchV2TilingData` (containing the three parts mc2InitTiling/mc2CcTiling1/moeDistributeDispatchV2Info):

- mc2 communication tiling: `AscendC::Mc2CcTilingConfig` (HCCL_CMD_ALLTOALLV + AIV_ENGINE) fills in mc2InitTiling and mc2CcTiling1
- info fields: the configuration parameters are assigned directly (epWorldSize, epRankId, moeExpertNum, globalBs, and so on); A5 does not support TP communication (tpWorldSize=1); totalWinSizeEp = HCCL_BUFFSIZE(MB)×1024²×2 (aligned with mc2 GetMaxWindowSize×2); hasExpertScales is determined by whether the expert_scales input exists; the number of AIV cores and the UB size are obtained from the platform APIs

**Stage 5: Obtain the Stream and Invoke Asynchronously**

The current NPU Stream is obtained, a lambda function is constructed to call moe_distribute_dispatch_v2_api, and the function is executed asynchronously through OpCommand::RunOpApiV2.

### 3. API Layer

The API layer function moe_distribute_dispatch_v2_api converts the Tensors into the address parameters required by the <<<>>> invocation, integrates Profiling, and calls the underlying Entry function.

#### Tensor Address Conversion

The get_first_tensor_address function obtains the device address pointers of the Tensors: for required Tensors, data_ptr is obtained directly; for optional Tensors, has_value is checked first, and nullptr is passed if the value does not exist.

#### TilingKey Computation

calculate_tilingkey_a5 computes a 4-digit decimal TilingKey based on the runtime configuration:

| Base value | 20000 |
|-------|-------|
| quantMode | Added to the ones digit (0 or 2) |
| xType is BF16 | Adds 10 (to the tens digit) |
| scales exists (smoothScale) | Adds 100 (to the hundreds digit) |

#### Tiling Data H2D

The TilingData struct is copied to tiling_buffer through `aclrtMemcpyAsync` (HostToDevice), and is read within the kernel through the tilingGM pointer using `GET_TILING_DATA_WITH_STRUCT`.

#### Profiling Integration

`aclprofRangePushEx` and `aclprofRangePop` wrap the <<<>>> invocation to form a profiling range, in which the operator metadata and Tensor information are recorded.

### 4. Entry Layer

The Entry layer contains the <<<>>> invocation entry and the SK adaptation, which are defined in the moe_distribute_dispatch_v2_entry.cpp and moe_distribute_dispatch_v2_entry.h files.

#### <<<>>> Invocation Entry

The moe_distribute_dispatch_v2_entry function uses the <<<>>> syntax to launch the moe_distribute_dispatch_v2_generic kernel, passing in the TilingKey, the number of Blocks (number of AIV cores), the Stream, the addresses of the Tensors, and tilingGM.

#### SK Adaptation

SK (Super Kernel) is an operator fusion mechanism provided by Ascend C. It fuses multiple operators in the same model into one integral operator, avoiding host dispatch overhead.

**SK mode description**:
- Functions marked with `__sk__` are SK kernels and support operator fusion
- The `split_num` template parameter indicates the splitting mode. Four modes (0-3) are supported, corresponding to different inter-core task allocation strategies
- The `SK_BIND` macro binds the different splitting modes of the SK kernel to the main kernel, and the underlying scheduler automatically selects the optimal mode

**Parameter packing**: The `MoeDistributeDispatchV2SkArgs` struct packs all kernel parameters, facilitating passing during the SK invocation. The struct contains fields such as tilingKey, the addresses of all input/output tensors, workspace, and tilingGM.

**Execution flow**: When the scheduler selects the SK mode for execution, it calls the `moe_distribute_dispatch_v2_generic_sk` function, parses the parameters from the SkArgs struct, and then executes the same logic as the ordinary kernel (calling `moe_dispatch_a5_switch`).

### 5. Kernel Layer

#### TilingKey Branch Selection

The kernel entry moe_dispatch_a5_switch selects the corresponding template instantiation based on the TilingKey through a switch statement (MTE non-fullmesh only):

- **20000/20010**: non-quantized (quant_mode=0), fp16/bf16, without smoothScale
- **20002/20012**: perToken dynamic quantization (quant_mode=2, int8 output), fp16/bf16, without smoothScale
- **20102/20112**: perToken dynamic quantization + smoothScale, fp16/bf16

The template parameters (input type, output type, quantization mode, and whether smoothScale exists) are implicitly determined by the digits of the TilingKey.

#### Overall Kernel Functions

The moe_distribute_dispatch_v2_generic kernel implements the MoE dispatch function based on the HCCL Window mechanism: it sends the Token data to the Window of each target card according to expert_ids, waits for all cards to finish sending, and then reads the data from the Window and writes it into the output Tensors.

The kernel implementation directly references the mc2 arch35 headers in the repository (`moe_distribute_dispatch_v2.h`, resolved through the repository-root include path), with no local copies. For the detailed algorithm principles, communication flows, and data layouts of the kernel, see [aclnnMoeDistributeDispatchV3_en](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3_en.md).

## MC2 Context Description

The A5 kernel builds the HCCL communication context through the updateContext flow (aligned with A3): updateContext produces the `HcclCombinOpParam` layout context that the kernel reads directly (workSpace/winSize/windowsIn/windowsOut/rankId/rankDim), and passes it into the kernel as the mc2Context parameter, which is used directly by the kernel class `Init(mc2Context, ...)`.

## Differences from aclnnMoeDistributeDispatchV3

### Parameter Comparison

| Parameter | aclnn version | <<<>>> version |
|-----|----------|-----------|
| epRankId, epWorldSize, groupEp | Passed in by the user | Passed in by the caller |
| tp domain parameters | Supported | Not supported (A5 does not support TP communication currently) |
| totalWinsizeEp size | The user configures the environment variable by themselves | The Buffer Size computation can be used, or the user configures it by themselves (the kernel side uses it with the HCCL_BUFFSIZE×2 semantics) |
| mc2_context | Not supported and not required | A required parameter, built by updateContext (aligned with A3) |

### Supported Quantization Modes

The following modes are supported: non-quantized (0) and PerToken dynamic quantization (2, int8 output)

### Supported commAlg Values

Supported values: "" (MTE by default, non-fullmesh)

## Invocation Example

The basic flow:

1. Initialize the distributed environment and create the EP communicator
2. Set HCCL_BUFFSIZE
3. Call torch.ops.ascend_ops.MoeDistributeDispatchV2 to perform dispatch
4. Call torch.ops.ascend_ops.MoeDistributeCombineV2 to perform combine (using the Dispatch outputs)

## Build Description

Before building, set the environment variable: export NPU_SOC_VERSION=ascend950, source the CANN package environment, and activate conda.

Compilation method: enter the examples\fast_kernel_launch_example directory and run the command: python3 -m build --wheel -n
After compilation, run the command
python3 -m pip install ./dist/*.whl --force-reinstall --no-deps
to install ascend_ops.

For the detailed build process, refer to [README_en.md](../../../../README_en.md).

The shared library is built through CMake and loaded by the PyTorch Extension.

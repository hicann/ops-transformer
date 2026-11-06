# FlashAttnGrad

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------- | ------ |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: Computes the gradients of Q/K/V (dq/dk/dv) based on the intermediate results of the forward attention computation (softmax_lse, attn_out) and the upstream gradient (do). When `sinks` is provided, the gradient of the sink (`dsink`) is also computed.
- Computation formula:

where $P = Softmax(scale \cdot QK^T - softmax\_lse)$. The backward computation formulas are:

$$
dV=P^TdO
$$

$$
dQ=scale \cdot (dS \cdot K)
$$

$$
dK=scale \cdot (dS^T \cdot Q)
$$

When `sinks` is provided, the forward softmax denominator additionally contains the sink term (this contribution is already included in `lse`),
and its gradient is reduced over batch / S1 / S2:

$$
dSink=-\sum_{b,s1,s2} P \cdot dP \cdot \exp(sink-softmax\_lse)
$$

where `dP=dO \cdot V^T`. The kernel processes each task (an s2 block). Within the block, it multiplies `P \cdot dP` row by row by
`\exp(sink-softmax\_lse)` and reduces the result, then further reduces it into `dsink`: each AIV sub-core accumulates the partial sums of its task
into its own exclusive slot row in the workspace using fp32 `atomicAdd` (one row per AIV; each row is written only by its own AIV, so no cross-core contention exists; the pre-stage zeroes the row before the main loop); after
`sync_all`, AIV0 reduces the rows into `dsink` in a fixed row order (aiv 0..aivCount-1).

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
|-----|-----------|----|---------|------|
| q | Input | Query, which is Q in the formula | BF16, FP16 | ND |
| k | Input | Key, which is K in the formula | BF16, FP16 | ND |
| v | Input | Value, which is V in the formula | BF16, FP16 | ND |
| do | Input | Upstream gradient, which is dO in the formula | BF16, FP16 | ND |
| attn_out | Input | Forward attention output | BF16, FP16 | ND |
| softmax_lse | Input | Forward softmax LSE | FP32 | ND |
| cu_seqlens_q | Optional input | Cumulative sequence lengths for the TND layout | INT32 | ND |
| cu_seqlens_kv | Optional input | Cumulative sequence lengths for the TND layout | INT32 | ND |
| seqused_q | Optional input | Actually used sequence lengths of Q | INT32 | ND |
| seqused_kv | Optional input | Actually used sequence lengths of KV | INT32 | ND |
| sinks | Optional input | Sink tensor | FP32 | ND |
| attn_mask | Optional input | Attention mask, used when mask_mode=3/4 | INT8 | ND |
| metadata | Optional input | FAG metadata tensor | INT32 | ND |
| dq | Output | Query gradient | BF16, FP16 | ND |
| dk | Output | Key gradient | BF16, FP16 | ND |
| dv | Output | Value gradient | BF16, FP16 | ND |
| dsink | Output | Sink gradient with shape (N1,). The torch interface returns it together with dq, dk, and dv | FP32 | ND |
| softmax_scale | Attribute | Softmax scaling factor. The default value 0.0 indicates 1/sqrt(d) | FLOAT | - |
| mask_mode | Attribute | Default 0. 0: full computation, 3: causal, 4: window | INT | - |
| win_left | Attribute | Left window of the window mask. Default -1 | INT | - |
| win_right | Attribute | Right window of the window mask. Default -1 | INT | - |
| max_seqlen_q | Attribute | Maximum sequence length of Q. Default -1 | INT | - |
| max_seqlen_kv | Attribute | Maximum sequence length of KV. Default -1 | INT | - |
| layout_q | Attribute | Q layout. Default BSND | STRING | - |
| layout_kv | Attribute | KV layout. Default BSND | STRING | - |
| layout_out | Attribute | Output layout. Default BSND | STRING | - |

## Invocation

| Invocation Method | Invocation Example | Description |
|---------|---------|------|
| PyTorch API | [Invocation example](docs/torchapi_flash_attn_grad_en.md#example) | Call the operator through [flash_attn_grad](docs/torchapi_flash_attn_grad_en.md). First use flash_attn_metadata to generate the metadata (is_grad_enabled=True), then call this operator. For compilation and installation, see Quick Start below. |

## Quick Start

### 1. Compiling and Installing the custom Package

Complete script (integrates the compilation and installation steps; `CANN_DIR` is the CANN installation root directory, adjusted according to the actual environment):
```bash
# Prerequisite: load the CANN environment
source ${CANN_DIR}/cann/set_env.sh

# Clean up historical build artifacts (residual files may affect incremental compilation)
rm -rf ./build ./build_out
rm -rf ${CANN_DIR}/vendors

# Compile (flash_attn_grad and flash_attn_metadata must be compiled together; metadata generates backward kernel-splitting information)
bash build.sh --pkg --soc=ascend950 --ops=flash_attn_grad,flash_attn_metadata

# Install
cd build_out
./cann-ops-transformer-*.run --install-path=${CANN_DIR}
```

Compilation artifact: `build_out/cann-ops-transformer-custom_linux-x86_64.run`

**Optional Parameter Description**:
- **`-j` (limits the number of parallel threads)**: By default, compilation runs in parallel based on the number of machine cores. When the machine memory is insufficient, or the number of cores actually limited by cgroup is smaller than the value reported by `/proc/cpuinfo`, causing compilation OOM or failure, explicitly specify a smaller value:

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn_grad,flash_attn_metadata -j16
  ```

### 2. Building and Installing the torch Extension Package

This step is mandatory before using the torch interface. Build the torch extension whl in the repository root directory and install it:

```bash
# Prerequisite: load the CANN environment
source ${CANN_DIR}/cann/set_env.sh

# Clean the torch extension cache (~ refers to the home directory of the current user, which must be consistent with the user who installs/runs torch, to avoid loading stale build artifacts)
rm -rf ~/.cache/torch_extensions/*

# Build the torch extension whl (full package; the package name cann_ops_transformer remains unchanged; the whl is output to build_out/)
bash build.sh --torch_extension --soc=ascend950

# Install
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

**Post-installation Verification**:
```bash
python3 -c "from cann_ops_transformer.ops import flash_attn_grad; print('ok')"
```

### 3. Interface Invocation

- **torch interface** (function prototype, parameter description, and return value description of `flash_attn_grad`): [torchapi_flash_attn_grad.md](docs/torchapi_flash_attn_grad_en.md)

Invocation consists of two steps. First, use `flash_attn_metadata` to generate the backward kernel-splitting metadata (you must set `is_grad_enabled=True`, and ensure that it is consistent with the shape, layout, mask, and sequence length parameters of the main operator). Then call the `flash_attn_grad` main operator. For the complete invocation example (including code), see the [Invocation example](docs/torchapi_flash_attn_grad_en.md#example) section in the interface documentation.

The import path is consistent with the installation package name (the full package built in step 2):

```python
from cann_ops_transformer.ops import flash_attn_grad
```

## Operator Directory Structure

| File | Description |
|------|------|
| `op_kernel/flash_attn_grad.py` | pypto-pro kernel implementation (BN2GS1S2 template) |
| `op_host/flash_attn_grad_tiling.cpp` | Tiling implementation (tilingkey encoding and workspace layout) |
| `op_host/flash_attn_grad_def.cpp` | Operator definition (op_proto/inputs/outputs/attributes) |
| `op_host/flash_attn_grad_infershape.cpp` | Shape/dtype inference |
| `op_host/config/ascend950/flash_attn_grad_binary.json` | Binary operator configuration |
| `torch_extension/flash_attn_grad.py` | PyTorch operator schema and Python binding |
| `torch_extension/csrc/flash_attn_grad.cpp` | PyTorch C++ extension |

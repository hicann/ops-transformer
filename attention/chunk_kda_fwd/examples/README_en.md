# ChunkKdaFwd Invocation Sample

English | [简体中文](./README.md)

`test_aclnn_chunk_kda_fwd.cpp` demonstrates the complete process of calling `ChunkKdaFwd` through the two-stage aclnn API, including:

1. Initializing AscendCL, the device, the context, and the stream.
2. Constructing valid inputs, allocating device memory, and creating contiguous ND tensors.
3. Calling `aclnnChunkKdaFwdGetWorkspaceSize` to obtain the workspace size and the executor.
4. Allocating the workspace and calling `aclnnChunkKdaFwd` to perform the computation.
5. Copying `attn_out`, `Aqk`, and `Akk` to the host side, checking that they are finite values, and printing the first 5 elements.
6. Releasing the tensors, device memory, and AscendCL resources.

## Sample Configuration

The sample uses the most basic fixed-length BSND layout:

| Parameter | Value or Description |
| :--- | :--- |
| Product | Ascend 950PR&950DT products, Atlas A3 products, Atlas A2 products |
| Data type | `q`, `k`, `v`, `attn_out`, `Aqk`, and `Akk` are BFLOAT16; `g`, `beta`, and `a_log` are FLOAT32 |
| Scale | B=1, S=64, H=HV=2, K=V=128, C=64 |
| `layout` | `BSND` |
| gate configuration | `use_gate_in_kernel=true`, `safe_gate=true`, `lower_bound=-5.0` |
| Optional inputs | `dt_bias`, `initial_state`, `cu_seqlens`, and `chunk_indices` are not passed |
| Optional outputs | Only the fixed `attn_out`, `Aqk`, and `Akk` are exported |

The input and output shapes are as follows:

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `q`, `k` | `[1, 64, 2, 128]` | Query and Key |
| `v` | `[1, 64, 2, 128]` | Value |
| `g` | `[1, 64, 2, 128]` | Raw gate when `use_gate_in_kernel=true` |
| `beta` | `[1, 64, 2]` | Delta coefficient |
| `a_log` | `[2]` | Gate parameter of each Value head |
| `attn_out` | `[1, 64, 2, 128]` | Attention output |
| `Aqk`, `Akk` | `[1, 2, 64, 64]` | Intra-chunk Query-Key coefficient matrices and triangular inversion results |

For the API, formulas, and complete parameter constraints, see [aclnnChunkKdaFwd](../docs/aclnnChunkKdaFwd_en.md).

## Compilation and Running

Run the commands in the repository root directory of ops-transformer. For Ascend 950, use `--soc=ascend950`; for Atlas A2/A3, replace it with the SoC corresponding to the build environment.

```bash
bash build.sh --pkg --soc=ascend950 --ops=chunk_kda_fwd
bash build_out/cann-ops-transformer-*.run --quiet --install-path=<install_path>
source <install_path>/vendors/custom_transformer/bin/set_env.bash
bash build.sh --run_example chunk_kda_fwd eager cust --soc=ascend950 --vendor_name=custom
```

After successful execution, `build.sh` outputs `execute samples success`, and the sample itself outputs the following content:

```text
ChunkKdaFwd execution succeeded.
  attnOut[0] = ...
  ...
  Aqk[0] = ...
  ...
  Akk[0] = ...
  ...
```

The sample verifies that all printed results are finite values. The numerical results can be used to confirm that the invocation chain and output layout are correct. To perform a numerical comparison with the reference implementation, use the [pytest test framework](../tests/pytest/README_en.md).

# EngramGate Invocation Samples

English | [简体中文](./README.md)

## Applicable Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR/950DT products</term> | √ |
| <term>Atlas A3 training/inference products</term> | × |
| <term>Atlas A2 training/inference products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

`test_aclnn_engram_gate.cpp` demonstrates the two-stage aclnn API:

1. Initialize AscendCL, the device, and the stream.
2. Construct `hidden_states`, `kv`, and `weight_fused`; allocate device memory and create ND tensors.
3. Call `aclnnEngramGateGetWorkspaceSize` for the workspace size and executor.
4. Allocate the workspace and execute `aclnnEngramGate`.
5. Copy `dot` to the host and check the unnormalized weighted dot product.
6. Release tensors, device memory, and AscendCL resources.

`test_geir_engram_gate.cpp` demonstrates the GE IR interface:

1. Initialize GE and create a Session.
2. Add input Data nodes and an EngramGate node with the corresponding descriptions and attributes.
3. Add and execute the graph.
4. Check the shapes, dtypes, and numerical values of the primary and four intermediate outputs.
5. Release the Session and GE resources.

## Parameters

The aclnn example uses:

| Parameter | Value or description |
| :--- | :--- |
| Product | Ascend 950PR/950DT products |
| Data types | `hidden_states`, `kv`, and `output` are FLOAT16; `weight_fused` and the four intermediates are FLOAT32 |
| Shape | T=2, H=4, D=64 |
| Input values | `hidden_states`, Key, Value, and `weight_fused` are all 1 |
| Attributes | `eps=1e-6`, `clamp_value=1e-6`, `save_for_backward=true` |
| Output check | Every element of `dot` is 64 |

The GE IR example uses FLOAT16 and BFLOAT16, both `save_for_backward` values, and the following shapes:

| T | H | D | Coverage |
| :---: | :---: | :---: | :--- |
| 2 | 4 | 64 | Smallest D |
| 2 | 4 | 5120 | Network model shape |
| 2 | 4 | 32768 | Chunked computation over D |

See [aclnnEngramGate](../docs/aclnnEngramGate.md) and [EngramGate](../README_en.md) for the interfaces, formulas, and full constraints.

## Constraints

- The examples use device 0; change the device ID in the examples to use a different device.
- Install the EngramGate operator package and source its environment script before running the examples.
- The aclnn example checks `dot`; the GE IR example checks the primary output and all four intermediate outputs.

## Build and Run

Run from the ops-transformer repository root with `--soc=ascend950`.

```bash
bash build.sh --pkg --soc=ascend950 --ops=engram_gate
bash build_out/cann-ops-transformer-*.run --quiet --install-path=<install_path>
source <install_path>/vendors/custom_transformer/bin/set_env.bash
```

Run the aclnn example:

```bash
bash build.sh --run_example engram_gate eager cust \
  --soc=ascend950 --vendor_name=custom --example_name=test_aclnn_engram_gate
```

Run the GE IR example:

```bash
bash build.sh --run_example engram_gate graph cust \
  --soc=ascend950 --vendor_name=custom --example_name=test_geir_engram_gate
```

On success, the aclnn example prints:

```text
EngramGate PASS; dot=64, workspace=...
```

On success, the GE IR example prints:

```text
EngramGate GE IR: 12/12 PASS
```

# APACE

English | [简体中文](./README.md)

## Directory Structure

```
apace/
├── kernel/                       # Communication-computation fusion operator implementations
│   ├── fusions/                  #   Communication + Matmul fusion operators
│   │   ├── all_gather_quant_matmul/
│   │   └── all_to_all_quant_matmul/
│   └── matmul/                   #   Matmul kernels
│       ├── grouped_matmul/
│       └── quant_batch_matmul/
├── block/                        # Reusable single-core computation block interfaces
│   ├── epilogue/                 #   Epilogue post-processing blocks
│   ├── mmad/                     #   MMAD blocks (quantization MMAD fragments)
│   └── scheduler/                #   Scheduler blocks
├── core/                         # Core communication and computation interfaces
│   ├── aiv_comm/                 #   AIV collective communication interfaces
│   │   ├── all_gather/           #     AllGather communication implementation
│   │   ├── all_to_all/           #     AllToAll communication implementation
│   │   ├── barrier/              #     Synchronization barrier
│   │   ├── collective_comm_api.h
│   │   ├── collective_comm_base.h
│   │   └── collective_comm_context.h
│   └── aiv_compute/              #   AIV vector computation interfaces
├── basic/                        # Basic data structures and abstractions
│   └── fragment_tensor/          #   FragmentTensor (unified abstraction over multiple fragments)
├── tiling/                       # Tiling algorithms
├── utils/                        # Common utilities and constants
├── tests/                        # Tests
│   └── st/                       #   System-level tests
├── docs/                         # Design documentation
└── README.md
```

---

## Module Description

### kernel Operator Layer

Implementation of communication-computation fusion operators. The kernel layer performs pipeline orchestration of communication and computation, AIV-AIC collaborative scheduling, synchronization, and other tasks. You can call the implementations directly or use them as a reference.

| Subdirectory | Description |
|:---|:---|
| `fusions` | Complete kernel implementations that fuse communication with Matmul (for example, AllGather/AllToAll + quantization Matmul). |
| `matmul` | Standalone Matmul kernels, including Grouped Matmul and quantization Batch Matmul. |

### block Interface Layer

Reusable single-core Matmul computation block interfaces, organized by computation stage. They support the kernel layer in combining blocks to build fusion operators.

| Subdirectory | Description |
|:---|:---|
| `mmad` | Quantization MMAD fragments based on the [ops-tensor](https://atomgit.com/cann/ops-tensor) Blaze library, connected to FragmentTensor. |
| `epilogue` | Matmul post-processing blocks. |
| `scheduler` | Scheduling blocks for multiple cores and multiple blocks. |

### core Layer

Core communication and computation interfaces that provide underlying capabilities for the block and kernel layers.

| Subdirectory | Description |
|:---|:---|
| `aiv_comm` | Collective communication interfaces with the AIV core as the initiation engine. They provide AllGather / AllToAll / Barrier capabilities based on hcomm + UDMA, including CRTP base classes and a unified API. |
| `aiv_compute` | AIV core vector computation interfaces (quantization/dequantization, type conversion, reduction, and so on) that support data processing before and after communication. |

### basic Layer

Provides common basic data structures and abstraction capabilities for the upper-layer blocks and kernels.

| Subdirectory | Description |
|:---|:---|
| `fragment_tensor` | A unified abstraction over multiple GM fragments. It supports Slice / Copy / Scatter operations assembled by axis, simplifying cross-fragment data access. |

### tiling

Provides three types of content:
1. `matmul_tiling/quant_matmul_tiling.h`: Quantization Matmul partitioning engine (shared types + common workflow + SWAT specialization). `quant_matmul_tiling_data.h` is the serialized result of the host-device contract.
2. `tiling_data_struct_define.h`: Common data structure `CommTilingData` for communication partitioning (shared by host partitioning and kernel communication).
3. Communication partitioning strategies: `core_occupancy_first_tiling.h` (core utilization first; communication follows computation core partitioning) and `fit_balance_tiling.h` (communication-computation balancing based on a calibrated latency model; operator-specific logic and fitting data are maintained on the `op_host/op_tiling` side of each operator).

### utils

A collection of common utilities, including shared constants, common data structures, and HCCL channel creation tools.

---

## Relationship with MC2 Operators

Specific communication-computation fusion operators under `mc2/` (such as `all_gather_matmul`, `matmul_all_reduce`, and `matmul_all_to_all`) can be implemented by referencing the interfaces in this software:

- The `op_host/op_tiling` layer of an operator can use the partitioning algorithms in `apace/tiling` to determine tiling parameters.
- The `op_kernel` layer of an operator can directly call the `apace/kernel` implementation, or combine the `apace/block` and `apace/core` interfaces to build a fusion kernel.

```
mc2/<op>/op_host    ──┐
                      └──▶ apace/tiling       (tiling interfaces, providing partitioning algorithms)
mc2/<op>/op_kernel  ──┐
                      ├──▶ apace/kernel       (kernel interfaces, can be called directly or used as a reference)
                      ├──▶ apace/block        (computation block interfaces, combined for building)
                      └──▶ apace/core         (core communication/computation interfaces)
```

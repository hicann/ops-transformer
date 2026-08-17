# APACE

## 目录结构

```
apace/
├── kernel/                       # 通算融合算子实现
│   ├── fusions/                  #   通信 + Matmul 融合算子
│   │   ├── all_gather_quant_matmul/
│   │   └── all_to_all_quant_matmul/
│   └── matmul/                   #   Matmul kernel
│       ├── grouped_matmul/
│       └── quant_batch_matmul/
├── block/                        # 可复用的单核计算 block 接口
│   ├── epilogue/                 #   Epilogue 后处理 block
│   ├── mmad/                     #   MMAD block（量化 MMAD fragment）
│   └── scheduler/                #   调度 block
├── core/                         # 核心通信与计算接口
│   ├── aiv_comm/                 #   AIV 集合通信接口
│   │   ├── all_gather/           #     AllGather 通信实现
│   │   ├── all_to_all/           #     AllToAll 通信实现
│   │   ├── barrier/              #     同步屏障
│   │   ├── collective_comm_api.h
│   │   ├── collective_comm_base.h
│   │   └── collective_comm_context.h
│   └── aiv_compute/              #   AIV 向量计算接口
├── basic/                        # 基础数据结构与抽象
│   └── fragment_tensor/          #   FragmentTensor（多 fragment 统一抽象）
├── tiling/                       # tiling 算法
├── utils/                        # 通用工具与常量
├── tests/                        # 测试
│   └── st/                       #   系统级测试
├── docs/                         # 设计说明
└── README.md
```

---

## 模块说明

### kernel 算子层

通算融合算子实现，kernel 层完成通信与计算的流水编排、AIV-AIC 协同调度、同步等，可直接调用或作为参考。

| 子目录 | 说明 |
|:---|:---|
| `fusions` | 通信与 Matmul 融合的完整 kernel 实现（如 AllGather/AllToAll + 量化 Matmul）。 |
| `matmul` | 独立的 Matmul kernel，含 Grouped Matmul 与量化 Batch Matmul。 |

### block 接口层

可复用的单核 Matmul 计算 block 接口，按计算阶段组织，支撑 kernel 层组合构建融合算子。

| 子目录 | 说明 |
|:---|:---|
| `mmad` | 基于 [ops-tensor](https://gitcode.com/cann/ops-tensor) Blaze 库的量化 MMAD fragment，对接 FragmentTensor。 |
| `epilogue` | Matmul 后处理 block。 |
| `scheduler` | 多核/多 block 调度 block。 |

### core 核心层

核心通信与计算接口，为 block / kernel 层提供底层能力。

| 子目录 | 说明 |
|:---|:---|
| `aiv_comm` | AIV 核作为发起引擎的集合通信接口，基于 hcomm + UDMA 提供 AllGather / AllToAll / Barrier 能力，含 CRTP 基类与统一 API。 |
| `aiv_compute` | AIV 核向量计算接口（量化/反量化、类型转换、规约等），支撑通信前后的数据处理。 |

### basic 基础层

为上层 block / kernel 提供通用的基础数据结构与抽象能力。

| 子目录 | 说明 |
|:---|:---|
| `fragment_tensor` | 多个 GM fragment 的统一抽象，支持按轴拼装的 Slice / Copy / Scatter 操作，简化跨 fragment 数据访问。 |

### tiling

提供量化 Matmul 切分算法（含 MX baseline 与 SWAT 切分策略）及通信 `CommTilingData` 结构定义，同时供 host 侧切分与 kernel 侧通信使用。

### utils

通用工具集合，包括共享常量、通用数据结构与 HCCL channel 创建工具等。

---

## 与 MC2 算子的关系

`mc2/` 下的具体通算融合算子（如 `all_gather_matmul`、`matmul_all_reduce`、`matmul_all_to_all` 等）可通过引用本软件中的接口完成实现：

- 算子的 `op_host/op_tiling` 层可使用 `apace/tiling` 中的切分算法确定 tiling 参数。
- 算子的 `op_kernel` 层可直接调用 `apace/kernel` 实现，也可基于 `apace/block` + `apace/core` 接口组合构建融合 kernel。

```
mc2/<op>/op_host    ──┐
                      └──▶ apace/tiling       (tiling 接口, 提供切分算法)
mc2/<op>/op_kernel  ──┐
                      ├──▶ apace/kernel       (kernel 接口，可直接调用或参考)
                      ├──▶ apace/block        (计算 block 接口，组合构建)
                      └──▶ apace/core         (通信/计算核心接口)
```

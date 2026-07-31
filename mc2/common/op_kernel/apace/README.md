# APACE

## 项目定位

**APACE**（**A**scend **PA**rallel **C**ommunication-compute **E**ngine）是昇腾 NPU 平台通信和计算融合算子的架构底座，为各类通算融合算子提供可复用的代码模块，降低算子开发成本。

**核心价值**：

- 降低开发门槛：提供分层接口与参考实现，开发者通过接口组合即可构建融合算子，无需从零实现通信与计算的协同编排
- 提升性能：通信与计算深度耦合，支持流水线重叠，充分发挥硬件算力
- 模块化复用：提供可复用的代码模块，可直接调用或者作为参考实现

---

## 目录结构

```
apace/
├── kernel/               # 通算融合算子实现
├── block/
│   ├── blaze_ext/        #   Blaze 范式扩展
│   ├── aiv_comm/         #   AIV 通信接口
│   └── aiv_compute/      #   AIV 计算接口
├── basic/                # 基础数据结构与抽象
│   └── fragment_tensor/  #   FragmentTensor（多 fragment 统一抽象）
├── tiling/               # tiling 算法
├── utils/                # 通用工具与常量
├── tests/                # 测试
├── docs/                 # 设计说明
└── README.md
```

---

## 模块说明

### kernel 算子层

通算融合算子实现，kernel 层完成通信与计算的流水编排、AIV-AIC 协同调度、同步等，可直接调用或作为参考。

### block 接口层

可复用的单核通信计算接口，按不同实现范式组织目录结构。

| 子目录 | 说明 |
|:---|:---|
| `blaze_ext` | 对 [ops-tensor](https://gitcode.com/cann/ops-tensor) Blaze 库在通算融合场景下的拓展实现。 |
| `aiv_comm` | AIV 核作为发起引擎的通信接口，提供跨卡数据搬移能力。 |
| `aiv_compute` | AIV 核实现向量计算接口（量化/反量化、类型转换、规约等），支撑通信前后的数据处理。 |

### basic 基础层

为上层 block / kernel 提供通用的基础数据结构与抽象能力。

| 子目录 | 说明 |
|:---|:---|
| `fragment_tensor` | 多个 GM fragment 的统一抽象，支持按轴拼装的 Slice / Copy / Scatter 操作，简化跨 fragment 数据访问。 |

### tiling

提供 Matmul 和通信数据的切分算法。

### utils

通用工具集合，包括通用常量和数据结构等。

---

## 与 MC2 算子的关系

`mc2/` 下的具体通算融合算子（如 `all_gather_matmul`、`matmul_all_reduce`、`matmul_all_to_all` 等）可通过引用本软件中的接口完成实现：

- 算子的 `op_host/op_tiling` 层可使用 `apace/tiling` 中的切分算法确定 tiling 参数。
- 算子的 `op_kernel` 层可直接调用 `apace/kernel` 实现，也可基于 `apace/block` 接口组合构建融合 kernel。

```
mc2/<op>/op_host    ──┐
                      └──▶ apace/tiling       (tiling 接口, 提供切分算法)
mc2/<op>/op_kernel  ──┐
                      ├──▶ apace/kernel       (kernel 接口，可直接调用或参考)
                      └──▶ apace/block        (block 接口，组合构建)
```

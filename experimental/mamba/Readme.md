
MambaV2系列算子（Ascend CANN / cann-ops-transformer）

本目录包含针对Nemotron-H系列模型的Mamba v2定制算子，基于cann-ops-transformer框架，在华为Ascend AI（昇腾NPU）910B平台上实现高性能加速。

## 背景与原理

Transformer凭借Self-Attention机制在序列建模上取得统治地位，但其 $O(N^2)$ 的计算复杂度与随序列长度线性增长的KV Cache内存开销，在长序列场景下成为严重的效率瓶颈。Mamba正是为解决这一问题而提出：它基于状态空间模型（SSM），将历史信息压缩到固定大小的递推隐藏状态中，以 $O(N)$ 线性复杂度替代Attention的平方复杂度，同时通过选择性机制（Selective Scan）让参数随输入动态变化，获得与Attention相当的内容感知建模能力。Mamba v2进一步提出状态空间对偶性（SSD），将递推计算等价转化为结构化矩阵乘法，解决了Mamba v1因时序依赖无法并行训练的问题。

### 状态空间模型（SSM）

Mamba系列模型基于状态空间模型（State-Space Model, SSM）进行序列建模。SSM通过一个递推的隐藏状态 $h(t)$ 来压缩历史序列信息，其离散形式可表示为：

$$h(t) = \bar{A} \cdot h(t-1) + \bar{B} \cdot x(t), \quad y(t) = C \cdot h(t)$$

其中 $\bar{A}, \bar{B}$ 由输入相关的步长 $\Delta$ 离散化得到。相较于Transformer依赖KV Cache存储全部历史token，SSM将历史信息压缩到固定大小的状态向量中，在长序列场景下具有更优的计算效率（线性复杂度 $O(N)$）与内存效率。

### 选择性机制（Selective SSM）

Mamba v1引入了选择性机制（Selective Scan）：将参数 $B, C, \Delta$ 设为输入相关的函数，使模型能够根据输入内容动态决定信息的保留与遗忘。这一机制打破了传统LTI（线性时不变）系统的限制，赋予模型内容感知的序列建模能力，但纯递推形式难以并行化。

### 状态空间对偶性（SSD, Mamba v2）

Mamba v2的核心理论贡献是状态空间对偶性（State Space Duality, SSD）。SSD证明了选择性SSM的递推计算可以等价改写为一种结构化矩阵（半可分矩阵, semiseparable matrix）的乘法，从而在SSM与Attention之间建立对偶关系：

- **SSM视角**：线性递推，$O(N)$ 复杂度，适合长序列
- **Attention视角**：结构化矩阵乘法，可并行，适合硬件加速

基于这一对偶性，Mamba v2采用**分块（Chunked）计算策略**：

1. **Chunk内（Intra-chunk）**：将SSM递推展开为块内矩阵乘法，利用Cube/矩阵乘单元并行计算，类似Attention的并行计算模式
2. **Chunk间（Inter-chunk）**：通过递推状态传递将各chunk的最终状态传播到后续chunk，保持SSM的线性复杂度特性

这种设计同时获得了并行计算的高吞吐与SSM的线性复杂度优势。

### 因果卷积（Causal Conv1d）

在SSM之前，Mamba使用短卷积（kernel width=4的depthwise causal conv1d）对输入进行局部上下文建模，再经过SiLU激活，为SSM提供局部特征增强。vLLM的 `causal_conv1d_fn` 实现支持变长（varlen）与连续批处理（continuous batching）：

- 输入 `x` 为2D张量 `(dim, cu_seq_len)`，其中 `cu_seq_len` 为batch中所有序列拼接后的总token数
- `query_start_loc` 记录各序列的累积长度边界，用于在拼接张量中索引各序列
- `conv_states` 作为卷积状态缓存，通过 `cache_indices` 将序列映射到缓存槽位，支持连续批处理下的状态复用与更新
- `has_initial_state` 标记是否使用缓存中的状态作为初始状态

### vLLM Mamba算子体系

vLLM在 `vllm.model_executor.layers.mamba.ops` 下实现了完整的Mamba v2算子体系，本目录算子与之对应：

| 本目录算子 | vLLM对应模块 | 功能 |
|-----------|-------------|------|
| mamba2_causal_conv1d | causal_conv1d | 因果卷积 + SiLU激活 |
| mamba2_chunk_cumsum | ssd_combined (cumsum部分) | chunk内累积求和，用于状态递推 |
| mamba2_chunk_state | ssd_chunk_state | chunk内离散状态更新 |
| mamba2_chunk_state_passing | ssd_state_passing | 跨chunk状态传递与衰减 |
| mamba2_chunk_scan | ssd_combined (scan部分) | selective scan扫描，结合状态与门控 |
| mamba2_rmsnormgated | layernorm_gated / gdn | RMSNorm + Gate融合归一化 |

## Nemotron-H网络中的MambaV2整层计算流

<img src="https://raw.gitcode.com/user-images/assets/7673863/5163260c-fe1e-4a06-90c6-5355ba23ea90/image.png" height="900">

## MambaV2 chunk scan combined计算流和融合算子设计

![image.png](https://raw.gitcode.com/user-images/assets/7673863/b98b23ee-bb4b-42b5-8270-92828d17c2d2/image.png 'image.png')

## 目录结构

```
experimental/
└── mamba/
    ├── mamba2_causal_conv1d/        # 因果卷积（Causal Conv1d + SiLU）
    ├── mamba2_chunk_cumsum/         # chunk内累积求和，用于streaming状态累积
    ├── mamba2_chunk_state/          # chunk内离散状态更新
    ├── mamba2_chunk_state_passing/  # 跨chunk状态传递
    ├── mamba2_chunk_scan/           # selective scan扫描机制
    ├── mamba2_rmsnormgated/         # RMSNorm + Gate融合算子
    ├── common/                      # 公共头文件（paramutils, tensorutils）
    └── utils/                       # 公共工具（精度比对、性能profiling）
```

其中 mamba2_chunk_xxx 四个算子为 Prefill 过程中 Chunk 计算的核心实现模块。

每个算子子目录包含：
- 算子实现（op_kernel/）
- torch封装（torch_interface.cpp）
- 精度和性能测试脚本（tests/）
- 算子介绍说明文档（README.md）

## 编译与使用

1. 进入编译目录

```
cd experimental/npu_ops_transformer_ext
```

2. 编译

```
python3 -m build --wheel -n
```

3. 安装

```
cd dist
pip3 install *.whl --force-reinstall --no-deps
```

4. 测试，以 mamba2_causal_conv1d 为例

```
cd experimental/mamba/mamba2_causal_conv1d/tests
python3 test_causal_conv1d.py
```

## 特性说明

1. 当前版本算子已支持FP32 / FP16输入输出精度；
2. 所有 mamba2_chunk_xxx 系列算子均支持BSND数据布局，其中S维度需在调用前pad至chunk_size的整数倍；
3. 当前版本仅支持固定 chunk_size = 256；
4. 已通过PyTorch参考实现的精度比对验证（测试脚本见各算子的tests目录）。

## 单融合算子性能加速比

（每个算子 tests 下测试脚本在910B3的profile结果）

<img src="https://raw.gitcode.com/user-images/assets/7673863/4d4b226f-cce2-4c93-bb01-fdaade6fff7e/image.png" height="200">

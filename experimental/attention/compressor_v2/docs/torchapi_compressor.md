# compressor

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/compressor_v2)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：compressor 是 CompressorV2 压缩算子的 PyTorch（PTA）接口，通过 `cann_ops_transformer.ops.ds41` 子模块对外提供。用于将每 `cmp_ratio` 个 token 的 KV cache 压缩成一个，同时将 `kv_state`/`score_state` 原位更新到 `state_cache`（循环 buffer）。主要计算过程为：
    1. 将输入 $X$ 与 $W^{KV}$ 做 Matmul 运算得到 $kv\_state$，将输入 $X$ 与 $W^{Gate}$ 做 Matmul 运算得到 $score\_state$，$kv\_state$ 与 $score\_state$ 根据输入的 start_pos 及 cu_seqlens 写入 state_cache（循环 buffer，写本次序列末尾 min(seqused, block_size) 个 token）。
    2. 对 $score\_state$ 进行 softmax 运算，将 softmax 结果与 $kv\_state$ 做 Mul 计算，后按压缩轴 ReduceSum 得到压缩结果 $cmp\_kv$。

- 计算公式：

    1. 计算矩阵乘法：

        $$
        \left[kv\_state, score\_state\right] = \left[X @ W^{KV}, X @ W^{Gate}\right]
        $$

    2. 计算分组Softmax：

        $$
        S_i^\prime = softmax(score\_state_i),~i=1,2,\cdots, \frac{s}{cmp\_ratio}
        $$

    3. 计算Hadamard乘积：

        $$
        S_H = S_i^\prime \odot kv\_state
        $$

    4. 沿着压缩轴分组求和：

        $$
        C_{i}^{\text{Comp}} = \left[1\right]_{1\times cmp\_ratio} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{cmp\_ratio}
        $$

## 函数原型

```python
from cann_ops_transformer.ops.ds41 import compressor

compressor(
    x,
    wkv,
    wgate,
    state_cache,
    cmp_ratio=4,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| x | Tensor | 必选 | 原始不经压缩的数据，对应公式中的 $X$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [B,S,H]、[T,H] |
| wkv | Tensor | 必选 | kv压缩权重，对应公式中的 $W^{KV}$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [D,H] |
| wgate | Tensor | 必选 | gate压缩权重，对应公式中的 $W^{Gate}$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [D,H] |
| state_cache | Tensor | 必选 | kv_state和score_state的历史数据，对应公式中的 $\left[kv\_state, score\_state\right]$。支持0轴非连续，数据格式支持ND。计算后 kv_state 和 score_state 会原位更新到此 Tensor。 | float32 | [block_num, block_size, 2\*D]，要求block_num>0 |
| cmp_ratio | int | 必选 | 数据压缩率。取值范围为[2, 128]内的整数。 | - | - |
| state_block_table | Tensor | 可选 | state_cache存储使用的block映射表。不支持非连续，数据格式支持ND。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作。 | int32 | [B] |
| cu_seqlens | Tensor | 可选 | 不同Batch上的有效token数。不支持非连续，数据格式支持ND。<br>当x的shape为[B,S,H]时，参数必须为空。<br>当x的shape为[T,H]时，输入shape必须为[B+1,]，该参数为前缀和数组，后一个元素≥前一个元素，第一位必须为0，最后一位必须为T。 | int32 | [B+1,] |
| seqused | Tensor | 可选 | 不同Batch中实际参与压缩的token数。不支持非连续，数据格式支持ND。<br>指定为None时，数值等于每个Batch上的Sequence Length。<br>[B,S,H]场景：0 ≤ seqused[n] ≤ S<br>[T,H]场景：0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]。 | int32 | [B,] |
| start_pos | Tensor | 可选 | 计算起始位置。不支持非连续，数据格式支持ND，输入为None时从0开始计算。 | int32 | [B,] |

> 说明：CompressorV2 固定 `coff=1`、`cache_mode=2`（循环buffer），无 ape 输入，仅输出 `cmp_kv`。

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| cmp_kv | Tensor | 必选 | 压缩后的数据。不支持非连续，数据格式支持ND；<br>当x的shape为[B,S,H]时，输出拼接：(\<batch0\>compressed_tokens+pad0) + (\<batch1\>compressed_tokens+pad1) + ... + (\<batchN\>compressed_tokens+padN)；<br>当x的shape为[T,H]时，输出拼接：\<batch0\>compressed_tokens + \<batch1\>compressed_tokens + ... + \<batchN\>compressed_tokens + pad。 | bfloat16、float16 | x=[B,S,H]：[B,ceil(S/cmp_ratio),D]<br>x=[T,H]：[min(T,T//cmp_ratio+B),D] |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式(aclgraph)调用。
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 该接口支持B、S泛化，且存在如下场景限制：
  - 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下state_cache不做更新，输出cmp_kv为空tensor。
  - state_block_table元素取值范围为[0, block_num)，block_num为state_cache第0维大小。元素值直接用作state_cache的block索引，越界会导致内存非法访问。元素值为0时：cache_mode=2（循环buffer）下读写操作均不跳过。算子不做重复校验，需由调用方保证元素值唯一性：所有元素值须全局唯一（0为有效物理块号）。重复会导致多个逻辑块/batch写同一物理块区域，造成state_cache数据踩踏覆盖。
- 支持D为128/512。
- 支持H为1K~10K，512对齐。
- 支持cmp_ratio为2/4/8/16/32/64/128。
- 支持block_size为1~1024。

## 确定性计算

- 默认确定性实现，相同输入多次调用结果一致。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    from cann_ops_transformer.ops.ds41 import compressor

    # 参数设置
    B = 1
    S = 128
    H = 4096
    D = 512
    cmp_ratio = 4
    block_size = 128

    block_num = B
    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((D, H), dtype=torch.bfloat16).npu()
    state_cache = torch.zeros((block_num, block_size, 2 * D), dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    state_block_table = torch.arange(B, dtype=torch.int32).npu()

    cmp_kv = compressor(
        x, wkv, wgate, state_cache,
        cmp_ratio=cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=None,
        seqused=None,
        start_pos=start_pos)
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```

- TorchAir图模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torchair
    from cann_ops_transformer.ops.ds41 import compressor
    from torchair.configs.compiler_config import CompilerConfig

    # 参数设置
    B = 1
    S = 128
    H = 4096
    D = 512
    cmp_ratio = 4
    block_size = 128

    block_num = B
    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((D, H), dtype=torch.bfloat16).npu()
    state_cache = torch.zeros((block_num, block_size, 2 * D), dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    state_block_table = torch.arange(B, dtype=torch.int32).npu()

    class CompressorNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, wkv, wgate, state_cache, cmp_ratio, block_table, start_pos):
            return compressor(
                x, wkv, wgate, state_cache,
                cmp_ratio=cmp_ratio,
                state_block_table=block_table,
                cu_seqlens=None,
                seqused=None,
                start_pos=start_pos)

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(CompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(x, wkv, wgate, state_cache, cmp_ratio, state_block_table, start_pos)
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```

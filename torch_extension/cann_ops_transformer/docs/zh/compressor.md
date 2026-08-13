# compressor

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
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

- **接口功能**：Compressor是推理场景下SMLA和QLI的前处理算子，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，Compressor可以有效地减少计算开销。主要计算过程为：
    1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。

- **计算公式**：

    1. 计算矩阵乘法：

        $$
        C4A：\left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right], \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right];
        $$

        $$
        C128A：\left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]
        $$

    2. 计算分组加法：

        $$
        C4A：score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A：score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
        $$

    3. 计算分组Softmax：

        $$
        C4A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
        $$

    4. 计算Hadamard乘积：

        $$
        C4A：(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A：S_H = S_i^\prime \odot kv\_state;
        $$

    5. 沿着压缩轴分组求和：

        $$
        C4A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
        $$

## 函数原型

```python
cann_ops_transformer.compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| x | Tensor | 必选 | 原始不经压缩的数据，对应公式中的 $X$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [B,S,H]、[T,H] |
| wkv | Tensor | 必选 | kv压缩权重，对应公式中的 $W^{KV}$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [coff\*D,H] |
| wgate | Tensor | 必选 | gate压缩权重，对应公式中的 $W^{Gate}$。不支持非连续，数据格式支持ND。 | bfloat16、float16 | [coff\*D,H] |
| state_cache | Tensor | 必选 | kv_state和score_state的历史数据，对应公式中的 $\left[kv\_state, score\_state\right]$。不支持非连续，数据格式支持ND。 | float32 | [block_num, block_size, 2\*coff\*D]，要求block_num>0 |
| ape | Tensor | 必选 | positional biases，对应公式中的 $Ape$。不支持非连续，数据格式支持ND。 | float32 | [cmp_ratio,coff\*D] |
| cmp_ratio | int | 必选 | 数据压缩率。取值范围为[2, 128]内的整数。 | - | - |
| state_block_table | Tensor | 可选 | state_cache存储使用的block映射表。不支持非连续，数据格式支持ND。 | int32 | cache_mode=1时，shape为[B,ceil(Smax/block_size)]，Smax为每个Batch中最大的Sequence Length，当x的shape为[B,S,H]时，Smax=max(start_pos)+S。当x的shape为[T,H]时，Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n])。cache_mode=2时，shape为[B]。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作 |
| cu_seqlens | Tensor | 可选 | 不同Batch上的有效token数。不支持非连续，数据格式支持ND。<br>当x的shape为[B,S,H]时，参数必须为空。<br>当x的shape为[T,H]时，输入shape必须为[B+1,]，该参数为前缀和数组，后一个元素≥前一个元素，第一位必须为0。 | int32 | [B+1,] |
| seqused | Tensor | 可选 | 不同Batch中实际参与压缩的token数。不支持非连续，数据格式支持ND。<br>指定为None时，数值等于每个Batch上的Sequence Length。<br>[B,S,H]场景：0 ≤ seqused[n] ≤ S<br>[T,H]场景：0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]。 | int32 | [B,] |
| start_pos | Tensor | 可选 | 计算起始位置。不支持非连续，数据格式支持ND，输入为None时从0开始计算 | int32 | [B,] |
| coff | int | 可选 | 表示是否进行overlap数据重排，默认值为1。<br>仅支持1/2：<br>coff=1：无需进行overlap数据重排。<br>coff=2：需要进行overlap数据重排。 | int | - |
| cache_mode | int | 可选 | state_cache的存储模式，默认值为1。<br>1：连续buffer。<br>2：循环buffer。 | int | - |

<!-- npu="A3" id7 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：cache_mode不支持输入2，且不支持0轴非连续；cmp_ratio仅支持2/4/8/16/32/64/128。
<!-- end id7 -->

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| cmp_kv | Tensor | 必选 | 压缩后的数据。不支持非连续，数据格式支持ND；<br>当x的shape为[B,S,H]时，输出拼接：(\<batch0\>compressed_tokens+pad0) +  (\<batch1\>compressed_tokens+pad1) + ... +  (\<batchN\>compressed_tokens+padN)；<br>当x的shape为[T,H]时，输出拼接：\<batch0\>compressed_tokens + \<batch1\>compressed_tokens + ... + \<batchN\>compressed_tokens + pad。 | bfloat16、float16 | x=[B,S,H]：[B,ceil(S/cmp_ratio),D]<br>x=[T,H]：[min(T,T//cmp_ratio+B),D] |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式(aclgraph)调用。
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 该接口支持B、S泛化，且存在如下场景限制：
  - 只支持B、S为0。
  - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 100px">
    </colgroup><thead>
    <tr>
    <th>B</th>
    <th>S</th>
    <th>H</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>100</td>
    <td>65525</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>25</td>
    <td>261120</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>100</td>
    <td>131072</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>100</td>
    <td>261120</td>
    <td>4096</td>
    </tr>
    </tbody>
    </table>
    </div>
- 支持D为128/512。
- 支持H为1K~10K，512对齐。
- 支持block_size为1~1024。
- 支持如下三种典型组合场景：
  - C4A: D=512, coff=2, cmp_ratio=4;
  - C4Li: D=128, coff=2, cmp_ratio=4;
  - C128A: D=512, coff=1, cmp_ratio=128。

## 确定性计算

- 默认支持确定性计算。
<!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term>：batch一致性：通过torch_npu.npu.set_deterministic_level()设置确定性级别为3开启batch一致性，开启后可以满足计算结果和所在批次大小和所在批次位置无关。
<!-- end id8 -->

## 调用示例

> **说明：**<br>
> - 以下示例以C128A场景为例（B=1、S=128、H=4096、D=512、coff=1、cmp_ratio=128），更多参数组合请参考[约束说明](#约束说明)。

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    import cann_ops_transformer

    # 参数设置
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1  # 1: no overlap  2: overlap
    cmp_ratio = 128
    cache_mode = 1
    block_size = 128

    # block_table构造：cache_mode=1时shape为[B, ceil(Smax/block_size)]
    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    # 构造输入
    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()

    # 调用compressor执行压缩计算
    cmp_kv = cann_ops_transformer.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio=cmp_ratio,
        state_block_table=block_table,
        cu_seqlens=None,
        seqused=None,
        start_pos=start_pos,
        coff=coff,
        cache_mode=cache_mode
    )
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```

- TorchAir图模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torchair
    import cann_ops_transformer
    from torchair.configs.compiler_config import CompilerConfig

    # 参数设置
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1
    cmp_ratio = 128
    cache_mode = 1
    block_size = 128

    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()


    class CompressorNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, wkv, wgate, state_cache, ape, block_table, start_pos):
            return torch.ops.cann_ops_transformer.compressor(
                x, wkv, wgate, state_cache, ape,
                cmp_ratio=cmp_ratio,
                state_block_table=block_table,
                cu_seqlens=None,
                seqused=None,
                start_pos=start_pos,
                coff=coff,
                cache_mode=cache_mode
            )


    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(CompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(x, wkv, wgate, state_cache, ape, block_table, start_pos)
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```

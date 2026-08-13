# quant_compressor

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

- **接口功能**：QuantCompressor是推理场景下SMLA和QLI的前处理算子，是[compressor](./compressor.md)的量化版本。用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，QuantCompressor可以有效地减少计算开销。与compressor的区别在于：输入$x$、$W^{KV}$、$W^{Gate}$为HIFLOAT8量化数据，直接以HIFLOAT8参与Matmul运算（硬件原生支持），再对Matmul输出的FLOAT32结果乘以合并后的缩放因子进行反量化，从而降低显存占用与搬运开销。主要计算过程为：
    1. Matmul与反量化：将HIFLOAT8量化的输入$X$与$W^{KV}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wkv\_descale$完成反量化得到$kv\_state$；将$X$与$W^{Gate}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wgate\_descale$完成反量化得到$score\_state$。其中x_descale为per-tensor缩放（单个标量），wkv_descale与wgate_descale为per-channel缩放（通道数为coff\*D），合并后仍为per-channel缩放。$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$按压缩比分组并与$Ape$相加，然后进行softmax运算，将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。

- **计算公式**：

    1. 计算矩阵乘法与反量化（quant_mode=1，A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL）：

    使用HIFLOAT8输入进行Matmul，输出FLOAT32结果后乘以合并缩放因子（x_descale与wkv/wgate_descale的乘积）完成反量化：

    $$
    C4A：\left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
    $$

    $$
    C128A：\left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
    $$

    其中x_descale为per-tensor标量（shape为[1,]），wkv_descale、wgate_descale为per-channel向量（shape为[coff\*D,]，沿权重输出通道维度缩放），两者相乘后合并为per-channel缩放因子。

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
cann_ops_transformer.quant_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    quant_mode,
    cmp_ratio,
    *,
    x_descale=None,
    wkv_descale=None,
    wgate_descale=None,
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
| x | Tensor | 必选 | 原始不经压缩的数据，对应公式中的 $X$，HIFLOAT8量化输入。不支持非连续，数据格式支持ND。 | uint8（HIFLOAT8） | [B,S,H]、[T,H] |
| wkv | Tensor | 必选 | kv压缩权重，对应公式中的 $W^{KV}$，HIFLOAT8量化输入。不支持非连续，数据格式支持ND。 | uint8（HIFLOAT8） | [coff\*D,H] |
| wgate | Tensor | 必选 | gate压缩权重，对应公式中的 $W^{Gate}$，HIFLOAT8量化输入。不支持非连续，数据格式支持ND。 | uint8（HIFLOAT8） | [coff\*D,H] |
| state_cache | Tensor | 必选 | kv_state和score_state的历史数据，对应公式中的 $\left[kv\_state, score\_state\right]$。不支持非连续，数据格式支持ND。 | float32 | [block_num, block_size, 2\*coff\*D]，要求block_num>0 |
| ape | Tensor | 必选 | positional biases，对应公式中的 $Ape$。不支持非连续，数据格式支持ND。 | float32 | [cmp_ratio,coff\*D] |
| quant_mode | int | 必选 | 量化模式。取值范围为[1]，1表示A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL（HIFLOAT8输入，x按per-tensor缩放、wkv/wgate按per-channel缩放反量化）。 | - | - |
| cmp_ratio | int | 必选 | 数据压缩率。取值范围为[2, 128]内的整数。 | - | - |
| x_descale | Tensor | 可选 | x的反量化缩放因子，per-tensor缩放。quant_mode=1时必选。不支持非连续，数据格式支持ND。 | float32 | [1,] |
| wkv_descale | Tensor | 可选 | wkv的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。不支持非连续，数据格式支持ND。 | float32 | [coff\*D,] |
| wgate_descale | Tensor | 可选 | wgate的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。不支持非连续，数据格式支持ND。 | float32 | [coff\*D,] |
| state_block_table | Tensor | 可选 | state_cache存储使用的block映射表。不支持非连续，数据格式支持ND。 | int32 | cache_mode=1时，shape为[B,ceil(Smax/block_size)]，Smax为每个Batch中最大的Sequence Length，当x的shape为[B,S,H]时，Smax=max(start_pos)+S。当x的shape为[T,H]时，Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n])。cache_mode=2时，shape为[B]。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作 |
| cu_seqlens | Tensor | 可选 | 不同Batch上的有效token数。不支持非连续，数据格式支持ND。<br>当x的shape为[B,S,H]时，参数必须为空。<br>当x的shape为[T,H]时，输入shape必须为[B+1,]，该参数为前缀和数组，后一个元素≥前一个元素，第一位必须为0。 | int32 | [B+1,] |
| seqused | Tensor | 可选 | 不同Batch中实际参与压缩的token数。不支持非连续，数据格式支持ND。<br>指定为None时，数值等于每个Batch上的Sequence Length。<br>[B,S,H]场景：0 ≤ seqused[n] ≤ S<br>[T,H]场景：0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]。 | int32 | [B,] |
| start_pos | Tensor | 可选 | 计算起始位置。不支持非连续，数据格式支持ND，输入为None时从0开始计算 | int32 | [B,] |
| coff | int | 可选 | 默认值1，仅支持1/2。<br>coff=1：无需进行overlap数据重排<br>coff=2：需要进行overlap数据重排。 | int32 | - |
| cache_mode | int | 可选 | state_cache的存储模式。<br>1：连续buffer<br>2：循环buffer<br>默认值1。 | int32 | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| cmp_kv | Tensor | 必选 | 压缩后的数据。不支持非连续，数据格式支持ND；<br>当x的shape为[B,S,H]时，输出拼接：(\<batch0\>compressed_tokens+pad0) +  (\<batch1\>compressed_tokens+pad1) + ... +  (\<batchN\>compressed_tokens+padN)；<br>当x的shape为[T,H]时，输出拼接：\<batch0\>compressed_tokens + \<batch1\>compressed_tokens + ... + \<batchN\>compressed_tokens + pad。 | bfloat16 | x=[B,S,H]：[B,ceil(S/cmp_ratio),D]<br>x=[T,H]：[min(T,T//cmp_ratio+B),D] |

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
- quant_mode取值为1（A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL），此时x_descale、wkv_descale、wgate_descale为必选输入。其中x_descale shape为[1,]（per-tensor缩放），wkv_descale、wgate_descale shape为[coff\*D,]（per-channel缩放）。
- 支持如下三种典型组合场景：
  - C4A: D=512, coff=2, cmp_ratio=4;
  - C4Li: D=128, coff=2, cmp_ratio=4;
  - C128A: D=512, coff=1, cmp_ratio=128。

## 确定性计算

- 默认支持确定性计算。
- batch一致性：通过torch_npu.npu.set_deterministic_level()设置确定性级别为3开启batch一致性，开启后可以满足计算结果和所在批次大小和所在批次位置无关。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    from cann_ops_transformer import quant_compressor
    import torch.nn as nn
    import math

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.uint8  # HIFLOAT8 在 torch 中以 uint8 承载
    hidden_size = 4096
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    cache_mode = 1
    head_dim = 512
    cu_seqlens = [0, 1]
    quant_mode = 1
    # -------------
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B # (B,)
    start_p=8191
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1
    save_state_seqlens = None
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    else:
        start_pos = torch.full((B,), start_p, dtype=torch.int32)

    if bs_combine_flag:
        if cu_seqlens is None:
            T = B * S
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((B+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(B):
            if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S_max = max(start_pos) + S
    ### ======================== gen input data start =============================
    # page state
    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        print(f"max_block_num_per_batch: {max_block_num_per_batch}")
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
            # 需要读取state的范围
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(cur_start_block_id, cur_end_block_id + 1):
                block_table[i][j] = next_block_id
                next_block_id = next_block_id + 1
            # 需要写入state的范围
            end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            if save_state_seqlens is not None:
                next_start = start_pos[i] + end_pos - save_state_seqlens[i]
                next_end = start_pos[i] + end_pos
            else:
                next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
                next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
                if (start_pos[i] + end_pos) % cmp_ratio == 0:
                    next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(next_start_block_id, next_end_block_id + 1):
                if block_table[i][j] == 0:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1

        if B==0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        block_size = (2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1)
        if B==0:
            kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (B, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (B, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type).npu()
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()
    # descale 缩放因子
    x_descale = torch.tensor(np.random.uniform(0.1, 1.0, (1,))).to(torch.float32).npu()
    wkv_descale = torch.tensor(np.random.uniform(0.1, 1.0, (coff * head_dim,))).to(torch.float32).npu()
    wgate_descale = torch.tensor(np.random.uniform(0.1, 1.0, (coff * head_dim,))).to(torch.float32).npu()
    if cache_mode == 1:  # 连续buffer
        state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2*kv_state.shape[2]))
        state_cache = state_cache.npu()
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
    else:
        layer_pad = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_pad-1)
        print(f"layer_pad: {layer_pad}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros((kv_state.shape[0],kv_state.shape[1]*kv_state.shape[2]*2+layer_pad))
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
        state_cache = state_cache_pad[:, layer_start_idx : layer_start_idx + kv_state.shape[1]*kv_state.shape[2]*2].view(-1, kv_state.shape[1], kv_state.shape[2]*2)
        state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
        print(f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}")

    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    cmp_kv = (
        quant_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            quant_mode = quant_mode,
            cmp_ratio = cmp_ratio,
            x_descale = x_descale,
            wkv_descale = wkv_descale,
            wgate_descale = wgate_descale,
            state_block_table = block_table,
            cu_seqlens = cu_seqlens,
            seqused = seqused,
            start_pos = start_pos,
            coff = coff,
            cache_mode = cache_mode
        )
    )
    ```

- TorchAir图模式调用：

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair
    from cann_ops_transformer import quant_compressor
    import math

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.uint8  # HIFLOAT8 在 torch 中以 uint8 承载
    hidden_size = 4096
    coff = 1 # 1:no overlap 2:overlap
    cmp_ratio = 128
    cache_mode = 1
    head_dim = 512
    cu_seqlens = [0, 1]
    quant_mode = 1
    # -------------
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B # (B,)
    start_p=8191
    seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

    # BS是否合轴
    bs_combine_flag = True
    update_flag = 1
    save_state_seqlens = None
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32)
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32)
    else:
        start_pos = torch.full((B,), start_p, dtype=torch.int32)

    if bs_combine_flag:
        if cu_seqlens is None:
            T = B * S
            if T !=0:
                cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
            else:
                cu_seqlens = torch.zeros((B+1), dtype=torch.int32)
        else:
            cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
        for i in range(B):
            if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
                S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i]
    else:
        cu_seqlens = None
        S_max = max(start_pos) + S
    ### ======================== gen input data start =============================
    # page state
    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        print(f"max_block_num_per_batch: {max_block_num_per_batch}")
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
            # 需要读取state的范围
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(cur_start_block_id, cur_end_block_id + 1):
                block_table[i][j] = next_block_id
                next_block_id = next_block_id + 1
            # 需要写入state的范围
            end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            if save_state_seqlens is not None:
                next_start = start_pos[i] + end_pos - save_state_seqlens[i]
                next_end = start_pos[i] + end_pos
            else:
                next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
                next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
                if (start_pos[i] + end_pos) % cmp_ratio == 0:
                    next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(next_start_block_id, next_end_block_id + 1):
                if block_table[i][j] == 0:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1

        if B==0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        block_size = (2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1)
        if B==0:
            kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(kv_state_datarange[0], kv_state_datarange[1], (B, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(score_state_datarange[0], score_state_datarange[1], (B, block_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type).npu()
    wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()
    # descale 缩放因子
    x_descale = torch.tensor(np.random.uniform(0.1, 1.0, (1,))).to(torch.float32).npu()
    wkv_descale = torch.tensor(np.random.uniform(0.1, 1.0, (coff * head_dim,))).to(torch.float32).npu()
    wgate_descale = torch.tensor(np.random.uniform(0.1, 1.0, (coff * head_dim,))).to(torch.float32).npu()
    if cache_mode == 1:  # 连续buffer
        state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2*kv_state.shape[2]))
        state_cache = state_cache.npu()
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
    else:
        layer_pad = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_pad-1)
        print(f"layer_pad: {layer_pad}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros((kv_state.shape[0],kv_state.shape[1]*kv_state.shape[2]*2+layer_pad))
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
        state_cache = state_cache_pad[:, layer_start_idx : layer_start_idx + kv_state.shape[1]*kv_state.shape[2]*2].view(-1, kv_state.shape[1], kv_state.shape[2]*2)
        state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
        print(f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}")

    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    class QuantCompressorNetwork(nn.Module):
        def __init__(self):
            super(QuantCompressorNetwork, self).__init__()

        def forward(self, x, wkv, wgate, state_cache, ape, quant_mode, cmp_ratio, x_descale=None, wkv_descale=None,
                    wgate_descale=None, state_block_table=None, cu_seqlens=None,
                    seqused=None, start_pos=None, coff=1, cache_mode=1):
            cmp_kv = (
                torch.ops.cann_ops_transformer.quant_compressor(
                    x,
                    wkv,
                    wgate,
                    state_cache,
                    ape,
                    quant_mode = quant_mode,
                    cmp_ratio = cmp_ratio,
                    x_descale = x_descale,
                    wkv_descale = wkv_descale,
                    wgate_descale = wgate_descale,
                    state_block_table = state_block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    coff = coff,
                    cache_mode = cache_mode
                )
            )
            return cmp_kv

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(QuantCompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(
                    x,
                    wkv,
                    wgate,
                    state_cache,
                    ape,
                    quant_mode = quant_mode,
                    cmp_ratio = cmp_ratio,
                    x_descale = x_descale,
                    wkv_descale = wkv_descale,
                    wgate_descale = wgate_descale,
                    state_block_table = block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    coff = coff,
                    cache_mode = cache_mode)
    ```

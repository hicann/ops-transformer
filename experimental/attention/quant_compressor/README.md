# QuantCompressor

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantCompressor是推理场景下SAS和QLI的前处理算子，是Compressor算子的量化版本。该算子将每$cmp\_ratio$个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，QuantCompressor可以有效地减少计算开销。与Compressor算子相比，QuantCompressor使用HIFLOAT8量化格式接收输入数据x、wkv、wgate，并通过反量化缩放系数（descale）在计算过程中将量化结果恢复为FP32精度。

- 计算公式：

    压缩阶段：
    1. 计算HIFLOAT8矩阵乘法（反量化）：
        - C4A: $\left[kv\_state^a, score\_state^a\right] = \text{Dequant}(X_{hifp8} @ W^{aKV}_{hifp8}) \cdot \text{Descale}, \left[kv\_state^b, score\_state^b\right] = \text{Dequant}(X_{hifp8} @ W^{bKV}_{hifp8}) \cdot \text{Descale};$
        - C4Li: $\left[kv\_state^a, score\_state^a\right] = \text{Dequant}(X_{hifp8} @ W^{aKV}_{hifp8}) \cdot \text{Descale}, \left[kv\_state^b, score\_state^b\right] = \text{Dequant}(X_{hifp8} @ W^{bGate}_{hifp8}) \cdot \text{Descale};$
        - C128A: $\left[kv\_state, score\_state\right] = \text{Dequant}(X_{hifp8} @ W^{KV}_{hifp8}) \cdot \text{Descale}$
        其中 $\text{Dequant}(\text{Matmul}_{\text{HIFP8}}) = \text{Matmul}_{\text{HIFP8}} \cdot (x\_descale \otimes w\_descale)$
    2. 计算分组加法：
        - C4A: $score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};$
    3. 计算分组Softmax：
        - C4A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};$
    4. 计算Hadamard乘积：
        - C4A: $(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ; kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $S_H = S_i^\prime \odot kv\_state;$
    5. 沿着压缩轴分组求和：
        - C4A: $C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};$
        - C128A: $C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};$

- 主要计算过程为：
    1. 将HIFLOAT8输入$X_{hifp8}$与$W^{KV}_{hifp8}$做Matmul运算（通过反量化缩放系数$x\_descale \otimes wkv\_descale$将结果反量化为FP32）得到$kv\_state$，将HIFLOAT8输入$X_{hifp8}$与$W^{Gate}_{hifp8}$做Matmul运算（通过反量化缩放系数$x\_descale \otimes wgate\_descale$将结果反量化为FP32）得到$score\_state$，并根据输入的start\_pos及cu\_seqlens完成state\_cache更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算，得到$cmp\_kv$结果输出。
    4. 更新后的state\_cache作为第二个输出返回。

## 参数说明

| 参数名                      | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| x | 输入 | 公式中的$X_{hifp8}$，表示原始不经压缩的数据。 | HIFLOAT8 | ND         |
| wkv | 输入 | 公式中的$W^{KV}_{hifp8}$，表示kv压缩权重。 | HIFLOAT8 | ND |
| wgate | 输入 | 公式中的$W^{Gate}_{hifp8}$，表示gate压缩权重。 | HIFLOAT8 | ND |
| state\_cache | 输入/输出 | 公式中的$\left[kv\_state, score\_state\right]$, 表示kv\_state和score\_state的历史数据。作为输入读取历史数据，作为输出写回更新后的数据。支持非连续输入（通过state\_cache\_stride\_dim0属性指定stride）。 | FLOAT32     | ND         |
| ape | 输入 | 公式中的$Ape$，表示positional biases。 | FLOAT32       | ND         |
| x\_descale | 可选输入 | 表示输入x的反量化缩放系数。当quant\_mode=1时为必选输入。shape为[1]。 | FLOAT32 | ND |
| wkv\_descale | 可选输入 | 表示输入wkv的反量化缩放系数。当quant\_mode=1时为必选输入。shape为[coff\*D]。 | FLOAT32 | ND |
| wgate\_descale | 可选输入 | 表示输入wgate的反量化缩放系数。当quant\_mode=1时为必选输入。shape为[coff\*D]。 | FLOAT32 | ND |
| quant\_mode | 属性 | 表示量化模式。当前仅支持1，代表HIFP8量化模式。 | INT32       | -         |
| cmp\_ratio | 属性 | 用于稀疏计算，表示数据压缩率。 | INT32          | -         |
| state\_block\_table | 可选输入 | 表示state\_cache存储使用的block映射表。 | INT32 | ND         |
| cu\_seqlens | 可选输入 | 表示不同Batch中的有效token数。TH布局下为必选输入，BSH布局下必须为空。 | INT32          | ND         |
| seqused | 可选输入 | 表示不同Batch中实际参与压缩的token数，如果指定为None时，表示和每个Batch上的Sequence Length长度相同。 | INT32          | ND         |
| start\_pos | 可选输入 | 表示计算起始位置。 | INT32          | ND         |
| coff | 可选属性 | 默认值1，支持1/2。当coff=1时，无需进行overlap数据重排。当coff=2时，需要进行overlap数据重排。 | INT32          | -         |
| cache\_mode | 可选属性 | 表示state\_cache的存储模式，1表示连续buffer，2表示循环buffer。默认值1。**目前A3暂不支持输入2**。 | INT32          | -         |
| state\_cache\_stride\_dim0 | 可选属性 | 表示state\_cache第0维的stride，用于支持非连续的state\_cache输入。默认值0。 | INT32          | -         |
| cmp\_kv | 输出 | 表示压缩后的数据。 | BFLOAT16         | ND          |
| state\_cache | 输出 | 更新后的state\_cache数据（与输入state\_cache共享存储）。 | FLOAT32         | ND          |

## 约束说明
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff\* D,H]
    - wgate支持输入shape[coff\* D,H]
    - state\_cache支持输入shape[block\_num,block\_size,2\* coff\* D]，要求block\_num>0。支持非连续输入（通过state\_cache\_stride\_dim0属性指定stride）。
    - ape支持输入shape[cmp\_ratio,coff\* D]
    - x\_descale支持输入shape[1]
    - wkv\_descale支持输入shape[coff\* D]
    - wgate\_descale支持输入shape[coff\* D]
    - start\_pos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cu\_seqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须位0。**TH布局下为必选输入**。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
        - state\_block\_table支持输入shape[B,ceil(Smax/block\_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n])。
        - cmp\_kv，输出shape为[min(T,T//cmp\_ratio+B),D]：<batch0>compressed\_tokens + <batch1>compressed\_tokens + ... + <batchN>compressed\_tokens + pad。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cu\_seqlens，**参数必须为空**。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - state\_block\_table支持输入shape[B,ceil(Smax/block\_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+S。
        - cmp\_kv，输出shape为[B,ceil(S/cmp\_ratio),D]：(<batch0>compressed\_tokens+pad0) + (<batch1>compressed\_tokens+pad1) + ...  + (<batchN>compressed\_tokens+padN)。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
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
- 输入属性限制：
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持cmp\_ratio为2/4/8/16/32/64/128。支持如下三种典型情况：
      - C4A: D=512, coff=2, cmp\_ratio=4；
      - C4Li: D=128, coff=2, cmp\_ratio=4；
      - C128A: D=512, coff=1, cmp\_ratio=128。
  - 支持cache\_mode为1/2，分别代表连续buffer和循环buffer。**目前A3暂不支持输入2**。
  - 支持quant\_mode为1，代表HIFP8量化模式。当quant\_mode=1时，x\_descale、wkv\_descale、wgate\_descale为必选输入。
  - state\_cache支持非连续输入（通过state\_cache\_stride\_dim0属性指定stride）。
  - 该接口支持aclgraph模式。

## Atlas A3 推理系列产品 调用说明
- 单算子模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import custom_ops
    import torch.nn as nn
    import math
    from hif8_utils import hif8_to_fp32_numpy, fp32_to_hif8_numpy

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.bfloat16
    hidden_size = 4096
    coff = 1
    cmp_ratio = 128
    cache_mode = 1
    head_dim = 512
    quant_mode = 1
    cu_seqlens = [0, 1]
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B
    start_p = 8191
    seqused = None

    bs_combine_flag = True
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
            if T != 0:
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

    # page state
    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
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

        if B == 0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        token_size = (2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1)
        if B == 0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, token_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, token_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (B, token_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (B, token_size, coff * head_dim))).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x_fp32 = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(torch.float32)
    wkv_fp32 = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(torch.float32)
    wgate_fp32 = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(torch.float32)

    x = fp32_to_hif8_numpy(x_fp32.numpy()).reshape(x_shape)
    x = torch.from_numpy(x).to(torch.uint8).npu()
    wkv = fp32_to_hif8_numpy(wkv_fp32.numpy()).reshape(coff * head_dim, hidden_size)
    wkv = torch.from_numpy(wkv).to(torch.uint8).npu()
    wgate = fp32_to_hif8_numpy(wgate_fp32.numpy()).reshape(coff * head_dim, hidden_size)
    wgate = torch.from_numpy(wgate).to(torch.uint8).npu()

    x_descale = torch.tensor([1.0], dtype=torch.float32).npu()
    wkv_descale = torch.ones(coff * head_dim, dtype=torch.float32).npu()
    wgate_descale = torch.ones(coff * head_dim, dtype=torch.float32).npu()

    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()

    if cache_mode == 1:
        state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2 * kv_state.shape[2]))
        state_cache = state_cache.npu()
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
    else:
        layer_pad = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_pad-1)
        state_cache_pad = torch.zeros((kv_state.shape[0], kv_state.shape[1] * kv_state.shape[2] * 2 + layer_pad))
        state_cache_pad = state_cache_pad.to("npu:0")
        state_cache = state_cache_pad[:, layer_start_idx : layer_start_idx + kv_state.shape[1] * kv_state.shape[2] * 2].view(-1, kv_state.shape[1], kv_state.shape[2] * 2)
        state_cache = state_cache.to("npu:0")
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()

    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    cmp_kv, = (
        torch.ops.custom.quant_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            x_descale = x_descale,
            wkv_descale = wkv_descale,
            wgate_descale = wgate_descale,
            state_block_table = block_table,
            cu_seqlens = cu_seqlens,
            seqused = seqused,
            start_pos = start_pos,
            quant_mode = quant_mode,
            cmp_ratio = cmp_ratio,
            coff = coff,
            cache_mode = cache_mode,
            state_cache_stride_dim0 = 0
        )
    )
    ```
- aclgraph调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import torchair
    import custom_ops
    import math
    from hif8_utils import hif8_to_fp32_numpy, fp32_to_hif8_numpy

    import torch.nn as nn

    def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
        if seqused is not None:
            return seqused[batch_idx]
        else:
            if cu_seqlens is not None:
                return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
            else:
                return S

    data_type = torch.bfloat16
    hidden_size = 4096
    coff = 1
    cmp_ratio = 128
    cache_mode = 1
    head_dim = 512
    quant_mode = 1
    cu_seqlens = [0, 1]
    B = 1
    S = 1
    S_max = 0
    block_size = 128
    start_pos = [8191] * B
    start_p = 8191
    seqused = None

    bs_combine_flag = True
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
            if T != 0:
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

    if cache_mode == 1:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
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

        if B == 0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        token_size = (2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1)
        if B == 0:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (0, token_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (0, token_size, coff * head_dim))).to(torch.float32)
        else:
            kv_state = torch.tensor(np.random.uniform(-10, 10, (B, token_size, coff * head_dim))).to(torch.float32)
            score_state = torch.tensor(np.random.uniform(-10, 10, (B, token_size, coff * head_dim))).to(torch.float32)

    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x_fp32 = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(torch.float32)
    wkv_fp32 = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(torch.float32)
    wgate_fp32 = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(torch.float32)

    x = fp32_to_hif8_numpy(x_fp32.numpy()).reshape(x_shape)
    x = torch.from_numpy(x).to(torch.uint8).npu()
    wkv = fp32_to_hif8_numpy(wkv_fp32.numpy()).reshape(coff * head_dim, hidden_size)
    wkv = torch.from_numpy(wkv).to(torch.uint8).npu()
    wgate = fp32_to_hif8_numpy(wgate_fp32.numpy()).reshape(coff * head_dim, hidden_size)
    wgate = torch.from_numpy(wgate).to(torch.uint8).npu()

    x_descale = torch.tensor([1.0], dtype=torch.float32).npu()
    wkv_descale = torch.ones(coff * head_dim, dtype=torch.float32).npu()
    wgate_descale = torch.ones(coff * head_dim, dtype=torch.float32).npu()

    ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()

    if cache_mode == 1:
        state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2 * kv_state.shape[2]))
        state_cache = state_cache.npu()
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
    else:
        layer_pad = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_pad-1)
        state_cache_pad = torch.zeros((kv_state.shape[0], kv_state.shape[1] * kv_state.shape[2] * 2 + layer_pad))
        state_cache_pad = state_cache_pad.to("npu:0")
        state_cache = state_cache_pad[:, layer_start_idx : layer_start_idx + kv_state.shape[1] * kv_state.shape[2] * 2].view(-1, kv_state.shape[1], kv_state.shape[2] * 2)
        state_cache = state_cache.to("npu:0")
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()

    block_table = block_table.npu()
    start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).npu()

    class QuantCompressorNetwork(nn.Module):
        def __init__(self):
            super(QuantCompressorNetwork, self).__init__()

        def forward(self, x, wkv, wgate, state_cache, ape,
                    x_descale, wkv_descale, wgate_descale,
                    quant_mode, cmp_ratio, state_block_table = None, cu_seqlens = None,
                    seqused = None, start_pos = None, coff = 1, cache_mode = 1,
                    state_cache_stride_dim0 = 0):
            cmp_kv, = torch.ops.custom.quant_compressor(
                    x,
                    wkv,
                    wgate,
                    state_cache,
                    ape,
                    x_descale = x_descale,
                    wkv_descale = wkv_descale,
                    wgate_descale = wgate_descale,
                    state_block_table = state_block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    quant_mode = quant_mode,
                    cmp_ratio = cmp_ratio,
                    coff = coff,
                    cache_mode = cache_mode,
                    state_cache_stride_dim0 = state_cache_stride_dim0
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
                    x_descale = x_descale,
                    wkv_descale = wkv_descale,
                    wgate_descale = wgate_descale,
                    state_block_table = block_table,
                    cu_seqlens = cu_seqlens,
                    seqused = seqused,
                    start_pos = start_pos,
                    quant_mode = quant_mode,
                    cmp_ratio = cmp_ratio,
                    coff = coff,
                    cache_mode = cache_mode,
                    state_cache_stride_dim0 = 0)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。

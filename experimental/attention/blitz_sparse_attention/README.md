# BlitzSparseAttention — Prompt Flash Attention with Block-Sparsity

This kernel is based on PromptFlashAttentionV3, extending it with a new argument `sabi` to enable block-sparse attention computation during prefill. We provide a **torch interface** to quickly try out our kernel in end-to-end Python pipelines that may benefit from sparse computation (e.g. Hunyuan-video). Documentation of the `sabi` argument is in [docs/aclnnBlitzSparseAttention.md](docs/aclnnBlitzSparseAttention.md).

## Known limitations / TODOs

> **TODO 1: 128×128 sabi granularity speedup.** The 128×512 sabi granularity has
> shown great speedups (1.89× at 50% sparsity). However, the current 128×128
> version still uses 128×512 matmul tiles internally without first compacting
> only the sabi-selected 128×128 sub-tiles into them. In the current kernel
> design the cube therefore performs redundant matmuls which are then masked
> out during softmax — only fully empty 512-long tiles (i.e. 4 consecutive
> non-selected blocks) are truly skipped. As a result the speedup ramps up
> from sparsity ≥ 10%, in proportion to the probability of 4 consecutive
> non-selected blocks. Multiple attempts to rewrite the matmul scheduling
> have shown that a proper fix requires a full kernel rewrite; the sibling
> `attention/block_sparse_attention` kernel handles this better with its
> bottom-up CATLASS-based design.
> **TODO 2: batch size > 1 is broken.** Only `batch_size=1` (`B=1`) is currently
> known to produce correct results. Runs with `B>1` produce incorrect outputs.
> All tests and benchmarks must be run with `B=1` until the multi-batch issue
> is diagnosed and fixed.

## Quick test and benchmark in python

build the kernel as a custom experimental package, install it, then install our "torch_bsa" torch interface package

```shell
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
(cd experimental/attention/blitz_sparse_attention/torch_interface && bash build.sh custom)
```

test and benchmark run times:

```shell
cd experimental/attention/blitz_sparse_attention/benchmark
pytest test_attn.py # attention_out correctness tests for sequence lengths 10k-30k 1-4 attention heads, compares our block-sparse BSA against npu_fusion_attention kernel and our own python implementation
pytest test_lse.py # softmax_lse correctness tests for sequence lengths 10k-30k 1-4 attention heads, compares our block-sparse BSA against npu_fused_infer_attention_score kernel
pytest test_joint.py # simultaneously check correctness of both kernel outputs
python benchmark.py # performance benchmarking - check the constant inputs shapes defined in the script
```

The tests should all be green. `benchmark.py` sweeps every sparsity at every
pair in `BLOCK_SHAPES` and labels each row with its active `block_shape`. The
`frame(L,R,T,B)` column shows the per-shape frame as a compact 4-tuple, or `-`
when no frame applies (sparsity 0 ⇒ every block kept ⇒ frame irrelevant).
Trimmed sample on Ascend910B2:

```shell
========================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'  SABI_SORTED=True  TORCH_REFERENCE='npu_fusion_attention'
========================================================================================================================
  block_shape   H   B    s_q   s_kv    D  frame(L,R,T,B)  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------------------------------------
      128x128   3   1 118806 118806  128               -      0.00             yes          160647.57          186419.60
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.50             N/A                N/A          126072.34
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.90             N/A                N/A           23824.46
      128x256   3   1 118806 118806  128               -      0.00             yes          162336.54          187044.41
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.50             N/A                N/A           94882.15
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.90             N/A                N/A           18682.85
      128x512   3   1 118806 118806  128               -      0.00             yes          163961.43          186603.97
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.50             N/A                N/A           85956.16
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.90             N/A                N/A           17384.59
========================================================================================================================
```

Rows are trimmed to a few sparsities per shape; the actual sweep emits one row
per sparsity for each block_shape. Narrow `BLOCK_SHAPES` at the top of
`benchmark.py` to benchmark a single granularity.
At S=118806 D=128 BF16, 128×256 breaks even with the dense PFA reference at
sparsity ≈ 0.05, 128×512 at ≈ 0.1; the historic 1.89× speedup at sparsity 0.5
still holds for 128×512 (and 128×256 is within ~10% of it while keeping a 2×
finer sabi resolution). See [benchmark/README.md](benchmark/README.md) for the
full table and a per-sparsity PFA-speedup summary.

To invoke our block-sparse prompt flash attention kernel from python, use our provided `torch_bsa` interface. The call is compatible with `torch_npu` conventions:

```python
import torch
import torch_bsa

# Sabi granularity. Both values must be in {128, 256, 512, 1024}; smaller
# values give finer per-block control at the cost of a larger sabi tensor.
# Default (when block_shape is omitted) is [128, 128].
BLOCK_SIZE_Q, BLOCK_SIZE_KV = 128, 128

# sabi: torch.uint16, shape [B, N, ceil(S/BLOCK_SIZE_Q), ceil(S/BLOCK_SIZE_KV)].
# Each row lists the kept KV-block column indices for that Q-block, padded on
# the right with 0xFFFF (the uint16 "skip" sentinel).
sabi = ...  # build from your sparsity pattern

# Returns a tuple (attention_out, softmax_lse).
# softmax_lse is a [B, N, S] float32 tensor when softmax_lse_flag=True,
# or an empty tensor ({0}-shaped) when softmax_lse_flag=False (default).
attention_out, softmax_lse = torch_bsa.blitz_sparse_attention(
    q, k, v,
    sabi=sabi,
    actual_seq_lengths=actseqlen,
    actual_seq_lengths_kv=actseqlenkv,
    num_heads=h,
    num_key_value_heads=h,
    input_layout='BNSD',
    scale_value=scale,
    sparse_mode=0,
    softmax_lse_flag=False,                   # set True to also return the log-sum-exp output
    block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
)
```

### softmax_lse output

| Property | Value |
|---|---|
| Controlled by | `softmax_lse_flag` (bool attr, default `False`) |
| Output index | 1 (always returned; empty when flag is `False`) |
| Shape when enabled | `[B, N, S]` |
| Dtype | `float32` (regardless of Q/K/V dtype) |
| Layout | Non-TND layouts only (`BNSD`, `BSH`, `BSND`); TND returns `{0}` |
| Semantics | Per-query log-sum-exp: `log(Σ exp(q·kᵀ / √d))` over all attended KV tokens |

The LSE is computed during the same kernel pass as the attention output at no additional memory-bandwidth cost.  It is useful for ring attention, speculative decoding rescaling, and any application that needs to merge partial attention results across segments.

When `softmax_lse_flag=False` the kernel skips the LSE write-out path and returns a zero-element placeholder tensor; the caller does not need to allocate memory for it.

## Example run in C++

A plain, pure C++, example is provided in examples subdirectory. Run it using:

```shell
bash build.sh --experimental --run_example blitz_sparse_attention eager cust --soc=ascend910b --vendor_name=custom
```

<details>
<summary>the output should be (click to expand):</summary>
``` shell
[2026-05-20 10:44:46] Warning: The current environment is configured for ascend910b, Please use Atlas A2 series hardware for optimal performance.
[2026-05-20 10:44:47] [2026-05-20 10:44:47] Start to run example,name:blitz_sparse_attention mode:eager
[2026-05-20 10:44:47] Start compile and run example file: ../experimental/attention/blitz_sparse_attention/examples/test_aclnn_blitz_sparse_attention.cpp
[2026-05-20 10:44:47] pkg_mode:cust vendor_name:custom
[2026-05-20 10:44:51] Initializing ACL...
[2026-05-20 10:44:51] Initializing tensors...
[2026-05-20 10:44:51] Tensor shapes:
[2026-05-20 10:44:51]   query:  [1, 8, 512, 128] (B, N, S, D) fp16
[2026-05-20 10:44:51]   key:    [1, 8, 512, 128] (B, N, S, D) fp16
[2026-05-20 10:44:51]   value:  [1, 8, 512, 128] (B, N, S, D) fp16
[2026-05-20 10:44:51]   sabi:   [1, 8, 4, 4] (B, N, Q_tiles, KV_tiles) uint16
[2026-05-20 10:44:51]   out:    [1, 8, 512, 128] (B, N, S, D) fp16
[2026-05-20 10:44:51]   lse:    [1, 8, 512]       (B, N, S)    float32
[2026-05-20 10:44:51] Executing BlitzSparseAttention...
[2026-05-20 10:44:51] Synchronizing stream...
[2026-05-20 10:44:51] Processing results...
[2026-05-20 10:44:51] Output results (first 10 values as raw fp16 hex):
[2026-05-20 10:44:51] output[0] = 0x3C00
[2026-05-20 10:44:51] output[1] = 0x3C00
[2026-05-20 10:44:51] output[2] = 0x3C00
[2026-05-20 10:44:51] output[3] = 0x3C00
[2026-05-20 10:44:51] output[4] = 0x3C00
[2026-05-20 10:44:51] output[5] = 0x3C00
[2026-05-20 10:44:51] output[6] = 0x3C00
[2026-05-20 10:44:51] output[7] = 0x3C00
[2026-05-20 10:44:51] output[8] = 0x3C00
[2026-05-20 10:44:51] output[9] = 0x3C00
[2026-05-20 10:44:51] LSE results (first 10 values, expect ~17.5520 for all-ones input):
[2026-05-20 10:44:51] lse[0] = 17.550825
[2026-05-20 10:44:51] lse[1] = 17.550825
[2026-05-20 10:44:51] lse[2] = 17.550825
[2026-05-20 10:44:51] lse[3] = 17.550825
[2026-05-20 10:44:51] lse[4] = 17.550825
[2026-05-20 10:44:51] lse[5] = 17.550825
[2026-05-20 10:44:51] lse[6] = 17.550825
[2026-05-20 10:44:51] lse[7] = 17.550825
[2026-05-20 10:44:51] lse[8] = 17.550825
[2026-05-20 10:44:51] lse[9] = 17.550825
[2026-05-20 10:44:51] Cleaning up resources...
[2026-05-20 10:44:51] Test completed successfully!
[2026-05-20 10:44:51] run test_aclnn_blitz_sparse_attention, execute samples success
[2026-05-20 10:44:51] Example completed successfully
```
</details>

## Kernel integration plan

If this block-sparse kernel is of interest, please consider merging it with the official `attention/prompt_flash_attention`. The source is based on `attention/prompt_flash_attention` at git commit `a574b5d71faa7c360934a6c7d1b4aa85e1a49147`.

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- 算子功能：全量推理场景的FlashAttention算子，支持sparse优化、支持actualSeqLengthsKv优化、支持INT8量化功能，支持高精度或者高性能模式选择。

- 计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中：$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmax_lse</td>
    <td>输出</td>
    <td>每个query token对应的log-sum-exp值：log(Σ exp(q·kᵀ/√d))，用于ring attention等需要合并partial attention结果的场景。softmax_lse_flag为False时返回空tensor（numel=0）。</td>
    <td>FLOAT32</td>
    <td>ND，shape [B, N, S]</td>
  </tr>
  <tr>
    <td>softmax_lse_flag</td>
    <td>属性（输入）</td>
    <td>是否输出softmax_lse。不需要LSE时建议传入False（默认）。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody>
</table>

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持FLOAT16、BFLOAT16、INT8。
- Atlas 推理系列产品：仅支持FLOAT16。

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor，则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。

- query，key，value输入，功能使用限制如下：

  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：

    - 支持B轴小于等于65536（64k），输入类型包含INT8时D轴非32对齐或输入类型为FLOAT16或BFLOAT16时D轴非16对齐时，B轴仅支持到128。

    - 支持N轴小于等于256。

    - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致bsa算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
      <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>Q_N</th>
      <th>Q_S</th>
      <th>D</th>
      <th>KV_N</th>
      <th>KV_S</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>1</td>
      <td>20</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>2</td>
      <td>20971520</td>
      <td>256</td>
      <td>2</td>
      <td>20971520</td>
      </tr>
      <tr>
      <td>20</td>
      <td>1</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>10</td>
      <td>2097152</td>
      <td>512</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      </tbody>
      </table>

    - 支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。

  - Atlas A2 训练系列产品/Atlas A2 推理系列产品：在TND场景下query，key，value输入的综合限制：
    - T小于等于65536。
    - N等于8/16/32/64/128，且Q_N、K_N、V_N相等。
    - Q_D、K_D等于192，V_D等于128/192。
    - 数据类型仅支持BFLOAT16。
    - sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask。
    - 当sparse=3时，要求每个batch单独的actualSeqLengths < actualSeqLengthsKv。

- 当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出attentionOut的shape为BSND；其余情况attentionOut的shape需要与入参query的shape保持一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_BlitzSparseAttention](./examples/test_aclnn_blitz_sparse_attention.cpp) | 通过[aclnnBlitzSparseAttention](./docs/aclnnBlitzSparseAttention.md)调用BlitzSparseAttention算子 |

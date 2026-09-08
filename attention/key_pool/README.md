# KeyPool

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：KeyPool是推理场景下的Key压缩算子。算子对每个输入token分别执行K
投影和Gate投影，将连续的`cmpRatio`个token分为一组，按Gate与位置偏置
确定的权重将组内K合并为一个Key，并将未完成压缩组所需的中间状态写入分页
`stateCache`。

- 计算公式：

  对当前输入的每个token：

  ```text
  K = hiddenStates @ wk.T
  G = hiddenStates @ gateWeight.T
  K_norm = LayerNorm(K, dim=-1)  # 提供normWeightOptional/normBiasOptional时
  ```

  对第`q`个全局压缩组：

  ```text
  logits[q, r] = G[q * cmpRatio + r] + ape[r]
  weight = softmax(logits, dim=0)
  pooledKeyOut[b, q - startPool[b]] = sum(weight[r] * K_norm[q * cmpRatio + r], dim=0)
  ```

  历史token通过`cacheBlockTable`从`stateCache`读取，当前输入token的投影
结果优先使用当前调用产生的数据。对于`startPos`未按`cmpRatio`对齐的场景，
首个压缩组可以由历史token和当前输入token共同组成。`ape`只参与池化计算，
不写入`stateCache`。RoPE参数入口保留用于后续扩展，但当前版本不实现RoPE。

  LayerNorm沿每个token的最后一维`D`独立计算，不跨token归约。

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
  <tr>
    <td>hiddenStates（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的hiddenStates，表示当前调用的hidden states。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wk（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的wk，表示K投影权重。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gateWeight（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的gateWeight，表示Gate投影权重。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ape（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的ape，表示位置偏置，对应公式中的<span class="math-inline">Ape</span>。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>stateCacheRef（aclTensor*）</td>
    <td>输入/输出</td>
    <td>表示K和Gate的历史状态，并由算子原地更新。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cacheBlockTable（aclTensor*）</td>
    <td>输入</td>
    <td>表示逻辑block到物理cache block的映射表。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>startPos（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个Batch当前输入在逻辑序列中的起始位置。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>normWeightOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示LayerNorm的权重参数。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>normBiasOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示LayerNorm的偏置参数。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cosOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示RoPE使用的余弦参数。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sinOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示RoPE使用的正弦参数。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cuSeqlensOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示TH场景下各Batch在hiddenStates首轴上的累计token边界。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sequsedOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>预留的有效序列长度参数。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cmpRatio（int64_t）</td>
    <td>输入</td>
    <td>公式中的cmpRatio，表示压缩率，即每组参与池化的token数。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>normEps（double）</td>
    <td>输入</td>
    <td>表示LayerNorm的epsilon。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rotaryMode（int64_t）</td>
    <td>输入</td>
    <td>表示RoPE模式。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stateCacheStrideDim0（int64_t）</td>
    <td>输入</td>
    <td>表示stateCache第0轴的stride。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pooledKeyOut（aclTensor*）</td>
    <td>输出</td>
    <td>表示按cmpRatio个token池化后的K。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回包含算子计算流程的执行器。</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

其中：

- `D`为K和Gate的head dimension，由`wk`的第0维决定；
- `R=cmpRatio`；
- `N`为物理block数；
- `BS`为block size；
- `L`为每个Batch的最大逻辑block数；
- `Sr=ceil(L*BS/cmpRatio)`，表示pooledKeyOut的容量。
- `BSH`：Batch-Sequence-Hidden的输入布局，`hiddenStates`的shape为
  `[B,S,H]`。其中`B`表示BatchSize，`S`表示每个Batch当前输入的序列
  长度，`H`表示hiddensize。
- `TH`：Token-Hidden的输入布局，`hiddenStates`的shape为`[T,H]`。
  其中`T`表示所有Batch当前输入token数的总和；`cuSeqlensOptional`
  是shape为`[B+1]`的前缀和数组，用于描述各Batch在首轴上的token边界。
- `RoPE`：Rotary Position Embedding，一种位置编码方式，通常使用`cos`
  和`sin`对K的最后一维中的成对通道执行旋转。本算子当前仅保留RoPE
  参数入口，不执行RoPE，`cosOptional`和`sinOptional`必须传空指针。

输入tensor使用ND格式。`stateCacheRef`是原地输入，用于保存跨算子调用
的未完成压缩组状态。

## 约束说明

- 支持BSH和TH两种场景：
  - BSH场景`cuSeqlensOptional`必须为空；
  - TH场景必须提供合法的`cuSeqlensOptional`前缀和数组，首元素为0，
    末元素为`T`，且单调不减。
- `cmpRatio`仅支持`2/4/8/16/32/64/128`；`normEps`必须大于0；
  `rotaryMode`仅支持`0/1`。
- `normWeightOptional`和`normBiasOptional`必须同时提供或同时为空。
- `cosOptional`和`sinOptional`必须同时为空，当前不支持RoPE；
  `sequsedOptional`必须为空，当前不支持通过`attention_mask`选择token。
- 支持BSH场景的`B=0`或`S=0`，以及TH场景的`T=0`。
- `cmpRatio`的默认值为4，取值范围是`2/4/8/16/32/64/128`。
- `normEps`的默认值为`1e-6`，取值范围是大于0。
- `rotaryMode`的默认值为1，取值范围是`0/1`，当前不执行RoPE。

## 调用方式

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_key_pool](./examples/test_aclnn_key_pool.cpp) | 通过[aclnnKeyPool](./docs/aclnnKeyPool.md)接口方式调用KeyPool算子。 |
| PyTorch API | - | 通过 [key_pool.md](../../torch_extension/cann_ops_transformer/docs/zh/key_pool.md) 接口调用 KeyPool 算子。 |

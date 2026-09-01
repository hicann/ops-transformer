# BlockAttnResUpdate

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|×|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|×|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- 算子功能：完成BlockAttnRes第二阶段更新。算子首先使用增量delta更新局部注意力结果partial_block，再计算更新结果与pseudo_query的归一化相关分数，并结合已有在线Softmax状态numerator、logit_max和exp_sum生成注意力输出h。

- 计算公式：

对于每个token $t$，首先更新局部注意力结果：

$$
p_t = partial\_block_t + \operatorname{float}(delta_t)
$$

计算更新结果的RMS和归一化相关分数：

$$
rms_t = \sqrt{\frac{1}{D}\sum_{j=0}^{D-1}p_{t,j}^{2} + eps}
$$

$$
score_t = \frac{\sum_{j=0}^{D-1}p_{t,j} \cdot pseudo\_query_j}{rms_t}
$$

使用在线Softmax方式融合已有状态和当前结果：

$$
max_t = \max(logit\_max_t, score_t)
$$

$$
alpha_t = \exp(logit\_max_t-max_t), \qquad
beta_t = \exp(score_t-max_t)
$$

$$
denominator_t = exp\_sum_t \cdot alpha_t + beta_t
$$

最终输出为：

$$
partial\_block\_out_t = p_t
$$

$$
h_t = \operatorname{bfloat16}\left(
numerator_t \cdot \frac{alpha_t}{denominator_t} +
p_t \cdot \frac{beta_t}{denominator_t}\right)
$$

其中，$T$表示token数，$D$表示隐藏维度，$t \in [0,T)$，$j \in [0,D)$。

对于token $t$ 的非空历史集合 $\mathcal{H}_t$，设历史logit为 $s_{t,i}$、对应的$D$维value为
$v_{t,i}$，则输入的在线Softmax状态必须满足：

$$
logit\_max_t = \max_{i \in \mathcal{H}_t}s_{t,i}
$$

$$
exp\_sum_t = \sum_{i \in \mathcal{H}_t}\exp(s_{t,i}-logit\_max_t) > 0
$$

$$
numerator_t = \sum_{i \in \mathcal{H}_t}\exp(s_{t,i}-logit\_max_t)v_{t,i}
$$

本算子只读取该历史状态，并将当前的$p_t$与其融合，不更新numerator、logit_max或exp_sum。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>partial_block</td>
      <td>输入</td>
      <td>待更新的局部注意力结果，公式中的partial_block，shape为$(T, D)$；与同名输出复用存储。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>输入</td>
      <td>局部注意力结果的增量，公式中的delta，shape为$(T, D)$。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>输入</td>
      <td>用于计算归一化相关分数的伪Query向量，公式中的pseudo_query，shape为$(D)$。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态对应的分子，公式中的numerator，shape为$(T, D)$；本算子只读。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态中每个token的有限最大值，公式中的logit_max，shape为$(T)$；本算子只读。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态中每个token的有限正指数和，公式中的exp_sum，shape为$(T)$；本算子只读。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>partial_block</td>
      <td>输出</td>
      <td>更新后的局部注意力结果，公式中的partial_block_out，shape为$(T, D)$；与同名输入构成引用关系。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输出</td>
      <td>融合更新后的局部结果与已有在线Softmax状态得到的注意力结果，公式中的h，shape为$(T, D)$。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>可选属性</td>
      <td>RMS计算中的稳定项，必须为有限值且大于0，默认值为$10^{-6}$。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- $T$和$D$必须是已知正整数，且$D$的取值范围为$[1, 8192]$。
- partial_block、delta、numerator及输出partial_block、h的shape必须相同，均为$(T, D)$；pseudo_query、logit_max和exp_sum的shape必须分别为$(D)$、$(T)$和$(T)$。
- 所有输入的原始格式和存储格式均必须为ND；所有输入、输出的StorageShape必须与OriginShape完全一致。
- numerator、logit_max和exp_sum是位于GM（Global Memory）中的运行时只读状态，Host Tiling不读取或校验其元素值。
  调用者必须保证每个token的历史状态非空：logit_max和numerator中的元素为有限值，exp_sum为有限值且严格大于0，
  并且三者来自同一份历史状态。当前不支持以`logit_max=-inf`、`exp_sum=0`和零numerator表示的空历史状态。
- $T \times D$不能超出有符号64位Kernel GM元素偏移的表示范围；按核切分后的$T$大小不能超出uint32_t的表示范围。
- 一行完整的$D$维数据必须能够以双缓冲方式放入统一缓冲区（Unified Buffer，UB），否则Tiling失败。
- partial_block为原地更新参数。框架调用时由引用关系保证输入、输出复用存储；Kernel直接通过partial_block输入地址读写，不访问对应的输出ABI地址。
- 当前Host Tiling申请的workspace大小为0，Kernel不访问workspace地址。
- RMS计算使用FP32的sqrt和除法，不进行牛顿迭代，也不对输入Tensor中的零值、极值、NaN或Inf做额外处理。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="examples/arch35/test_aclnn_block_attn_res_update.cpp">test_aclnn_block_attn_res_update.cpp</a></td>
    <td>通过aclnnBlockAttnResUpdate接口调用，详细说明参见<a href="docs/aclnnBlockAttnResUpdate.md">接口文档</a>。</td>
  </tr>
  <tr>
    <td>PyTorch API</td>
    <td>-</td>
    <td>通过<a href="../../torch_extension/cann_ops_transformer/docs/zh/block_attn_res_update.md">cann_ops_transformer.block_attn_res_update</a>接口调用。</td>
  </tr>
</tbody>
</table>

```bash
## 快速启动
# 在仓库根目录执行，假设已准备好环境变量
bash build.sh --soc=${soc_version} --ops=block_attn_res_update
```

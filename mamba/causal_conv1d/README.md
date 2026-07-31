# CausalConv1d

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      x     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      x     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：完成因果一维卷积（Causal Conv1d）计算，支持前向计算（prefill / chunk-prefill）和状态更新（decode / update）两种运行模式，模式由输入形状自动推断。

- 计算公式：

  Causal Conv1d 是一种因果一维卷积算子，常用于序列建模中。在每个时间步 $t$，根据当前输入 $x_t$、卷积权重 $w$ 和历史状态，计算卷积输出 $y_t$。

  $$
  y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
  $$

  其中，$W$ 为卷积核宽度（支持2、3、4），$w_j$ 为卷积权重，$b$ 为偏置（可选），$\text{Activation}$ 为激活函数（可选，SiLU）。当 `activation_mode="none"` 时不使用激活函数，`activation_mode="silu"` 时使用 SiLU 激活函数。

   算子同时维护卷积状态 `conv_states`，用于在增量推理时缓存历史输入，实现高效的状态更新。

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
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>输入序列，公式中的x。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>卷积权重，公式中的w。</td>
    <td>同x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>conv_states</td>
    <td>输入/输出</td>
    <td>
      <ul>
        <li>卷积状态，缓存历史输入用于因果卷积计算。</li>
        <li>各序列计算完成后原地更新。</li>
      </ul>
    </td>
    <td>同x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>可选输入</td>
    <td>偏置，公式中的b。若不提供则默认为0。</td>
    <td>同x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query_start_loc</td>
    <td>可选输入</td>
    <td>序列起始位置索引，记录各序列在拼接张量x中的起始位置。queryStartLoc[0]必须为0，queryStartLoc[-1]必须为cu_seq_len。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cache_indices</td>
    <td>可选输入</td>
    <td>缓存索引，指定每个序列对应的缓存状态在conv_states中的索引。不传时使用恒等映射。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initial_state_mode</td>
    <td>可选输入</td>
    <td>初始状态标志。1：使用缓存的conv_states作为历史，0：零初始化历史（冷启动）。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>num_accepted_tokens</td>
    <td>可选输入</td>
    <td>每个序列接受的token数量，用于投机解码。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>activation_mode</td>
    <td>可选属性</td>
    <td>
      <ul>
        <li>激活函数类型。"silu"：使用SiLU激活函数，"none"：不使用激活函数。</li>
        <li>默认值为"silu"。</li>
      </ul>
    </td>
    <td>STR</td>
    <td>-</td>
  </tr>
  <tr>
    <td>null_block_id</td>
    <td>可选属性</td>
    <td>
      <ul>
        <li>无效缓存槽位的标记ID，用于跳过不需要计算的序列。</li>
        <li>默认值为0。</li>
      </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>卷积输出，公式中的y。</td>
    <td>同x</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入tensor的shape大小需满足一定约束，具体见 [aclnnCausalConv1dFn](./docs/aclnnCausalConv1dFn.md) 和 [aclnnCausalConv1dUpdate](./docs/aclnnCausalConv1dUpdate.md)。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 (prefill) | [test_aclnn_causal_conv1d_fn.cpp](./examples/test_aclnn_causal_conv1d_fn.cpp) | 通过 [aclnnCausalConv1dFn](./docs/aclnnCausalConv1dFn.md) 调用 prefill 模式 |
| aclnn接口 (update) | [test_aclnn_causal_conv1d_update.cpp](./examples/test_aclnn_causal_conv1d_update.cpp) | 通过 [aclnnCausalConv1dUpdate](./docs/aclnnCausalConv1dUpdate.md) 调用 update 模式 |

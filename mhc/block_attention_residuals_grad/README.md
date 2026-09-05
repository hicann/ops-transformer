# BlockAttentionResidualsGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：`BlockAttentionResidualsGrad` 是正向算子 `BlockAttentionResiduals`（注意力残差）的反向传播算子，融合 softmax、RMS 归一化与注意力输出的反向计算。
- **主要输出**：`gradPartialBlock`、`gradBlockRes`、`gradProjWeight`、`gradNormWeight`。
- **前向缓存依赖**：前向保存的 `invNorm`、`probs`；当前版本直接使用保存结果计算，不依赖 `validBlockNum` 重新构造掩码。

- **计算公式**：

反向计算中的拼接 value、归一化中间量定义如下：

$$
V_{t,i,h} =
\begin{cases}
block\_res_{t,i,h}, & i < N \\
partial\_block_{t,h}, & i = N
\end{cases}
$$

$$
k_{t,i,h} = V_{t,i,h} \cdot inv\_norm_{t,i}
$$

$$
score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
$$

`out` 对 `V`、`probs` 的反向梯度：

$$
g_{t,i} = \sum_{h=0}^{H-1} grad\_output_{t,h} \cdot V_{t,i,h}
$$

$$
grad\_score_{t,i} = probs_{t,i} \cdot \left(g_{t,i} - \sum_{j=0}^{N} probs_{t,j} \cdot g_{t,j}\right)
$$

经 `score_weight` 与 RMS 归一化回传的梯度：

$$
grad\_k_{t,i,h} = grad\_score_{t,i} \cdot score\_weight_{h}
$$

$$
grad\_score\_weight_{h} = \sum_{t=0}^{T-1}\sum_{i=0}^{N} grad\_score_{t,i} \cdot k_{t,i,h}
$$

$$
grad\_inv\_norm_{t,i} = \sum_{h=0}^{H-1} grad\_k_{t,i,h} \cdot V_{t,i,h}
$$

`V` 的总梯度及最终输出梯度：

$$
grad\_V_{t,i,h} = grad\_output_{t,h} \cdot probs_{t,i} + grad\_k_{t,i,h} \cdot inv\_norm_{t,i}
                - \frac{grad\_inv\_norm_{t,i} \cdot inv\_norm_{t,i}^{3}}{H} \cdot V_{t,i,h}
$$

$$
grad\_block\_res_{t,i,h} = grad\_V_{t,i,h}, \quad i < N
$$

$$
grad\_partial\_block_{t,h} = grad\_V_{t,N,h}
$$

$$
grad\_norm\_weight_{h} = grad\_score\_weight_{h} \cdot proj\_weight_{0,h}
$$

$$
grad\_proj\_weight_{0,h} = grad\_score\_weight_{h} \cdot norm\_weight_{h}
$$

其中 $T$ 为 token 数，$N$ 为 `blockRes` 的 block 数，$H$ 为 hidden size。

## 参数说明

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 520px">
  <col style="width: 220px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>partialBlock</td>
      <td>输入</td>
      <td>前向输入前缀和，拼接后作为第 $N+1$ 个 value，shape 为 $[T,H]$。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockRes</td>
      <td>输入</td>
      <td>前向输入分块残差，拼接后作为前 $N$ 个 value，shape 为 $[T,N,H]$。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>projWeight</td>
      <td>输入</td>
      <td>前向投影权重，与 normWeight 共同构成 score_weight，shape 为 $[1,H]$。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normWeight</td>
      <td>输入</td>
      <td>前向归一化权重，与 projWeight 共同构成 score_weight，shape 为 $[H]$。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradHiddenStates</td>
      <td>输入</td>
      <td>前向输出 out 的上游梯度，shape 为 $[T,H]$。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invNorm</td>
      <td>输入</td>
      <td>前向保存的逐行归一化系数，shape 为 $[T,N+1]$。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probs</td>
      <td>输入</td>
      <td>前向 softmax 输出概率，shape 为 $[T,N+1]$。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>validBlockNum</td>
      <td>属性</td>
      <td>预留属性，默认值为 0，当前版本不参与计算。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradPartialBlock</td>
      <td>输出</td>
      <td>partialBlock 的梯度，shape 与 partialBlock 一致。</td>
      <td>同主输入</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradBlockRes</td>
      <td>输出</td>
      <td>blockRes 的梯度，shape 与 blockRes 一致。</td>
      <td>同主输入</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradProjWeight</td>
      <td>输出</td>
      <td>projWeight 的梯度，shape 与 projWeight 一致。</td>
      <td>同主输入</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradNormWeight</td>
      <td>输出</td>
      <td>normWeight 的梯度，shape 与 normWeight 一致。</td>
      <td>同主输入</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- $T \ge 1$，$0 \le N \le 128$，$H \ge 1$；各张量中的 $T$、$H$ 以及 `invNorm/probs` 的第 2 维 $N+1$ 需保持一致。
- 主输入 `partialBlock/blockRes/projWeight/normWeight/gradHiddenStates` 支持 FLOAT16、BFLOAT16、FLOAT32，dtype 需一致；`invNorm/probs` 仅支持 FLOAT32。
- 输入支持非连续 Tensor，接口内部会先转为 Contiguous 再计算。
- 输出 dtype 与对应主输入保持一致。
- `validBlockNum` 为预留属性，不同取值不影响当前版本的计算结果。
- `aclnnBlockAttentionResidualsGrad` 默认确定性实现。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="./examples/test_aclnn_block_attention_residuals_grad.cpp">test_aclnn_block_attention_residuals_grad.cpp</a></td>
    <td>通过 <a href="./docs/aclnnBlockAttentionResidualsGrad.md">aclnnBlockAttentionResidualsGrad.md</a> 调用算子。</td>
  </tr>
  <tr>
    <td>SPLIT_H 示例</td>
    <td><a href="./examples/test_aclnn_block_attention_residuals_grad_split_h.cpp">test_aclnn_block_attention_residuals_grad_split_h.cpp</a></td>
    <td>演示大 H 场景下 SPLIT_H 模板的调用方式。</td>
  </tr>
</tbody>
</table>

```bash
# 在 ops-transformer 仓库根目录执行
bash build.sh --pkg --soc=ascend950 --ops=block_attention_residuals_grad
bash build.sh --run_example block_attention_residuals_grad eager cust --soc=ascend950 --vendor_name=custom
```

# BlockAttnResPrepare

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：完成Attention Residuals历史残差注意力两阶段计算的第一阶段，批量计算全部 $S$ 个目标slot的历史在线Softmax状态 $(O,M,L)$，供下一阶段与当前残差合并。

- 计算公式：

  设 $V$ 为历史残差`block_res`的FP32表示，$Q$ 为有效Query`pseudo_query`，$N_v=\min(\texttt{valid\_blocks[0]},N)$ 为有效历史残差块数，$\epsilon=\operatorname{FP32}(\texttt{eps})$。

  当 $N_v>0$ 时，先计算历史残差的均方根归一化（Root Mean Square Normalization，RMSNorm）和注意力logit：

  $$
  \widehat V_{t,n}=\operatorname{RMSNorm}(V_{t,n},\mathbf{1},\epsilon),\qquad
  Z_{s,t,n}=\left\langle Q_s,\widehat V_{t,n}\right\rangle
  $$

  再沿有效历史残差块维度计算在线Softmax状态：

  $$
  M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},\qquad
  E_{s,t,n}=\exp(Z_{s,t,n}-M_{s,t})
  $$

  $$
  L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},\qquad
  O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
  $$

  其中：

  - $T$、$N$、$S$、$D$ 分别表示token数、历史残差块容量、目标slot数和隐藏维度；$t\in[0,T)$、$n\in[0,N_v)$、$s\in[0,S)$、$d\in[0,D)$ 为对应索引，$V_{t,n}$ 和 $Q_s$ 为 $D$ 维向量。
  - RMSNorm沿 $D$ 维计算，缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。
  - $O$、$M$、$L$ 分别对应输出`numerator`、`logit_max`和`exp_sum`。$O$ 是未除以 $L$ 的加权分子。

  当 $N_v=0$ 时，返回在线Softmax的空状态：

  $$
  O=\mathbf{0}_{S\times T\times D},\qquad
  M=m_{\min}\mathbf{1}_{S\times T},\qquad
  L=\mathbf{0}_{S\times T},\qquad
  m_{\min}=-3.4028234663852886\times10^{38}
  $$

  $m_{\min}$ 为FLOAT32最小有限值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| block_res | 输入 | 历史残差块，对应公式中的V，shape为[T, N, D]。 | BFLOAT16、FLOAT32 | ND |
| valid_blocks | 输入 | 有效历史残差块数，shape为[1]。0表示空历史，大于N时按N处理。 | UINT64 | ND |
| pseudo_query | 输入 | 伪Query矩阵Q，shape为[S, D]，每行对应一个目标slot。 | FLOAT32 | ND |
| numerator | 输出 | softmax加权分子，对应公式中的O，shape为[S, T, D]。 | FLOAT32 | ND |
| logit_max | 输出 | 有效历史来源logit的最大值，对应公式中的M，shape为[S, T]。 | FLOAT32 | ND |
| exp_sum | 输出 | max-shift后的指数和，对应公式中的L，shape为[S, T]。 | FLOAT32 | ND |
| eps | 可选属性 | RMS归一化的数值稳定项，取值范围为FLOAT32正规正数`[FLT_MIN, FLT_MAX]`，默认值为`1e-6`。 | FLOAT32 | - |

## 约束说明

- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- 非空计算（`T > 0`且`S > 0`）时，`T`和`S`不能超过`4294967295`（`UINT32_MAX`），`T * S`不能超过`9223372036854775807`（`INT64_MAX`）。
- 所有输入和输出Tensor均须连续。
- `eps`必须为有限值且位于FLOAT32正规正数范围`[FLT_MIN, FLT_MAX]`（约`[1.1754943508222875e-38, 3.4028234663852886e+38]`）。ACLNN接收的double值经范围校验后转换为FLOAT32属性，不支持次正规数或收窄后为0、Inf的值。
- `T == 0`或`S == 0`时，在参数校验和形状推导通过后返回对应shape的空输出，不发起AICore计算；上述非空计算的上限不适用于该路径，`eps`有效范围及其他接口校验仍适用。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_block_attn_res_prepare](examples/arch35/test_aclnn_block_attn_res_prepare.cpp) | 通过[aclnnBlockAttnResPrepare](docs/aclnnBlockAttnResPrepare.md)接口调用。编译和运行方法参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)。 |
| PyTorch API | [调用示例](docs/torchapi_block_attn_res_prepare.md#调用示例) | 通过[block_attn_res_prepare](docs/torchapi_block_attn_res_prepare.md)接口调用。 |

```bash
# 在ops-transformer仓库根目录执行
bash build.sh --pkg --soc=ascend950 --ops=block_attn_res_prepare
(cd build_out && ./cann-ops-transformer-*.run)
bash build.sh --run_example block_attn_res_prepare eager cust --soc=ascend950 --vendor_name=custom
```

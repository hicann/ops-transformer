# DequantRopeQuantKvcache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量（x）进行dequant（可选）后，按`sizeSplits`（为切分的长度）对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut和kOut，之后对kOut和vOut进行量化并按照`indices`更新到kCacheRef和vCacheRef上。

- 计算公式：

  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$

  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$

  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$

  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$

  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$

  如果cacheModeOptional为contiguous则：

  $$
  kCacheRef[i][indices[i]]=quantK[i]
  $$

  $$
  vCacheRef[i][indices[i]]=quantV[i]
  $$

  如果cacheModeOptional为page则：

  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$

  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$

  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$

  $$
  vCacheRefView[indices[i]]=quantV[i]
  $$

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于切分的输入`x`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、INT32、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cos</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于位置编码的输入`cos`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sin</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于位置编码的输入`sin`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">BFLOAT16、FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中用于缓存k的输入`kCacheRef`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中用于缓存v的输入`vCacheRef`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">indices</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中表示Kvcache的token位置信息的输入`indices`。</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleK</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入`scaleK`用于量化`k`的scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleV</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入`scaleV`用于量化`v`的scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetKOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`offsetKOptional`用于量化k的offset因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetVOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`offsetVOptional`用于量化v的offset因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weightScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`weightScaleOptional`用于反量化的权重scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`activationScaleOptional`用于反量化的激活scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">biasOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入用于反量化的偏置`biasOptional`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT16(HALF)、INT32、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sizeSplits</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">表示输入的qkv进行切分的长度。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quantModeOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示支持的量化类型，目前仅支持static。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示支持的数据格式，目前仅支持BSND。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kvOutput</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Host侧表达式布尔值。表示是否输出kOut和vOut。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cacheModeOptional</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示kCacheRef的更新方式，目前仅支持page和contiguous，默认为contiguous。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示经过处理的q，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示经过处理的k，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示经过处理的v，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>: 不支持BFLOAT16。

## 约束说明

- cacheModeOptional为contiguous时：kCacheRef的第0维大于等于x的第0维；x为3维时，indices数据值大于等于0且小于等于kCacheRef的第1维减x的第1维；x为2维时，indices数据值大于等于0且小于等于kCacheRef的第1维减1。cacheModeOptional为page时：indices数据值大于等于0，小于kCacheRef的第0维*第1维且不重复。
- x的尾轴小于等于4096，且按64对齐。
- 输入x不为int32时，x、cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时activationScaleOptional，weightScaleOptional、biasOptional不生效；x为int32时，cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时weightScaleOptional必选，activationScaleOptional、biasOptional可选（biasOptional不需要与其他输入类型一致）。
- sizeSplits的元素个数必须为3，各元素均不得小于0；sizeSplits[1]与sizeSplits[2]必须相等，且分别等于Nkv*D；x的尾轴必须等于sizeSplits三个元素之和。
- kCacheRef和vCacheRef必须为4维且shape一致；D必须为16的正整数倍。scaleK、scaleV以及可选输入offsetKOptional、offsetVOptional的元素个数均必须为Nkv*D。
- qOut的shape在x为3维时必须为[B, S, Nq, D]，在x为2维时必须为[B, Nq, D]。kvOutput为true时，kOut和vOut在x为3维时的shape均必须为[B, S, Nkv, D]，在x为2维时均必须为[B, Nkv, D]。维度数和各轴大小均须满足约束，不支持仅保持元素个数相同的其他shape。
- kvOutput为false时，kOut和vOut均必须为空Tensor（元素个数为0），不能传入空指针。通过aclnn调用时，输出shape的tiling校验失败会使第一段接口返回ACLNN_ERR_INNER_TILING_ERROR（561002）。
- 支持B=0或S=0的空tensor场景：x为2维且B=0，或x为3维且B=0或S=0时，qOut、kOut、vOut的元素个数为0，kCacheRef、vCacheRef保持输入值不变。该场景仍要求x尾轴、Nkv、D、scaleK、scaleV等结构参数满足上述约束；D=0、Nkv=0或必选scale为空不属于本支持场景。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_rope_quant_kvcache](examples/test_aclnn_dequant_rope_quant_kvcache.cpp) | 通过接口方式调用[DequantRopeQuantKvcache](docs/aclnnDequantRopeQuantKvcache.md)算子。 |

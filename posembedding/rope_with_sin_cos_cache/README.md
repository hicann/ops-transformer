# RopeWithSinCosCache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。
- 计算公式：

    1、**mrope模式**：positions的shape输入是[3, numTokens]：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    cos0 = cos[0, :, :mropeSection[0]]
    $$

    $$
    cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
    $$

    $$
    cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
    $$

    $$
    cos = torch.cat((cos0, cos1, cos2), dim=-1)
    $$

    $$
    sin0 = sin[0, :, :mropeSection[0]]
    $$

    $$
    sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
    $$

    $$
    sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
    $$

    $$
    sin= torch.cat((sin0, sin1, sin2), dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$
    
    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    2、**rope模式**：positions的shape输入是[numTokens]：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = query\_rot[..., ::2]
    $$

    $$
    x2 = query\_rot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
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
      <td style="white-space: nowrap">positions</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，输入索引。</td>
      <td style="white-space: nowrap">INT32、INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryIn</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，表示要执行旋转位置编码的第一个张量，公式中的`query`。</td>
      <td style="white-space: nowrap">BFLOAT16、FLOAT16、FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">keyIn</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，表示要执行旋转位置编码的第二个张量。</td>
      <td style="white-space: nowrap">BFLOAT16、FLOAT16、FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cosSinCache</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，表示参与计算的位置编码张量。</td>
      <td style="white-space: nowrap">BFLOAT16、FLOAT16、FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropeSection</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">mrope模式下用于整合输入的位置编码张量信息，公式中的`mropeSection`。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">headSize</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">表示每个注意力头维度大小。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">isNeoxStyle</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">true表示rotate\_half（GPT-NeoX style）计算模式，false表示rotate\_interleaved（GPT-J style）计算模式。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">输出query执行旋转位置编码后的结果。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">keyOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">输出key执行旋转位置编码后的结果。</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

## 约束说明

- queryIn、keyIn、cosSinCache只支持2维shape输入。
- headSize支持范围: 16~128。数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数。
- rotaryDim支持范围: 16~128，始终小于等于headSize。数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数。
- 当输入tensor positions中值域超过cosSinCache的0维maxSeqLen，会有越界报错。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rope_with_sin_cos_cache](examples/test_aclnn_rope_with_sin_cos_cache.cpp) | 通过[aclnnRopeWithSinCosCache](docs/aclnnRopeWithSinCosCache.md)接口方式调用RopeWithSinCosCache算子。 |

# MatmulAllReduceAddRmsNorm

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   | x |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：完成mm + all_reduce + add + rms_norm计算。
- 计算公式：
  - 情景一：

    $$
    mm\_out = allReduce(x1 @ x2 + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

 - 情景二：

    $$
    mm_out = allReduce(dequant_scale * (x1_{int8}@x2_{int8} + bias_{int32}))
    $$

    $$
    y = mm_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

  - 情景三：

    $$
    mm\_out = allReduce(x1 @ (x2*antiquant_scale + antiquant_offset) + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>x1</td>
      <td>输入</td>
      <td>MatMul左矩阵，即公式中的输入x1。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>MatMul右矩阵，即公式中的输入x2。</td>
      <td>FLOAT16、BFLOAT16、INT8、INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>Matmul计算之后的Add项，即公式中的输入bias。</td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>residual</td>
      <td>输入</td>
      <td>AddRmsNorm融合算子的残差输入，即公式中的输入residual。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>AddRmsNorm融合算子的RmsNorm计算输入，即公式中的输入gamma。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_scale</td>
      <td>可选输入</td>
      <td>公式中的输入antiquant_scale。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_offset</td>
      <td>可选输入</td>
      <td>对x2进行伪量化计算的offset参数，公式中的输入antiquant_offset。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequant_scale</td>
      <td>可选输入</td>
      <td>mm计算后的全量化系数，公式中的输入dequant_scale。</td>
      <td>FLOAT16、BFLOAT16、UINT64、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>mm + all_reduce + add的结果。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>norm_out</td>
      <td>输出</td>
      <td><ul><li>公式中的输出norm_out。</li><li>mm + all_reduce + add + rms_norm的结果。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><ul><li>通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></ul></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduceOp</td>
      <td>可选属性</td>
      <td><ul><li>reduce操作类型。</li><li>默认值为"sum"。</li></ul></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>可选属性</td>
      <td><ul><li>决定x1是否执行矩阵乘前进行转置。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_b</td>
      <td>可选属性</td>
      <td><li>决定x2是否执行矩阵乘前进行转置。</li><li>默认值为false。</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>可选属性</td>
      <td><ul><li>通信数据切分数，即总数据量/单次通信量。</li><li>默认值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_group_size</td>
      <td>可选属性</td>
      <td><ul><li>伪量化pergroup模式下，对x2进行反量化计算的groupSize输入。</li><li>默认值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，即公式中的输入epsilon。</li><li>epsilon取值满足取值范围(0,1)。</li><li>默认值为1e-6。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

* 增量场景不开启MC2，全量场景开启MC2
* 输入x1可为二维或者三维，其shape为(b, s, k)或者(s, k)。x2必须是二维，其shape为(k, n)，轴满足mm算子入参要求，k轴相等，m的范围为[1, 2147483647]，k的范围为[1, 65535]，n的范围为[0, 65535]。bias若非空，bias为一维，其shape为(n)。bias可选，可为空，非空时当前版本仅支持一维输入。
* 输入residual必须是三维，其shape为(b, s, n)，当x1为二维时，residual的(b*s)等于x1的s，不支持非连续的tensor。输入gamma必须是一维，其shape为(n)，不支持非连续的tensor。
* antiquant_scale满足pertensor场景shape为(1)，perchannel场景shape为(1,n)/(n)，pergroup场景shape为(ceil(k,antiquant_group_size),n)。antiquant_offset可选，可为空，非空时shape与antiquant_scale一致。
* dequant_scale的shape在pertensor场景为(1)，perchannel场景为(n)/(1, n)。
* 输出y和normOut的维度和数据类型同residual。bias若非空，shape大小与normOut最后一维相等。
* bias、residual、gamma、y、normOut计算输入的数据类型要一致。
* antiquant_group_size在不支持pergroup场景时，传入0，在支持pergroup场景时，传入值的范围为[32, min(k-1,INT_MAX)]，且为32的倍数。k取值范围与mm接口保持一致。
* 支持(b*s)、n为0的空tensor，不支持k为0的空tensor。
* 只支持x2矩阵转置/不转置，x1矩阵支持不转置场景。
* 属性reduceOp当前版本仅支持输入"sum"。
* 属性commTurn当前版本仅支持输入0。
* 支持1、2、4、8卡，并且仅支持hccs链路all mesh组网。
* <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：一个模型中的通算融合MC2算子，仅支持相同通信域。类型要一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_all_reduce_add_rms_norm.cpp](./examples/test_aclnn_matmul_all_reduce_add_rms_norm.cpp) | 通过[aclnnMatmulAllReduceAddRmsNorm](./docs/aclnnMatmulAllReduceAddRmsNorm.md)接口方式调用MatmulAllReduceAddRmsNorm算子。 |

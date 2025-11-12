# MatmulAllReduce

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | x |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |

## 功能说明


- 算子功能：完成MatMul计算与AllReduce通信融合。
- 计算公式：

  非量化场景：
  - 情形1：
    $$
    y_i=x_i\times weight_i + bias_i
    $$
  - 情形2：
    $$
    output = allreduce(x1 @ x2 + bias + x3)
    $$
  - 情形3：对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。
    $$
    output= allReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
    $$
  - 情形4：对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant和pertoken计算，接着与x3进行Add操作，最后做AllReduce计算。
    $$
    output= allReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
    $$
  - 情形5：对量化后的入参x1、x2进行MatMul、Dequant和pertoken计算，接着与x3进行Add操作，再对输出进行per-channel量化，然后进行AllToAll通信，对第一次通讯结果进行reduceSum计算，接着进行AllGather通信，最后对第二次通信结果进行Dequant，得到最终输出。
    $$
    matmulAddOutput = (dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
    $$

    $$
    alltoallOutPut_{int8} = alltoall(matmulAddOutput / commQuantScale1Optional); 
    $$

    $$
    reduceSumOutPut_{int8} = (add(alltoallOutPut_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
    $$

    $$
    outPut = (allgather(reduceSumOutPut_{int8}) * commQuantScale2Optional);
    $$
  - 情形6：
    - commQuantScale1Optional, commQuantScale2Optional不为空时:

      $$
      matmulAddOutput = (x2Scale * x1ScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
      $$

      $$
      alltoallOutput_{int8} = alltoall(matmulAddOutput / commQuantScale1Optional);
      $$

      $$
      reduceSumOutput_{int8} = (add(alltoallOutput_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
      $$

      $$
      output = (allgather(reduceSumOutput_{int8}) * commQuantScale2Optional);
      $$
    - x1，x2为INT8，无x1ScaleOptional，x2Scale为INT64/UINT64，可选biasOptional为INT32，out为BFLOAT16/FLOAT16：

      $$
      output = allReduce((x1@x2 + biasOptional) * x2Scale + x3Optional)
      $$
    - x1，x2为INT8，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32/BFLOAT16，可选biasOptional为INT32, out为FLOAT16/BFLOAT16：

      $$
      output = allReduce((x1@x2 + biasOptional) * x2Scale * x1ScaleOptional + x3Optional)
      $$
    - x1，x2为FLOAT4_E2M1/FLOAT4_E1M2/FLOAT8_E4M3FN/FLOAT8_E5M2，x1ScaleOptional为FLOAT8_E8M0，x2Scale为FLOAT8_E8M0，可选biasOptional为FLOAT32, out为FLOAT16/BFLOAT16/FLOAT32：

      $$
      output = allReduce((x1* x1ScaleOptional)@(x2* x2Scale) + biasOptional + x3Optional)
      $$
    - x1，x2为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32，可选bias为FLOAT32, out为FLOAT16/BFLOAT16/FLOAT32：

      $$
      output = allReduce((x1@x2 + biasOptional) * x2Scale * x1ScaleOptional + x3Optional)
      $$
    - x1，x2为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8，x1ScaleOptional为FLOAT32，x2Scale为FLOAT32，无biasOptional。当x1为(a0, a1)，x2为(b0, b1)时x1ScaleOptional为(ceildiv(a0，128), ceildiv(a1，128))x2Scale为(ceildiv(b0，128), ceildiv(b1，128)), out为FLOAT16/BFLOAT16/FLOAT32:

      $$
      output_{pq} = allReduce(\sum_{0}^{\left \lfloor \frac{k}{128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1ScaleOptional_{pr}*x2Scale_{rq})) + x3)
      $$
  - 情形7：
    $$
    output = allreduce(x1 @ ((x2 + antiquantOffset) *antiquantScale) + bias+ x3) 
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
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>MatMul右矩阵，即公式中的输入x2。</td>
      <td>FLOAT16、BFLOAT16、INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND、FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>公式中的输入bias。</td>
      <td>FLOAT16、BFLOAT16、INT32、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>可选输入</td>
      <td>MatMul计算后的Add计算，即公式中的输入x3。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>可选输入</td>
      <td>公式中的输入antiquantScale。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>可选输入</td>
      <td>公式中的输入antiquantOffset。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequantScale</td>
      <td>可选输入</td>
      <td>MatMul计算后的去量化系数，即公式中的输入dequantScale。</td>
      <td>FLOAT16、BFLOAT16、FLOAT、UINT64、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertoken_scale</td>
      <td>可选输入</td>
      <td>MatMul计算后的pertoken去量化系数，即公式中的输入pertoken_scale。</td>
      <td>FLOAT、BFLOAT16、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>comm_quant_scale_1</td>
      <td>可选输入</td>
      <td>matmulAdd计算后的per-channel量化系数，即公式中的输入comm_quant_scale_1。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>comm_quant_scale_2</td>
      <td>可选输入</td>
      <td>allGather计算后的per-channel量化系数，即公式中的输入comm_quant_scale_2。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>计算+通信的结果，即公式中的输出output。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><li>Host侧标识列组的字符串，通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduceOp</td>
      <td>可选属性</td>
      <td><li>reduce操作类型。</li><li>默认值为"sum"。</li></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>可选属性</td>
      <td><li>决定x1是否执行矩阵乘前进行转置。</li><li>默认值为false。</li></td>
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
      <td><li>通信数据切分数，即总数据量/单次通信量。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_group_size</td>
      <td>可选属性</td>
      <td>伪量化per-group模式下，对x2进行反量化计算的groupSize输入。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_size</td>
      <td>可选属性</td>
      <td><li>在输入张量 x1/x2 中，M、N、K 维度上的数值数量共同对应一个反量化系数。</li><li>默认值为0。</li><li>保留属性</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>comm_quant_mode</td>
      <td>可选属性</td>
      <td><li>静态量化和动态量化的标志位，数值为0和1。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
* 增量场景不使能MC2，全量场景使能MC2。
* 输入x1可为二维或者三维，其shape为(b, s, k)或者(m, k)，不支持非连续输入。
* 输入x2必须是二维。其shape为(k, n)，k轴满足mm算子入参要求，k轴相等，m的范围为[1, 2147483647]，k、n的范围为[1, 65535]。
* 输入x2的数据格式：
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持ND（当前版本仅支持二维输入）和FRACTAL_NZ格式（当前版本仅支持四维输入）。
    * 当x2的数据格式为FRACTAL_NZ时，配合aclnnCalculateMatmulWeightSizeV2和aclnnTransMatmulWeight到数据格式NZ的转换，非连续Tensor仅支持transpose场景。当x2的数据格式为ND时，当前版本仅支持二维输入。
* 传入的x1、x2、antiquantScale或者output不为空指针。
* 当输入x1的shape为(b, s, k)时，x3（非空场景）与输出output的shape为(b, s, n)，pertoken_scale的shape为(b*s)；当输入x1的shape为(m, k)时，x3（非空场景）与输出output的shape为(m, n)，pertoken_scale的shape为(m)。
* 输入comm_quant_scale_1和comm_quant_scale_2可选，可为空，当x2为(k, n)时, shape可为(n)或者(1,n)。
* 输入dequantScale可选，可为空，shape在pertensor场景为(1)，per-channel场景为(n)/(1, n)。输出为BFLOAT16时，直接将BFLOAT16类型的dequantScale传入本接口。输出为FLOAT16时，如果pertokenScale不为空，可直接将FLOAT32类型的dequantScale传入本接口，如果pertokenScale为空，则需提前调用TransQuantParamV2算子的aclnn接口来将dequantScale转成INT64/UINT64数据类型。
* bias若非空，当前版本仅支持一维，shape大小与output最后一维大小相等。antiquantScale在per-tensor场景下shape为(1)，在per-channel场景下shape为(1,n)/(n)，在per-group场景shape为(ceil(k,antiquantGroupSize), n)。antiquantOffset若非空，其shape与antiquantScale一致。
* x1和x2，x3（非空场景）、antiquantScale、antiquantOffset（非空场景）、output、bias（非空场景）的数据类型和数据格式需要在支持的范围之内。
* x1，antiquantScale，antiquantOffset（非空场景），x3（非空场景）、bias（非空场景）output的数据类型相同。antiquantGroupSize在不支持per-group场景时，传入0，在支持per-group场景时，传入值的范围为[32, min(k-1,INT_MAX)]，且为32的倍数。k取值范围与mm接口保持一致。
* group_size在perblock场景下，只支持549764202624。其他场景，只支持0。
* 只支持x2矩阵转置/不转置，x1矩阵不支持转置场景。
* 属性reduceOp当前版本仅支持输入"sum"。
* 属性commTurn当前版本仅支持输入0。
* 在长序列场景，随着b/s或者m的增大，可能出现OOM或者计算超时。
* 仅支持hccs链路all mesh组网。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持1、2、4、8卡。
* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：一个模型中的通算融合MC2算子，仅支持相同通信域。


## 调用说明


| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_all_reduce.cpp](./examples/test_aclnn_matmul_all_reduce.cpp) | 通过[aclnnMatMulAllReduce](docs/aclnnMatmulAllReduce.md)接口方式调用MatMulAllReduce算子。 |

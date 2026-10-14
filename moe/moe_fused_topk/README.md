# MoeFusedTopk

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
- **计算公式**：

  对输入做sigmoid：
  $$
  sigmoidRes=sigmoid(x)
  $$
  加上addNum：
  $$
  normOut = sigmoidRes + addNum
  $$
  对计算结果按照groupNum进行分组，每组按照topN的sum值对group进行排序，取前groupTopk个组：
  $$
  groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
  $$
  根据上一步的groupId获取normOut中对应的元素，将数据再做TopK，得到indices的结果：
  $$
  normY,indices=TopK(normOut[groupId, :],k=k)
  $$
  根据indices从sigmoidRes中选出y:
  $$
  y = gather(sigmoidRes, indices)
  $$
  如果isNorm为true，对y按照输入的scale参数进行计算，得到y的结果：
  $$
  y = y / (ReduceSum(y, dim=-1))*scale
  $$
  如果enableExpertMapping为true，再将indices中的物理专家按照输入的mappingNum和mappingTable映射到逻辑专家，得到输出的indices。

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>每个token对应各个专家的分数，对应公式中的`x`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>addNum</td>
        <td>输入</td>
        <td>与输入x进行计算的偏置值，对应公式中的`addNum`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mappingNum</td>
        <td>输入</td>
        <td>`enableExpertMapping`为false时不启用，每个物理专家被实际映射到的逻辑专家数量。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mappingTable</td>
        <td>输入</td>
        <td>`enableExpertMapping`为false时不启用，每个物理专家/逻辑专家映射表。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupNum</td>
        <td>属性</td>
        <td>分组数量。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupTopk</td>
        <td>属性</td>
        <td>被选择的组的数量。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topN</td>
        <td>属性</td>
        <td>组内选取的用于求和的专家数量。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>属性</td>
        <td>最终选取的专家数量。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activateType</td>
        <td>属性</td>
        <td>激活类型，当前只支持0(ACTIVATION_SIGMOID)。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNorm</td>
        <td>属性</td>
        <td>是否对输出进行归一化。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scale</td>
        <td>属性</td>
        <td>归一化后的系数乘。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>enableExpertMapping</td>
        <td>属性</td>
        <td>是否开启物理专家到逻辑专家的映射。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>输出每个token的topK分数，对应公式中的`y`。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输出</td>
        <td>topK个专家和tokens的映射关系，对应公式中的`indices`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- x和addNum数据类型必须一致。
- expertNum必须为groupNum的整数倍。
- groupTopk小于等于groupNum。
- maxMappingNum小于等于128。
- TopK小于等于expertNum。
- TopN小于等于expertNum / groupNum。
- expertNum小于等于1024。
- groupNum小于等于256。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_fused_topk](examples/test_aclnn_moe_fused_topk.cpp) | 通过[aclnnMoeFusedTopk](docs/aclnnMoeFusedTopk.md)接口方式调用MoeFusedTopk算子。 |

## 测试说明

ACLNN 回归用例统一放在 [ttk_aclnn_moe_fused_topk.csv](tests/st/arch35/ttk_aclnn_moe_fused_topk.csv)，使用 [golden.py](tests/assets/golden.py)。安装对应产品的算子包并配置 CANN 环境后，在 TTK 仓库目录执行（`OP_DIR` 指向本算子目录）：

```bash
python3 -m ttk aclnn -i "$OP_DIR/tests/st/arch35/ttk_aclnn_moe_fused_topk.csv" \
  --plugin "$OP_DIR/tests/assets/" --compare stat_rel_err --run 3 --dl 1
python3 -m ttk aclnn -i "$OP_DIR/tests/st/arch35/ttk_aclnn_moe_fused_topk.csv" \
  --plugin "$OP_DIR/tests/assets/" --ti 0-20 --compare binary --run 3 --dl 1
```

- 前 21 条覆盖 FP32、FP16、BF16 下的映射数量边界，包括 INT32 最小值、负数、零、有效数量、超过映射表宽度的数量和 INT32 最大值。Golden 为这些用例生成可区分的专家分数和映射表，映射数量小于等于零时使用首列，超过表宽时按表宽取模；第二条命令逐元素精确比较输出。
- 后 24 条覆盖 `y` 和 `indices` 的输出布局：14 条合法布局应通过精度和确定性检查，10 条元素重叠布局应在 `aclnnMoeFusedTopkGetWorkspaceSize` 返回 `ACLNN_ERR_PARAM_INVALID`（161002），且不进入二阶段执行。各用例的 `remark.expected_acl_status` 记录预期返回码。TTK 不会自动把该字段作为断言，预期拒绝用例在原生报告中显示为 N/A，需核对返回码；API 单元测试中的 `output_layouts` 同时断言这些布局的返回码。

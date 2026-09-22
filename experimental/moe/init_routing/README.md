# 算子名称：InitRouing

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                | 是       |

## 功能说明

- 算子功能：初始化路由，将输入token按expert分组并连续排列。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>blockDim</td>
      <td>输入</td>
      <td>AI CORE的数量，比如：Ascend910B是40。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
	<tr>
      <td>stream</td>
      <td>输入</td>
      <td>Device端的stream</td>
      <td>AclrtStream</td>
      <td>-</td>
    </tr>
	<tr>
      <td>in</td>
      <td>输入</td>
      <td>公式中的输入张量x，shape为(token_num, hidden_size)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>token_table</td>
      <td>输入</td>
      <td>token到expert的映射表， shape为(expert_num, token_num)</td>
      <td>int32_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>token_list</td>
      <td>输入</td>
      <td>每个expert累计token数量，shape为(expert_num)</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>out</td>
      <td>输出</td>
      <td>同一expert的token连续排列，shape为(expert_num*topk, hidden_size)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>expert_num</td>
      <td>输入</td>
      <td>expert数量</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>copy_byte</td>
      <td>输入</td>
      <td>每个token拷贝字节数，等于hidden_size * element_size</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
- token_list最后一个元素的值为总token数
- copy_byte按64bytes对齐

## 调用说明
torch.ops.npu_ops_transformer_ext.init_routing(block_dim, input, token_table,token_list, output)

# AttentionUpdate 

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

+ 算子功能：将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。
+ 计算公式：输入$lse_i$和$O_i$、输出$O$。

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">lse<sub>i</sub></td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">各SP域的局部lse。</td>
    <td class="tg-0pky">FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">O<sub>i</sub></td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">各SP域的局部attentionout。</td>
    <td class="tg-0pky">FLOAT32，FLOAT16，BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">lse<sub>m</sub></td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">更新后的全局lse。</td>
    <td class="tg-0pky">FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">O</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">更新后的全局attentionout。</td>
    <td class="tg-0pky">FLOAT32，FLOAT16，BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## 约束说明

* <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持FLOAT32、FLOAT16、BFLOAT16的O<sub>i</sub>和O。
* <term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT32、FLOAT16、BFLOAT16的O<sub>i</sub>和O，且O<sub>i</sub>和O数据类型相同。
* 序列并行的并行度sp取值范围[1, 16]。
* headDim取值范围[8, 512]且是8的倍数。
* 不支持非连续的Tensor。
* 支持空Tensor。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_attention_update](./examples/test_aclnn_attention_update.cpp) | 通过[aclnnAttentionUpdate](./docs/aclnnAttentionUpdate.md)接口方式调用AttentionUpdate算子。 |

# AttentionToFFN

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>       |    √     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将Attention节点上数据发往FFN节点。

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
    <td style="white-space: nowrap">x</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">本卡发送的token数据，3D Tensor，shape为 <code>(X, Bs, H)。</code></td>
    <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">sessionId</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">表示当前Attention Worker节点的Id，1D Tensor，shape为 <code>(X, )。</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">microBatchId</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">表示当前microBatch的Id，1D Tensor，shape为 <code>(X, )。</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">layerId</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">表示当前模型层数的Id，1D Tensor，shape为 <code>(X, )。</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertIds</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">每个micro batch组中每个token的topK个专家索引，3D Tensor，shape为 <code>(X, Bs, K)</code>，expertIds取值区间为[0, moeExpertNum)。</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertRankTable</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">每个micro batch组中专家Id到FFN卡专家部署的映射表（外部需保证值正确），3D Tensor，shape为 <code>(L, moeExpertNum + sharedExpertNum, M)。</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">scalesOptional</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">可选参数，表示每个专家的量化平滑参数，3D Tensor，shape为 <code>(L, moeExpertNum + sharedExpertNum, H)。</code></td>
    <td style="white-space: nowrap">FLOAT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">activeMaskOptional</td>
    <td style="white-space: nowrap">输入</td>
    <td style="white-space: nowrap">可选参数，表示token是否参与通信，可传有效数据或空指针，2D Tensor，shape为 <code>(X, Bs)。</code></td>
    <td style="white-space: nowrap">BOOL</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">group</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">通信域名称（专家并行），字符串长度[1, 128)。</td>
    <td style="white-space: nowrap">STRING</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">worldSize</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">通信域大小，取值区间[2, 768]。</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenInfoTableShape</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示FFN节点上token信息表格shape大小的列表，长度为3。</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenDataShape</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示FFN节点上token数据表格shape大小的列表，长度为5。</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">attnTokenInfoTableShape</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示Attention节点上token信息表格shape大小的列表，长度为3。</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">moeExpertNum</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">MoE专家数量，取值范围(0, 1024]。</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">quantMode</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示量化模式，支持0（非量化）、2（动态量化）。</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">syncFlag</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示FFN节点同步模式，支持0（同步）、1（异步）。</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnStartrankId</td>
    <td style="white-space: nowrap">属性</td>
    <td style="white-space: nowrap">表示FFN节点的起始Id。</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 调用算子过程中使用的`group`、`worldSize`、`tokenInfoTableShape`、`tokenDataShape`参数及`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致。

- 参数说明里shape格式说明：
    - `X`：表示micro batch sequence size（token组数），当前版本只支持`X` = 1。
    - `BS`：表示batch sequence size（本卡最终输出的token数量）取值范围为0 < `BS` ≤ 512。
    - `K`：表示选取topK个专家，，取值范围为0 < `K` ≤ 16且满足0 < `K` ≤ moeExpertNum。
    - `H`：表示hidden size（隐藏层大小），取值范围为1024 ≤ `H` ≤ 8192。
    - `L`：表示模型层数，当前版本只支持`L` = 1。
    - `M`：表示expertRankTable最后一维的长度，具体体现为部署在FFN节点上数量最多的专家部署信息列表的长度。
    - `moeExpertNum`：表示MoE专家数量，取值范围为0 < `moeExpertNum` ≤ 1024。

- 通信域使用约束：
    - AttentionToFFN算子的通信域中不允许有其他算子。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_attention_to_ffn.cpp](./examples/test_aclnn_attention_to_ffn.cpp) | 通过 [aclnnAttentionToFFN](./docs/aclnnAttentionToFFN.md)接口方式调用AttentionToFFN算子。 |

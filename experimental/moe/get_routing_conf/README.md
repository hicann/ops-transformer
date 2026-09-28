# 算子名称：GetRoutingConfigV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2训练系列产品                                         | 是       |

## 功能说明

- 算子功能：moeinitrouting计算前的数据准备工作。
- 计算公式：
  -

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
      <td>indices</td>
      <td>输入</td>
      <td>专家索引，shape(token, topk)</td>
      <td>int32_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>scores</td>
      <td>输入</td>
      <td>专家得分，shape(token, topk)</td>
      <td>bfloat16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>init_token_table</td>
      <td>输出</td>
      <td>当前卡上每个token对应的专家的token位置，shape为((expert, token))</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>final_token_table</td>
      <td>输出</td>
      <td>每条routing信息（token-expert）的编号， shape为(token, expert)</td>
      <td>int32</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>final_score_table</td>
      <td>输出</td>
      <td>每条routing信息的对应得分， shape(token, expert)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>token_idx_intra_expert</td>
      <td>输出</td>
      <td>中间结果，每条routing信息在其对应专家内部的编号，是initTokenTable的逆映射，shape(expert, token)</td>
      <td>int32</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>start_expert_id</td>
      <td>输入</td>
      <td>当前eprank上专家的起始id</td>
      <td>int64</td>
      <td>-</td>
    </tr>
	<tr>
      <td>end_expert_id</td>
      <td>输入</td>
      <td>当前eprank上最后专家的id的后一位id</td>
      <td>int64</td>
      <td>-</td>
    </tr>
  <tr>
      <td>local_expert_num</td>
      <td>输入</td>
      <td>每个eprank上的专家数</td>
      <td>int64</td>
      <td>-</td>
    </tr>
  <tr>
      <td>token_num</td>
      <td>输入</td>
      <td>所有eprank总token数</td>
      <td>int64</td>
      <td>-</td>
    </tr>
  <tr>
      <td>ub_max_token</td>
      <td>输入</td>
      <td>ub能存放的最大token, 1024</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入输出仅支持BFLOAT16类型。

## 价值/作用

为MoERouting计算前准备必要的数据。

## 设计方案

### Tilling策略

- 分核策略：
init_table：分割tokenNum
init_list：单核计算
init_other：分割localExpertNum，得到每个expert的专属token编号
final：分割tokenNum

### Kernel侧设计

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、数据搬出（CopyOut）三个阶段，实现RoutingConfig算子计算。

- 初始化（Init）
    - 计算loop_count
    - 建立GM Tensor映射，初始化队列缓冲区，初始化临时缓冲区
- 计算流程（Compute）
    - 调用CAST将输入转为float类型
    - 调用Ascendc命令完成计算
    - 调用CAST将fp32表示的结果转为bfloat16/half类型
- 数据搬入（CopyIn）搬出（CopyOut）
    - CopyIn: 将输入数据从inGM中搬入到inQueue
    - CopyOut: 将输出结果数据从outQueue搬出到outGM
- 流程调度（Process）
    - 遍历loop_count，按CopyIn->Compute->CopyOut的顺序调度

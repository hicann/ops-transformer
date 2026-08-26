# AttentionWorkerScheduler

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。

  **该算子不建议单独使用，建议与FFNToAttention和AttentionWorkerCombine算子配合使用，形成完整的工作流。**

    1. 接收FFNToAttention算子发送的数据。该数据以ScheduleContext结构体存储。该结构体包含CommonArea，ControlArea，AttentionArea，FfnArea域。本接口涉及CommonArea(用于存储配置信息，如session_num，micro_batch_num，micro_batch_size，selected_expert_num)，ControlArea(用于上层控制进程是否退出)，AttentionArea域(负责管理算子计算过程中所需的核心数据缓冲区与状态信息，其中token_info_buf存储了与输入相关的数据信息)。

    2. 读取ScheduleContext.AttentionArea域中token_info_buf存储的flag信息，查看通信数据是否准备就绪。

    3. 数据全部准备就绪后，后续可供AttentionWorkerCombine算子使用。

- 计算公式：

$$
\text{Initialize:} \quad \text{ready\_count} = 0, \quad \text{flag\_num} = \text{micro\_batch\_size} \times \text{selected\_expert\_num}
$$

$$
\text{Check if run\_flag is 0:}
\quad \text{if run\_flag} = 0, \quad \text{exit and log}
$$

$$
\text{Loop:} \quad \text{while run\_flag} \neq 0:
\quad \text{ready\_count} = \sum\_{i=1}^{\text{flag\_num}} \mathbf{1}\_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready\_count} = \text{flag\_num}, \quad \text{break}
$$

$$
\text{Reset flags:}
\quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag\_num}
$$

$$
\text{Set micro\_batch\_id:} \quad \text{micro\_batch\_id} = (\text{micro\_batch\_id} + 1) \% \text{micro\_batch\_num}
$$

备注：micro_batch_size、selected_expert_num、run_flag、micro_batch_id是入参ScheduleContext结构体的参数，该结构体信息在调用示例中进行展示说明。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>schedule_context</td>
      <td>输入</td>
      <td>Attention侧接收的待处理数据，表示输入ScheduleContext信息。不支持空Tensor，shape固定为(1024)。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>schedule_context</td>
      <td>输出</td>
      <td>处理后的ScheduleContext信息，shape、dtype和format与输入一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_inplace_attention_worker_scheduler](./examples/test_aclnn_inplace_attention_worker_scheduler.cpp) | 通过[aclnnInplaceAttentionWorkerScheduler](./docs/aclnnInplaceAttentionWorkerScheduler.md)接口方式调用AttentionWorkScheduler算子。 |

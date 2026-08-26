# FfnWorkerScheduler

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

- 算子功能：Attention和FFN分离场景下，FFN侧数据扫描算子。该算子接收AttentionToFFN算子发送的数据，进行扫描并完成数据整理。

    **不建议直接使用，需要与AttentionToFFN，FFNWorkerBatching配合使用。**

    1. 接收AttentionToFFN算子发送的数据。该数据以ScheduleContext结构体内存排布方式存储。该结构体包含CommonArea，ControlArea，AttentionArea，FfnArea域。本接口涉及CommonArea(用于存储配置信息，如session_num，micro_batch_num，micro_batch_size，selected_expert_num)，ControlArea(用于上层控制进程是否退出)，FfnArea域(负责管理本算子计算过程中所需的输入及输出缓冲区，其中token_info_buf字段用来存储该算子的输入信息)。

    2. 扫描token_info_buf存储的信息，当通信数据准备就绪时，本算子开始进行数据整理。整理如下图所示，将layer id， session id，micro batch id，expert ids分别写入layer_id_buf，session_id_buf，micro_batch_id_buf，expert_ids_buf的device内存上。

    ```mermaid
    graph TB
        %% 输入缓冲区
        A[token_info_buf输入]

        %% Session层级结构
        A --> Session0
        A --> Session1

        %% Session 0内部结构
        subgraph Session0[session 0]
            direction TB
            S0_M1[micro batch id 0]:::micro
            S0_L1[layer id 0]:::layer
            S0_S1[session id 0]:::session0
            S0_E1[expert ids 0]:::expert
        end

        %% Session 1内部结构
        subgraph Session1[session 1]
            direction TB
            S1_M1[micro batch id 0]:::micro
            S1_L1[layer id 0]:::layer
            S1_S1[session id 1]:::session1
            S1_E1[expert ids 0]:::expert
        end

        %% 输出缓冲区索引区域
        subgraph Output[输出区域]
            direction TB
            O1[layer_ids_buf]:::layer
            O2[session_ids_buf]:::output
            O3[micro_batch_ids_buf]:::micro
            O4[expert_ids_buf]:::expert
        end

        %% 数据流向
        S0_L1 -.-> O1
        S0_S1 -.-> O2
        S0_M1 -.-> O3
        S0_E1 -.-> O4

        S1_L1 -.-> O1
        S1_S1 -.-> O2
        S1_M1 -.-> O3
        S1_E1 -.-> O4

        classDef layer fill:#c8e6c9
        classDef session0 fill:#ffcdd2
        classDef session1 fill:#ffccbc
        classDef output fill:#e3f2fd
        classDef micro fill:#e1f5fe
        classDef expert fill:#bbdefd

        %% 添加子图背景色样式
        style Session0 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
        style Session1 fill:#fce4ec,stroke:#e91e63,stroke-width:2px
        style Output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    ```

    3. 完成数据整理后，后续可供FFNWorkerBatching算子使用。

- 计算公式：

1. 初始化，根据入参ScheduleContext中的session_num和sync_group_size计算分组个数。

 - 若分组个数为1，表示全同步处理数据，待全部session数据准备就绪后，进行数据整理。
 - 若分组个数不为1，表示非全同步处理数据，待group内的session数据准备就绪后，进行数据整理。

     $$
     \text{Initialize:} \quad\text{group\_num} = \frac{\text{session\_num}}{\text{sync\_group\_size}}
     $$

$$
\text{Process} =
\begin{cases}
\text{check\_all\_session\_ready()} \quad \text{data\_reorganization()} & \text{if } \text{group\_num} = 1 \\
\text{check\_all\_sessions\_of\_group\_ready()} \quad \text{data\_reorganization()} & \text{otherwise}
\end{cases}
$$

## 参数说明
  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 146px">
  <col style="width: 135px">
  <col style="width: 326px">
  <col style="width: 246px">
  <col style="width: 275px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>schedule_context</td>
      <td>输入</td>
      <td>FFN侧接收的待处理数据，表示ScheduleContext信息，详细结构参见调用示例</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>schedule_context</td>
      <td>输出</td>
      <td>处理后的ScheduleContext信息，shape、dtype和format与输入一致。</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sync_group_size</td>
      <td>属性</td>
      <td>每个同步组处理的session个数。</td>
      <td>取值范围为(0，session_num]，session_num表示待处理数据的最大会话数。</td>
      <td>Int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>execute_mode</td>
      <td>属性</td>
      <td>执行模式。</td>
      <td>只支持模式0，表示执行完一次退出。</td>
      <td>Int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_inplace_ffn_worker_scheduler](./examples/test_aclnn_inplace_ffn_worker_scheduler.cpp) | 通过[`aclnnInplaceFfnWorkerScheduler`](./docs/aclnnInplaceFfnWorkerScheduler.md)接口方式调用FfnWorkScheduler算子。 |

# FfnWorkerScheduler

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: A data scanning operator on the FFN side in the scenario where Attention and FFN are deployed separately. This operator receives the data sent by the AttentionToFFN operator, scans the data, and completes data reorganization.

    **This operator is not recommended for standalone use. It needs to be used together with the AttentionToFFN and FFNWorkerBatching operators.**

    1. Receives the data sent by the AttentionToFFN operator. The data is stored in the memory layout of the ScheduleContext structure. The structure contains the CommonArea, ControlArea, AttentionArea, and FfnArea fields. This interface involves CommonArea (used to store configuration information such as session_num, micro_batch_num, micro_batch_size, and selected_expert_num), ControlArea (used by upper-layer processes to control whether to exit), and the FfnArea field (responsible for managing the input and output buffers required during operator computation, where the token_info_buf field stores the input information of this operator).

    2. Scans the information stored in token_info_buf. When the communication data is ready, this operator starts data reorganization. The reorganization is shown in the following figure. Layer IDs, session IDs, micro batch IDs, and expert IDs are written to the device memory of layer_id_buf, session_id_buf, micro_batch_id_buf, and expert_ids_buf, respectively.

    ```mermaid
    graph TB
        %% Input buffer
        A[token_info_buf input]

        %% Session hierarchy
        A --> Session0
        A --> Session1

        %% Internal structure of Session 0
        subgraph Session0[session 0]
            direction TB
            S0_M1[micro batch id 0]:::micro
            S0_L1[layer id 0]:::layer
            S0_S1[session id 0]:::session0
            S0_E1[expert ids 0]:::expert
        end

        %% Internal structure of Session 1
        subgraph Session1[session 1]
            direction TB
            S1_M1[micro batch id 0]:::micro
            S1_L1[layer id 0]:::layer
            S1_S1[session id 1]:::session1
            S1_E1[expert ids 0]:::expert
        end

        %% Output buffer index area
        subgraph Output[Output Area]
            direction TB
            O1[layer_ids_buf]:::layer
            O2[session_ids_buf]:::output
            O3[micro_batch_ids_buf]:::micro
            O4[expert_ids_buf]:::expert
        end

        %% Data flow
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

        %% Add subgraph background color styles
        style Session0 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
        style Session1 fill:#fce4ec,stroke:#e91e63,stroke-width:2px
        style Output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    ```

    3. After data reorganization is complete, the data can be used by the FFNWorkerBatching operator.

- Computation formula:

1. Initialization: calculates the number of groups based on session_num and sync_group_size in the input parameter ScheduleContext.

 - If the number of groups is 1, data is processed in the fully synchronous mode. Data reorganization starts after the data of all sessions is ready.
 - If the number of groups is not 1, data is processed in the non-fully synchronous mode. Data reorganization starts after the data of sessions in a group is ready.

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

## Parameter Description
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
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>schedule_context</td>
      <td>Input</td>
      <td>Data to be processed received on the FFN side, indicating the ScheduleContext information. For the detailed structure, see the invocation example</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1D, shape (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>schedule_context</td>
      <td>Output</td>
      <td>Processed ScheduleContext information, with the same shape, dtype, and format as the input.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1D, shape (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sync_group_size</td>
      <td>Attribute</td>
      <td>Number of sessions processed by each synchronization group.</td>
      <td>Value range: (0, session_num]. session_num indicates the maximum number of sessions of the data to be processed, and session_num must be divisible by the synchronization group size.</td>
      <td>Int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>execute_mode</td>
      <td>Attribute</td>
      <td>Execution mode.</td>
      <td>Only mode 0 is supported, which indicates that the operator exits after one execution.</td>
      <td>Int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

None.

## Invocation Description

| Invocation method  | Sample code                                                                | Description                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn interface | [test_aclnn_inplace_ffn_worker_scheduler](./examples/test_aclnn_inplace_ffn_worker_scheduler.cpp) | Invokes the FfnWorkScheduler operator through the [`aclnnInplaceFfnWorkerScheduler`](docs/aclnnInplaceFfnWorkerScheduler_en.md) interface. |

# FfnWorkerBatching

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------------- | :-----------: |
|<term>Ascend 950PR/Ascend 950DT</term>| √ |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>| √ |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>| √ |
|<term>Atlas 200I/500 A2 推理产品</term>| × |
|<term>Atlas 推理系列产品</term>| × |
|<term>Atlas 训练系列产品</term>| × |

## 功能说明

- 算子功能：`FFNWorkerBatching`在Attention与FFN分离部署场景下，完成FFN worker上的token重排操作。Attention将token按专家路由发送到对应FFN worker的预分配数据区，FFNWorkerBatching从该数据区中扫描调度信息，按专家维度聚合并重排token，产出各专家对应的连续token数据块。

- 计算步骤：

    1. 对`expert_ids_in`中所有token的专家ID进行排序（被mask的token初始化为大值），生成gather索引。
    2. 多核并行按gather索引从`token_data`中提取token的hidden states和dynamic scale，同时查表得到对应的`session_id`、`micro_batch_id`、`token_id`。
    3. 单核扫描排序后的专家ID序列，查找跳变点，生成`group_list`（每个专家处理的token起止偏移）。

    其中 $Y = A \times BS \times (K+1)$，$A$ 为Attention worker数量，$BS$ 为micro batch size，$K+1$ 为topK加共享专家数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
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
      <td>调度上下文数据结构，内含CommonArea、ControlArea、AttentionArea、FfnArea。算子从FfnArea中读取token_info_buf和token_data_buf获取待重排的token数据与描述信息，并获取layer_id、session_id、micro_batch_id、expert_ids等路由信息。为一维Tensor，shape为 [1024]（固定1024字节结构体），不支持空Tensor。</td>
      <td>INT8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_num</td>
      <td>属性</td>
      <td>本卡专家总数，等于每层本卡专家数 × layer_num。用于推导group_list输出大小。取值范围为 (0, 8192]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_out_shape</td>
      <td>属性</td>
      <td>输出shape上限，格式为 {A, BS, topK+1, H}。用于推导y输出的shape上限Y = A × BS × (topK+1)，以及H值。长度必须为4。其中A（Attention worker数量）≤ 1024；第3维topK+1（topK加共享专家数）≤ 64；BS（micro batch size）> 0 且无硬上限（受内存限制）；H（hidden size）> 0，支持泛化。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>token_dtype</td>
      <td>属性</td>
      <td>输入token的数据类型，取值范围为 [0, 2]，默认值为0。决定单token搬运字节宽度及输出y的数据类型：<br>0：FP16，单token 2字节，y输出FP16；<br>1：BF16，单token 2字节，y输出BF16；<br>2：INT8动态量化，单token 1字节，y输出INT8。<br>取值0与1的计算路径一致（均按2字节原样搬运重排），仅y的浮点类型标记不同。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>need_schedule</td>
      <td>属性</td>
      <td>调度模式。0表示仅做batching不扫描数据；1表示先扫描数据再做batching。默认值为0。取值范围为 [0, 1]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layer_num</td>
      <td>属性</td>
      <td>层数，每层专家独立索引。默认值为0。取值范围为 [0, expert_num]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>重排后的token hidden states，按专家ID排序后连续存放。shape为 [Y, H]。数据类型由token_dtype决定：0为FP16、1为BF16、2为INT8。</td>
      <td>FP16、BF16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>输出</td>
      <td>每个专家处理的token范围，shape为 [expert_num, 2]。每行格式为 [expert_id, expert_token_num]，未使用的专家填 [0, 0]。示例：[[1, 20], [10, 40], [22, 15], ...]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>session_ids</td>
      <td>输出</td>
      <td>每个输出token对应的Attention session ID，shape为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>micro_batch_ids</td>
      <td>输出</td>
      <td>每个输出token对应的micro batch ID，shape为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>token_ids</td>
      <td>输出</td>
      <td>每个输出token在原始输入中的位置索引，shape为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_offsets</td>
      <td>输出</td>
      <td>每个输出token在其所属专家分组内的偏移，shape为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dynamic_scale</td>
      <td>输出</td>
      <td>动态量化的scale值，仅在token_dtype=2时有效并写出，shape为 [Y]。</td>
      <td>FP32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_token_num</td>
      <td>输出</td>
      <td>所有专家有效token数之和，标量输出。shape为 []。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

> 各参数（属性/输入/输出）自身的取值范围、shape 与格式约束见上文「参数说明」表；本节仅列出跨参数的全局约束。

- 该接口支持图模式（GEIR）和单算子模式（aclnn）。
- 参数M（micro batch数量，取自schedule_context.common.micro_batch_num）支持 ≤ 64。
- 输出token数上限 Y = A × BS × (topK+1)，其中 A、BS、topK+1 分别为 max_out_shape 的第 1、2、3 维；Y 支持泛化，无硬上限（受内存限制）。
- 精度口径为二进制一致（非计算类算子：整数索引排序 + 原样字节搬运，无浮点运算）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_ffn_worker_batching.cpp](examples/test_geir_ffn_worker_batching.cpp) | 通过[算子IR](op_graph/ffn_worker_batching_proto.h)构图方式调用FfnWorkerBatching算子。 |
| aclnn接口 | [test_aclnn_ffn_worker_batching.cpp](examples/test_aclnn_ffn_worker_batching.cpp) | 通过[aclnnFfnWorkerBatching](docs/aclnnFfnWorkerBatching.md)接口方式调用FfnWorkerBatching算子。 |

## 调度上下文数据结构

```cpp
struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;           // Attention节点数
    uint32_t micro_batch_num;       // micro batch拆分数量
    uint32_t micro_batch_size;      // batch_size / micro_batch_num
    uint32_t selected_expert_num;   // topK个数 + 1（含共享专家）
    uint32_t expert_num;            // 每层专家个数
    uint32_t attn_to_ffn_token_size;// 每个token在FFN window数据区占用大小，对齐到512
    uint32_t ffn_to_attn_token_size;// 每个token在Attention window数据区占用大小，对齐到512
    int32_t  schedule_mode;         // 0:只调度FFN, 1:只调度Attention, 2:同时调度
    int8_t   reserve0[96];          // padding to 128 bytes
  };
  struct ControlArea {
    int32_t run_flag;               // 控制循环退出
    int8_t  reserve1[124];
  };
  struct AttentionArea {
    uint64_t token_info_buf;        // [M, DataDesc]，DataDesc含flags[batch_size][topK+1]
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id;        // 轮询用，初始值micro_batch_num-1
    int8_t   reserve5[92];
  };
  struct FfnArea {
    // FFN输入区
    uint64_t token_info_buf;        // [A, M, F]，DataDesc含flag/layer_id/expert_ids
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [A, M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint64_t polling_index;
    int8_t   reserve3[88];
    // FFN输出区
    uint64_t layer_ids_buf;         // [session_num]
    uint64_t layer_ids_buf_size;
    uint64_t session_ids_buf;       // [session_num]
    uint64_t session_ids_buf_size;
    uint64_t micro_batch_ids_buf;   // [session_num]
    uint64_t micro_batch_ids_buf_size;
    uint64_t expert_ids_buf;        // [session_num, BS, K+1]
    uint64_t expert_ids_buf_size;
    uint32_t out_num;               // 实际收齐的session个数
    int8_t   reserve4[60];
  };
  CommonArea    common;
  ControlArea   control;
  AttentionArea attention;
  FfnArea       ffn;
  int8_t        reserve6[384];      // padding to 1024 bytes
}; // 总大小1024字节
```

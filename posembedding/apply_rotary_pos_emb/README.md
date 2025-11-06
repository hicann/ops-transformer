# ApplyRotaryPosEmb

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：执行旋转位置编码计算，推理网络为了提升性能，将query和key两路算子融合成一路。计算结果执行原地更新。
- 计算公式：
  （1）rotaryMode为"half"：
  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$
  
  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$
  
  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$
  （2）rotaryMode为"quarter"：
  $$
  query\_q1 = query[..., : query.shape[-1] // 4]
  $$
  
  $$
  query\_q2 = query[..., query.shape[-1] // 4 : query.shape[-1] // 2]
  $$

  $$
  query\_q3 = query[..., query.shape[-1] // 2 : query.shape[-1] // 4 * 3]
  $$

  $$
  query\_q4 = query[..., query.shape[-1] // 4 * 3 :]
  $$
  
  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1, -query\_q4, query\_q3), dim=-1)
  $$
  
  $$
  key\_q1 = key[..., : key.shape[-1] // 4]
  $$
  
  $$
  key\_q2 = key[..., key.shape[-1] // 4 : key.shape[-1] // 2]
  $$

  $$
  key\_q3 = key[..., key.shape[-1] // 2 : key.shape[-1] // 4 * 3]
  $$

  $$
  key\_q4 = key[..., key.shape[-1] // 4 * 3 :]
  $$
  
  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1, -key\_q4, key\_q3), dim=-1)
  $$
  
  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$
  （3）rotaryMode为"interleave"：
  $$
  query\_q1 = query[..., ::2].view(-1, 1)
  $$
  
  $$
  query\_q2 = query[..., 1::2].view(-1, 1)
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1).view(query.shape[0], query.shape[1], query.shape[2], query.shape[3])
  $$

  $$
  key\_q1 = key[..., ::2].view(-1, 1)
  $$
  
  $$
  key\_q2 = key[..., 1::2].view(-1, 1)
  $$

  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1), dim=-1).view(key.shape[0], key.shape[1], key.shape[2], key.shape[3])
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$
  
  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述            | 数据类型                   | 数据格式 |
|-----|----------|---------------|------------------------|------|
| query      | 输入输出 | 输入、输出tensor，即公式中的输入张量query和输出张量q_embed，4维张量 | BFLOAT16、FLOAT16、FLOAT | ND   |
| key        | 输入输出 | 输入、输出tensor，即公式中的输入张量key和输出张量k_embed，4维张量   | BFLOAT16、FLOAT16、FLOAT | ND   |
| cos        | 输入     | 公式中的输入张量cos，4维张量   | BFLOAT16、FLOAT16、FLOAT | ND   |
| sin        | 输入     | 公式中的输入张量sin，4维张量   | BFLOAT16、FLOAT16、FLOAT | ND   |
| layout     | 属性     | 表示输入张量的布局格式 | int64：1-BSND、2-SBND、3-BNSD | - |
| rotary_mode| 属性     | 公式中的旋转模式       | string："half"、"interleave"、"quarter"模式 | - |

- <term>Atlas 推理系列产品</term>：不支持BFLOAT16


## 约束说明

    - 输入张量query、key、cos、sin只支持4维的shape，layout只支持1-BSND，且4个输入shape的前2维和最后一维必须相等，cos和sin的shape第3维必须等于1，输入shape最后一维必须等于128。
    - 输入张量query、key、cos、sin的dtype必须相同。
    - 输入query的shape用（q_b, q_s, q_n, q_d）表示，key shape用（q_b, q_s, k_n, q_d）表示，cos和sin shape用（q_b, q_s, 1, q_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。

      - 当输入是BFLOAT16时，cast表示为1，castSize为4，DtypeSize为2
      - 当输入是FLOAT16或FLOAT32时，cast表示为0，castSize = DtypeSize（FLOAT16时为2，FLOAT32时为4）

      需要使用的UB空间大小计算方式：`ub_required = (q_n + k_n) * 128 * castSize * 2 + 128 * DtypeSize * 4 + (q_n + k_n) * 128 * castSize + (q_n + k_n) * 128 * castSize * 2 + cast * (128 * 4 * 2)`，
      当计算出`ub_required`的大小超过当前AI处理器的UB空间总大小时，不支持使用该融合算子。
    - rotary_mode只支持"half"。
    - 不支持空tensor场景。
  
    - 输入张量query、key、cos、sin只支持4维的shape，对于任意layout，query与key除N维度外其他维度必须相同；cos与sin shape必须相同；cos与sin的B维度与query、key的B维度一致，或者等于1；cos和sin的N维度必须等于1；query、key、cos、sin的S,D维度必须相同,且D维度小于等于1024。
    - 输入张量query、key、cos、sin的dtype必须相同。
    - rotary_mode为"half"和"interleave"时，输入shape最后一维必须被2整除；rotary_mode为"quarter"时，输入shape最后一维必须被4整除。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_rotary_pos_emb.cpp](examples/test_aclnn_apply_rotary_pos_emb.cpp) | 通过[aclnnApplyRotaryPosEmb](./docs/aclnnApplyRotaryPosEmb.md)接口方式调用ApplyRotaryPosEmb算子。    |
| aclnn调用 | [test_aclnn_apply_rotary_pos_emb_v2.cpp](examples/test_aclnn_apply_rotary_pos_emb_v2.cpp) | 通过[aclnnApplyRotaryPosEmbV2](./docs/aclnnApplyRotaryPosEmbV2.md)接口方式调用ApplyRotaryPosEmb算子。    |



# ApplyRotaryPosEmbGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：执行双路旋转位置编码[ApplyRotaryPosEmb](../apply_rotary_pos_emb/README.md)的反向计算。同时计算 **query** 和 **key** 的 rope 反向梯度，融合为一次 kernel 调用，节省 cos/sin 加载和 kernel launch 开销。

- **计算公式**：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，则计算公式可表达如下：

    （1）half模式：
    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|----------|---------------|------------------------|------|
| grad_query_embed | 输入 | 正向输出query的导数，对应公式中$grad\_q_{embed}$。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| grad_key_embed | 输入 | 正向输出key的导数，对应公式中$grad\_k_{embed}$。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| cos | 输入 | 正向计算输入cos，需与grad_query_embed数据类型一致。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| sin | 输入 | 正向计算输入sin，需与grad_query_embed数据类型一致。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| query | 可选输入 | 正向计算输入query。如果为空指针，则不计算grad_cos和grad_sin；必须与key同时传入或同时不传入。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| key | 可选输入 | 正向计算输入key。如果为空指针，则不计算grad_cos和grad_sin；必须与query同时传入或同时不传入。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| rotary_mode | 属性 | 旋转模式，仅支持"half"。 | STRING | - |
| layout | 属性 | 输入Tensor的布局格式。1=BSND，2=SBND，4=TND。默认值为1。 | INT64 | - |
| grad_query | 输出 | 正向计算输入query的导数，shape与grad_query_embed相同。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| grad_key | 输出 | 正向计算输入key的导数，shape与grad_key_embed相同。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| grad_cos | 输出 | 正向计算输入cos的导数，仅当query和key非空时有效。 | BFLOAT16、FLOAT16、FLOAT32 | ND |
| grad_sin | 输出 | 正向计算输入sin的导数，仅当query和key非空时有效。 | BFLOAT16、FLOAT16、FLOAT32 | ND |

## 约束说明

  - 各参数的约束描述如下：
    - 输入输出Tensor只支持3维或4维：layout为1或2时为4维，layout为4时为3维。
    - 输入输出Tensor的dtype必须相同。
    - 输入输出Tensor不支持空Tensor（各维度必须大于0）。
    - 输入输出Tensor的layout必须相同。
    - 输入输出Tensor的D轴必须相同，在half模式下必须≤1024且能被2整除。
    - `grad_query_embed`、`grad_query`的shape必须相同，`grad_key_embed`、`grad_key`的shape必须相同。
    - 对于任意`layout`，`grad_query_embed`和`grad_key_embed`除N维度外其它维度必须相同。
    - `cos`、`sin`的N维度必须等于1；`layout`为1（BSND）或2（SBND）时，`cos`、`sin`的B维度可以等于1，也可以和`grad_query_embed`的B维度一致；`layout`为4（TND）时，`cos`、`sin`的T维度必须和`grad_query_embed`的T维度一致；除N（及BSND、SBND布局下可选广播的B）维度外，其余维度需与`grad_query_embed`一致。
    - `cos`、`sin`、`grad_cos`、`grad_sin`的shape必须相同。
    - `query`维度需与`grad_query_embed`一致，`key`维度需与`grad_key_embed`一致，且`query`和`key`必须同时传入或同时不传入。
    - `rotary_mode`仅支持"half"。
    - `layout`仅支持{1, 2, 4}，对应{BSND, SBND, TND}。3=BNSD为预留，暂不支持。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_rotary_pos_emb_grad](./examples/test_aclnn_apply_rotary_pos_emb_grad.cpp) | 通过[aclnnApplyRotaryPosEmbGrad](./docs/aclnnApplyRotaryPosEmbGrad.md)接口方式调用ApplyRotaryPosEmbGrad算子。             |

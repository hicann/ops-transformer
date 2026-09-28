# QuantFlashAttnGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | ------ |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：计算量化注意力（QuantFlashAttention）的反向梯度。该算子为`aclnnQuantFlashAttention`正向算子的配套反向接口，用于计算Query、Key、Value的梯度（dq、dk、dv）以及sink梯度（dsink）。支持HIFLOAT8量化数据类型，支持BSND、BNSD、TND三种数据排布格式，支持全量计算（mask_mode=0）、因果mask（rightDownCausal，mask_mode=3）、滑窗mask（band，mask_mode=4）三种mask模式。TND排布下Q/K/V按变长序列打包存储，需通过cu_seqlens_q、cu_seqlens_kv描述各batch的序列边界，还可通过seqused_q、seqused_kv进一步指定各batch实际参与运算的序列长度；BSND、BNSD排布下可通过seqused_q、seqused_kv截断各batch的冗余运算。

- 前置依赖：调用本算子前必须先调用`quant_flash_attn_metadata`算子生成metadata并传入。metadata记录AICore/AIVCore的任务切分（分核调度）结果；在TND排布或传入seqused_q、seqused_kv的变长场景下，metadata还携带逐batch的调度信息（轮次前缀、每batch的S1/S2分块数、band右下角锚点token、调度类型等）。主算子与`quant_flash_attn_metadata`的入参需保持一致，否则会发生未定义行为。

- 计算公式：

    量化注意力反向梯度计算分为以下几个阶段：

    阶段一：反量化输入数据，将量化输入转换到高精度浮点域

    $$
    Q_{fp} = Q_{quant} \times q\_descale
    $$

    $$
    K_{fp} = K_{quant} \times k\_descale
    $$

    $$
    V_{fp} = V_{quant} \times v\_descale
    $$

    $$
    dO_{fp} = dO_{quant} \times do\_descale
    $$

    阶段二：计算Softmax梯度（PreSfmg阶段）

    $$
    dP = dO \times V
    $$

    $$
    dS = dP \times p\_scale
    $$

    $$
    dS_{grad} = (dS - ds\_scale \times softmax(lse)) \times softmaxLse
    $$

    阶段三：计算Key/Value梯度（Cube主计算阶段）

    $$
    dV = dS_{grad}^{T} \times Q
    $$

    $$
    dK = dS_{grad}^{T} \times Q
    $$

    $$
    dQ = dS_{grad} \times K
    $$

    阶段四：后处理（Post阶段），将中间结果转换输出dtype

    $$
    dq = dQ \times softmax\_scale
    $$

    $$
    dk = dK \times softmax\_scale
    $$

    $$
    dv = dV \times softmax\_scale
    $$

    $$
    dsink = reduce(dS_{grad}, dim=S)
    $$

    其中，$q\_descale$、$k\_descale$、$v\_descale$、$do\_descale$为量化缩放因子，$p\_scale$为P矩阵的缩放因子，$ds\_scale$为反量化缩放因子，$softmax\_scale$为缩放系数（建议值为$\sqrt{D}$的倒数）。

### 数据排布说明

Q、K、V数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim、T（Total-Token-Num）表示TND排布下所有batch的序列长度累加和。各排布下的shape约定如下：

| 排布格式 | q/dout/attn_out/dq | k/v/dk/dv | softmax_lse |
| -------- | ------------------ | --------- | ----------- |
| BSND | (B, S1, N1, D) | (B, S2, N2, D) | (B, N1, S1) |
| BNSD | (B, N1, S1, D) | (B, N2, S2, D) | (B, N1, S1) |
| TND | (T1, N1, D) | (T2, N2, D) | (N1, T1) |

TND排布说明：

- TND（Token-Num-Dim）为变长场景的打包排布：各batch的序列在T维上连续拼接，batch边界由cu_seqlens_q（K/V侧由cu_seqlens_kv）描述，即T1 = cu_seqlens_q[B]、T2 = cu_seqlens_kv[B]。
- batch数B由cu_seqlens_q（或cu_seqlens_kv）的长度减1推导，不单独传入。
- kernel按cu_seqlens_q/cu_seqlens_kv计算各batch在GM中的起始token偏移（q_start/kv_start）进行寻址，Q/K/V共享T维的连续存储；mask坐标始终为batch内局部坐标。
- TND下dq/dk/dv的shape与q/k/v一致，即dq为(T1, N1, D)，dk/dv为(T2, N2, D)；dsink的shape为(N1,)。

### 变长序列说明（cu_seqlens与seqused）

| 参数 | 语义 |
| ---- | ---- |
| cu_seqlens_q | 各batch中Query序列长度的累加和形式，INT32，shape为(B+1,)。首元素必须为0，非递减排列，末元素等于T1。 |
| cu_seqlens_kv | 各batch中Key/Value序列长度的累加和形式，INT32，shape为(B+1,)。首元素必须为0，非递减排列，末元素等于T2，长度需与cu_seqlens_q一致。 |
| seqused_q | 各batch中Query实际参与运算的序列长度（取对应cu_seqlens段或S1的前缀），INT32，shape为(B,)。 |
| seqused_kv | 各batch中Key/Value实际参与运算的序列长度（取对应cu_seqlens段或S2的前缀），INT32，shape为(B,)。 |

- cu_seqlens_q、cu_seqlens_kv仅在TND排布下有效：TND排布下必须传入；BSND/BNSD排布下不支持传入。
- seqused_q、seqused_kv在TND与BSND/BNSD排布下均为可选：TND排布下其值需小于等于对应batch的cu_seqlens段长度；BSND/BNSD排布下其值需小于等于S1/S2。
- 传入seqused_q、seqused_kv后，每batch仅前seqused个token参与计算，用于截断冗余运算：BSND/BNSD排布下跳过各batch的padding部分；TND排布下与cu_seqlens联用（cu_seqlens定义打包边界，seqused定义实际有效长度）。
- TND排布或传入seqused_q、seqused_kv时走变长（varlen）路径：PreSfmg阶段按有效长度截断加载dO/O（TND且传入seqused_q时按effective_q = seqused_q[b]截断、按global_q = cu_seqlens_q[b] + local_q寻址）；Cube阶段的分块数（s1_outer/s2_outer）与mask边界均按各batch实际有效长度计算，逐batch信息由metadata下发。
- cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv属于tensor输入，tiling阶段无法获取具体数值，不做值校验，正确性需用户自行保证；传入非法值会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### mask_mode说明

| mask_mode取值 | 名称 | 语义 |
| -------------- | ---- | ---- |
| 0 | No mask（ALL_MASK） | 全量计算，不施加mask。attn_mask必须为空。 |
| 3 | rightDownCausal | 以右下角为基准的下三角mask：query第i行仅attend满足j ≤ i + (S2 - S1)的key。S1 = S2时即标准causal（下三角）；S1 > S2时前S1 - S2行query整体位于因果区外，对应冗余分块自动跳过。 |
| 4 | band（Sliding Window） | 以右下角为基准的滑窗mask：将query/key按右下角对齐后，query第i行仅attend与其距离落在[-win_right, win_left]区间内的key。S1 ≠ S2时host对win_left/win_right做S1 - S2校正；变长（TND/seqused）场景下按各batch实际序列长度在metadata中逐batch计算锚点。 |

- mask_mode=3或4时必须传入attn_mask；mask_mode=0时不允许传入attn_mask。
- attn_mask为INT8掩码模板，shape固定为(2048, 2048)，典型构造方式为下三角模板（如`torch.tril(torch.ones(2048, 2048, dtype=torch.int8))`）。kernel按当前分块（tile）相对于掩码边界的偏移从模板中截取对应窗口：causal场景对齐因果线取一窗；band场景分别对齐滑窗右/左边界截取两个窗口并取差集得到滑窗区域。
- win_left/win_right仅在mask_mode=4时可设置为非-1值，其余mask_mode下必须保持默认值-1；取值需大于等于-1，-1表示该方向不限窗（host按无穷大处理），其余负值暂不支持。
- mask_mode=3/4为稀疏调度：与mask区域不相交的分块整块跳过，边界分块施加元素级mask；该跳过逻辑对BSND/BNSD定长与TND/seqused变长场景均生效。

### kernel计算流说明

算子整体计算流与上述四个阶段对应，变长与mask相关流程如下：

- PreSfmg阶段（Vector核）：反量化dout/attn_out并计算Softmax梯度。变长场景下先对sfmg及dk/dv workspace清零初始化（稀疏调度可能重访同一head/KV列，变长场景需为未访问列补零）。
- Cube主循环阶段（Cube核）：按确定性调度遍历(batch, head, s1_block, s2_block)。变长场景的调度由metadata下发（BSND/BNSD+seqused为dense swizzle，TND为line swizzle反对角调度），保证dk/dv跨核唯一归属与确定性累加顺序；稀疏场景（mask_mode=3/4）按mask区域做分块级跳过。
- Post阶段（Vector核）：将workspace中的累加结果写出为dq/dk/dv（BF16），并计算dsink（FLOAT）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| q | 输入 | attention结构的输入Q，量化数据。shape：BSND (B, S1, N1, D)、BNSD (B, N1, S1, D)、TND (T1, N1, D)。 | HIFLOAT8 | ND |
| k | 输入 | attention结构的输入K，量化数据。shape：BSND (B, S2, N2, D)、BNSD (B, N2, S2, D)、TND (T2, N2, D)。 | HIFLOAT8 | ND |
| v | 输入 | attention结构的输入V，量化数据。shape与k一致。 | HIFLOAT8 | ND |
| dout | 输入 | 正向输出attn_out对应的梯度，量化数据。shape与q一致。 | HIFLOAT8 | ND |
| attn_out | 输入 | 正向计算输出的attn_out。shape与q一致。 | BF16 | ND |
| q_descale | 输入 | Query的反量化缩放因子，每个量化块对应一个缩放因子。 | FLOAT | ND |
| k_descale | 输入 | Key的反量化缩放因子，每个量化块对应一个缩放因子。 | FLOAT | ND |
| v_descale | 输入 | Value的反量化缩放因子，shape需与k_descale一致。 | FLOAT | ND |
| do_descale | 输入 | dout的反量化缩放因子，shape需与q_descale一致。 | FLOAT | ND |
| p_scale | 输入 | P矩阵的量化缩放因子。 | FLOAT | ND |
| ds_scale | 输入 | 反量化缩放因子。 | FLOAT | ND |
| softmax_lse | 输入 | 注意力正向计算的输出softmaxLse。shape：BSND/BNSD (B, N1, S1)、TND (N1, T1)。 | FLOAT | ND |
| cu_seqlens_q | 可选输入 | Query的累积序列长度，用于TND变长序列的batch边界划分。INT32，shape为(B+1,)，首元素必须为0，非递减排列，末元素等于T1。仅TND排布支持且必须传入，非TND排布不支持传入。 | INT32 | ND |
| cu_seqlens_kv | 可选输入 | Key/Value的累积序列长度，用于TND变长序列的batch边界划分。INT32，shape为(B+1,)，首元素必须为0，非递减排列，末元素等于T2，长度需与cu_seqlens_q一致。仅TND排布支持且必须传入，非TND排布不支持传入。 | INT32 | ND |
| seqused_q | 可选输入 | 各batch中q实际参与运算的序列长度，用于截断冗余运算。INT32，shape为(B,)，值为非负整数。TND排布下需小于等于对应batch的cu_seqlens_q段长度；BSND/BNSD排布下需小于等于S1。 | INT32 | ND |
| seqused_kv | 可选输入 | 各batch中k/v实际参与运算的序列长度，用于截断冗余运算。INT32，shape为(B,)，值为非负整数。TND排布下需小于等于对应batch的cu_seqlens_kv段长度；BSND/BNSD排布下需小于等于S2。 | INT32 | ND |
| sinks | 可选输入 | sink场景下的输入tensor。当前版本不支持非空输入，需传空tensor。 | FLOAT | ND |
| attn_mask | 可选输入 | 注意力mask模板，INT8，shape固定为(2048, 2048)。mask_mode=3/4时必须传入，mask_mode=0时不允许传入。 | INT8 | ND |
| metadata | 可选输入 | tiling下沉的aicpu算子（quant_flash_attn_metadata）输出结果。当前版本必须传入，且必须由quant_flash_attn_metadata生成。INT32，shape为(2, max_schedule_size)，第二维不小于：TND排布17+10×B；非TND传入seqused_q/seqused_kv时17+3×B（mask_mode=3/4时为17+10×B）；其他场景16。 | INT32 | ND |
| dq | 输出 | Query的梯度，dtype固定为BF16，shape与q一致。 | BF16 | ND |
| dk | 输出 | Key的梯度，dtype固定为BF16，shape与k一致。 | BF16 | ND |
| dv | 输出 | Value的梯度，dtype固定为BF16，shape与v一致。 | BF16 | ND |
| dsink | 输出 | sink的梯度，dtype固定为FLOAT，shape为(N1,)。 | FLOAT | ND |
| quant_mode | 必选属性 | 量化模式。0：HIFLOAT8 per-tensor量化。 | INT64 | - |
| softmax_scale | 可选属性 | 缩放系数，默认值为1.0。推荐值：sqrt(head_dim)的倒数。 | FLOAT | - |
| mask_mode | 可选属性 | 表示q和k计算的mask模式。0：No mask（全量计算，attn_mask需为空）；3：rightDownCausal（以右下角为基准的下三角causal）；4：band（滑窗，需配合win_left/win_right使用）。默认值为0。详见mask_mode说明。 | INT64 | - |
| win_left | 可选属性 | 滑窗mask的左窗口宽度，仅mask_mode=4时有效，-1表示该方向不限窗，其余负值不支持；mask_mode不为4时必须为-1。默认值为-1。 | INT64 | - |
| win_right | 可选属性 | 滑窗mask的右窗口宽度，仅mask_mode=4时有效，-1表示该方向不限窗，其余负值不支持；mask_mode不为4时必须为-1。默认值为-1。 | INT64 | - |
| max_seqlen_q | 可选属性 | Query的最大序列长度，仅TND排布支持传入大于0的值，-1表示自动推导；非TND排布仅支持-1。默认值为-1。 | INT64 | - |
| max_seqlen_kv | 可选属性 | Key/Value的最大序列长度，仅TND排布支持传入大于0的值，-1表示自动推导；非TND排布仅支持-1。默认值为-1。 | INT64 | - |
| layout_q | 可选属性 | 表示输入q的数据排布格式，支持"BSND"、"BNSD"、"TND"，默认值为"BSND"。layout_q与layout_kv必须保持一致。 | STRING | - |
| layout_kv | 可选属性 | 表示输入k/v的数据排布格式，支持"BSND"、"BNSD"、"TND"，默认值为"BSND"。layout_q与layout_kv必须保持一致。 | STRING | - |

## 约束说明

- 确定性说明：aclnnQuantFlashAttnGrad默认确定性实现。
- layout约束：
    - 支持BSND、BNSD或TND layout，且layout_q与layout_kv必须保持一致。
    - TND排布下q/k/v/dout/attn_out均为3维tensor；BSND/BNSD排布下均为4维tensor。
- 关于数据shape的约束：
    - B：BSND/BNSD排布下支持0 < B < 65536；TND排布下B由cu_seqlens_q（或cu_seqlens_kv）长度减1推导，要求B ≥ 1且B ≤ T1、B ≤ T2。
    - S1、S2：泛化支持，支持S1、S2不等长；TND排布下各batch序列长度可不等长（由cu_seqlens逐batch划分）。
    - N1：支持1~128。BSND/BNSD排布下需满足GQA约束（N1 = N2 × G，num_heads_q必须能被num_heads_k整除）；TND排布下不支持GQA，要求N1 = N2。
    - N2：支持1~128。
    - D：量化场景下固定为128。
- 量化场景下，q、k、v、dout、attn_out的维度必须一致，k和v的shape必须一致。
- q_descale和do_descale的shape必须一致，k_descale和v_descale的shape必须一致。
- TND排布约束：
    - 仅支持quant_mode=0，mask_mode仅支持0/3/4。
    - cu_seqlens_q、cu_seqlens_kv必须传入（INT32，shape为(B+1,)，首元素为0，非递减，末元素分别等于T1/T2，两者长度一致）；BSND/BNSD排布下不支持传入cu_seqlens_q、cu_seqlens_kv。
    - softmax_lse的shape为(N1, T1)；dout、attn_out的shape必须与q完全一致。
    - max_seqlen_q/max_seqlen_kv仅TND排布支持传入大于0的值（-1表示自动推导），非TND排布仅支持-1。
- seqused约束：
    - INT32，shape为(B,)，值为非负整数。
    - TND排布下seqused_q/seqused_kv的值需小于等于对应batch的cu_seqlens段长度；BSND/BNSD排布下需小于等于S1/S2。
    - 传入seqused_q/seqused_kv时走变长路径，metadata必须由quant_flash_attn_metadata生成并传入，且与主算子的入参保持一致。
- mask_mode支持：
    - 0：不做mask操作（ALL_MASK），attn_mask必须为空。
    - 3：rightDownCausal模式的mask，以右下角为基准划分下三角场景，attn_mask不能为空。
    - 4：band模式的mask，以右下角为基准划分滑窗场景，需配合win_left和win_right属性使用，attn_mask不能为空。
    - win_left/win_right仅在mask_mode=4时可设置非-1值，取值需大于等于-1（-1表示该方向不限窗）；mask_mode不为4时必须保持-1。
- metadata约束：
    - 当前版本metadata必须传入，且必须由quant_flash_attn_metadata生成，shape为(2, max_schedule_size)。
    - 第二维不小于：17 + 10×B（TND排布）；17 + 3×B（BSND/BNSD传入seqused_q/seqused_kv且mask_mode=0）；17 + 10×B（BSND/BNSD传入seqused_q/seqused_kv且mask_mode=3/4）；16（其他场景）。
- cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、attn_mask的值在tiling阶段不校验，正确性需用户自行保证，传入非法值会触发未定义行为。
- 空tensor（shape size为0）场景将全部拦截报错。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| PyTorch API | - | 通过[quant_flash_attn_grad](../../attention/quant_flash_attn_grad/docs/torchapi_quant_flash_attn_grad.md)接口调用算子。 |

调用流程：

1. 准备q、k、v等输入（TND排布下同时准备cu_seqlens_q/cu_seqlens_kv，按需准备seqused_q/seqused_kv）。
2. 调用`quant_flash_attn_metadata`生成metadata（反向场景需设置is_grad_enabled=True，且入参与主算子保持一致）。
3. 调用`quant_flash_attn_grad`，将metadata传入主算子；mask_mode=3/4时还需传入attn_mask模板。

## 参考资源（可选）

无

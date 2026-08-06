# QuantSparseFlashMla

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantSparseFlashMla 算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention：

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|q|输入|公式中的$Q$，不支持非连续，layout_q为BSND时shape为(batch_size, seqlen_q, nheads_q, headdim_q)，layout_q为TND时shape为(total_q, nheads_q, headdim_q)。|HIFLOAT8|ND|
|ori_kv|可选输入|公式中的$\tilde{K}$和$\tilde{V}$的一部分，为原始不经压缩的KV，支持非连续，layout_kv为PA_BBND时shape为(block_num_orikv, block_size_orikv, nheads_kv, headdim_kv)|HIFLOAT8|ND|
|cmp_kv|可选输入|公式中的$\tilde{K}$和$\tilde{V}$的一部分，为经过压缩的KV，支持非连续，layout_kv为PA_BBND时shape为(block_num_cmpkv, block_size_cmpkv, nheads_kv, headdim_kv)|HIFLOAT8|ND|
|q_descale|输入|输入q的反量化值，shape固定为(1,)。|FLOAT|ND|
|ori_kv_descale|可选输入|输入ori_kv的反量化值，shape固定为(1,)。|FLOAT|ND|
|cmp_kv_descale|可选输入|输入cmp_kv的反量化值，shape固定为(1,)。|FLOAT|ND|
|ori_sparse_indices|可选输入|预留参数，当前不生效，代表离散取oriKvCache的索引，不支持非连续，layout_q为BSND时shape为(batch_size, seqlen_q, nheads_kv, topk_orikv)，layout_q为TND时shape为(total_q, nheads_kv, topk_orikv)|INT32|ND|
|cmp_sparse_indices|可选输入|代表离散取cmpKvCache的索引，不支持非连续，layout_q为BSND时shape为(batch_size, seqlen_q, nheads_kv, topk_cmpkv)，layout_q为TND时shape为(total_q, nheads_kv, topk_cmpkv)|INT32|ND|
|ori_block_table|可选输入|PageAttention中oriKvCache存储使用的block映射表，shape约束见下方约束说明|INT32|ND|
|cmp_block_table|可选输入|PageAttention中cmpKvCache存储使用的block映射表，shape约束见下方约束说明|INT32|ND|
|cu_seqlens_q|可选输入|表示当前Batch及前序Batch中q的有效token数的累加和，维度为B+1，仅layout_q为TND场景需传入|INT32|ND|
|cu_seqlens_ori_kv|可选输入|表示当前Batch及前序Batch中ori_kv的有效token数的累加和，维度为B+1，仅layout_kv为TND场景需传入|INT32|ND|
|cu_seqlens_cmp_kv|可选输入|表示当前Batch及前序Batch中cmp_kv的有效token数的累加和，维度为B+1，仅layout_kv为TND场景需传入|INT32|ND|
|seqused_q|可选输入|表示不同Batch中q的真实使用长度，维度为B|INT32|ND|
|seqused_ori_kv|可选输入|表示不同Batch中ori_kv的真实使用长度，维度为B|INT32|ND|
|seqused_cmp_kv|可选输入|表示不同Batch中cmp_kv的真实使用长度，维度为B|INT32|ND|
|cmp_residual_kv|可选输入|cmp_mask_mode=3且cmp_ratio≠1时必须传入，表示每个Batch的余数，用于精确计算mask边界|INT32|ND|
|ori_topk_length|可选输入|预留参数，当前不生效|INT32|ND|
|cmp_topk_length|可选输入|预留参数，当前不生效|INT32|ND|
|sinks|可选输入|注意力下沉tensor|FLOAT32|ND|
|metadata|可选输入|aicpu算子（npu_quant_sparse_flash_mla_metadata）的分核结果，shape固定为[1024]|INT32|ND|
|qkv_quant_mode|可选属性|默认值为1，表示Q、K、V 的量化模式，当前仅支持1，1表示Q、K、V 输入为HIFLOAT8类型的per_tensor量化。|INT32|-|
|softmax_scale|可选属性|默认值为None，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值|FLOAT32|-|
|cmp_ratio|可选属性|默认值为None，表示对ori_kv的压缩率，kv压缩场景支持1~128，非kv压缩场景仅支持传1|INT32|-|
|ori_mask_mode|可选属性|默认值为0，表示q和ori_kv计算的mask模式，当前仅支持4，代表band模式的mask，需与ori_win_left和ori_win_right配合使用|INT32|-|
|cmp_mask_mode|可选属性|默认值为0，表示q和cmp_kv计算的mask模式，当前仅支持3，代表rightDownCausal模式的mask，对应以右下顶点为划分的下三角场景|INT32|-|
|ori_win_left|可选属性|默认值为-1，表示q和ori_kv计算中q对过去token计算的数量，当前仅支持127|INT32|-|
|ori_win_right|可选属性|默认值为-1，表示q和ori_kv计算中q对未来token计算的数量，当前仅支持0|INT32|-|
|layout_q|可选属性|默认值为BSND，用于标识输入q的数据排布格式，支持BSND和TND|STRING|-|
|layout_kv|可选属性|默认值为BSND，用于标识输入ori_kv和cmp_kv的数据排布格式，支持BSND、TND、PA_BBND|STRING|-|
|topk_value_mode|可选属性|默认值为1，当前仅支持1|INT32|-|
|return_softmax_lse|可选属性|默认值为False，预留参数，当前暂不支持，表示是否返回softmax_lse。True表示返回，False表示不返回|BOOL|-|
|attention_out|输出|当layout_q为BSND时shape为(batch_size, seqlen_q, nheads_q, headdim_q)，当layout_q为TND时shape为(total_q, nheads_q, headdim_q)|BFLOAT16|ND|
|softmax_lse|可选输出|输出q乘k的结果先取max得到softmax_max，q乘k的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。当layout_q为BSND时shape为(batch_size, nheads_kv, seqlen_q, nheads_q/nheads_kv)，当layout_q为TND时shape为(nheads_kv, total_q, nheads_q/nheads_kv)。目前softmax_lse输出为无效值|FLOAT32|ND|

## 约束说明

-   该接口支持aclgraph模式。
-   参数q、ori_kv和cmp_kv中的D仅支持512。
-   参数ori\_kv、cmp\_kv的数据类型必须保持一致，且不可同时为空。
-   参数q中的N1当前支持64/128，ori\_kv、cmp\_kv中的KV\_N仅支持1。
-   参数ori\_kv和cmp\_kv中的block\_size1和block\_size2需为16的倍数，最大支持1024；block\_num1及block_num2为PageAttention时block总数。
-   参数ori\_sparse\_indices与cmp\_sparse\_indices中的K1与K2为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，当前不支持传入ori\_sparse\_indices，cmp\_sparse\_indices中K2仅支持512/1024。
-   参数cu\_seqlens\_q、cu\_seqlens\_ori\_kv及cu\_seqlens\_cmp\_kv维度为B + 1，要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
-   参数seqused\_q维度为B，要求其值表示每个Batch中的有效token数。
-   参数seqused\_ori\_kv、seqused\_cmp\_kv维度为B，表示每个Batch中ori_kv和cmp_kv的真实使用长度。若不传入，seqused\_ori\_kv默认使用全部长度，seqused\_cmp\_kv默认根据seqused\_ori\_kv和cmp\_ratio计算（floor(seqused\_ori\_kv / cmp\_ratio)）。
-   参数cmp\_residual\_kv维度为B，当cmp\_mask\_mode=3且cmp\_ratio≠1时必须传入，用于解决mask计算中的余数问题，防止越界访问。
-   若同时传入seqused\_ori\_kv、seqused\_cmp\_kv和cmp\_residual\_kv，需满足：seqused\_ori\_kv[i] == seqused\_cmp\_kv[i] * cmp\_ratio + cmp\_residual\_kv[i]，且cmp\_residual\_kv[i] < cmp\_ratio。
-   参数ori\_block\_table的shape为2维，其中第一维长度为B，第二维长度不小于所有Batch中最大的S2对应的block数量，即S2\_max / block\_size1向上取整。
-   参数cmp\_block\_table的shape为2维，其中第一维长度为B，第二维长度不小于floor(S2\_max \/ cmp\_ratio)对应的block数量，即floor(S2\_max \/ cmp\_ratio) \/ block\_size2向上取整。
-   ori\_mask\_mode及cmp\_mask\_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/sparse_mode_introduction.md)。
-   q、ori_kv、cmp_kv参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
-   Q\_S和S1表示q shape中的S，S2表示ori\_kv shape中的S，Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori\_kv\_heads和num\_cmp\_kv\_heads；T1表示q shape中的T。

## 调用说明

-   调用方式：使用npu_ops_tranformer包中的npu_quant_sparse_flash_mla接口进行调用，
             详见torch_extension/npu_ops_transformer/ops/quant_sparse_flash_mla.py
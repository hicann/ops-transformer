# ChunkKdaFwd

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：完成不涉及CP切分的KDA分块正向计算，计算注意力输出、最终状态以及可选的反向中间量。

- 计算公式：

  将每条序列按$C=chunk\_size$划分为$M$个chunk。以第$c$个chunk为例，$i$、$j$为chunk内token下标，$l_c$为该chunk最后一个有效token下标。公式省略batch和head下标；GQA场景中，每个Value head使用其对应的Query/Key head。

  令$x_{c,i,d}=g_{c,i,d}+dt\_bias_d$，未传入`dt_bias`时令$dt\_bias_d=0$。激活后gate为：

  $$
  \gamma_{c,i,d}=
  \begin{cases}
  g_{c,i,d}, & use\_gate\_in\_kernel=false,\\
  -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
      & use\_gate\_in\_kernel=true,\ safe\_gate=false,\\
  lower\_bound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
      & use\_gate\_in\_kernel=true,\ safe\_gate=true.
  \end{cases}
  $$

  `gk`为chunk-local的log2累计gate：

  $$
  gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d}.
  $$

  `qg`和`kg`分别为gate缩放后的Query和Key：

  $$
  qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
  kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
  $$

  对同一chunk内的token $i$、$j$，`Aqk`为包含对角线的下三角Query-Key系数矩阵：

  $$
  Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
  \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
  $$

  严格下三角矩阵$L_c$及`Akk`为：

  $$
  L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
  \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
  Akk_c=(I+L_c)^{-1}.
  $$

  令$w^{seed}$和$u^{seed}$为：

  $$
  w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
  u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e}.
  $$

  `w`和`u`为：

  $$
  w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
  $$

  令$h_c$为第$c$个chunk计算前的状态；传入`initial_state`时$h_0=initial\_state$，否则$h_0=0$。`v_new`和下一个chunk状态为：

  $$
  v^{new}_c=u_c-w_c\,h_c,
  $$

  $$
  h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
  $$

  其中$2^{gk_{c,l_c}}\odot h_c$表示按$K$维缩放状态矩阵。`h`输出保存各个chunk的输入状态，`final_state`为最后一个chunk更新后的状态：

  $$
  h[c]=h_c,\qquad final\_state=h_M.
  $$

  `attn_out`为：

  $$
  attn\_out_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
  $$

  Python接口的第12项返回值为输入对象透传：

  $$
  initial\_state\_out=initial\_state.
  $$

  可选输出参数只决定是否向调用方返回对应中间量，不改变上述计算语义。

## 参数说明

本文使用以下维度符号：

- B：输入样本batch大小。
- S：rank-4输入的序列长度。
- T：rank-3输入的总token数。
- H：Query和Key的head数。
- HV：Value和gate的head数。
- K：Query和Key的head dim。
- V：Value的head dim。
- N：逻辑序列数。
- C：chunk大小。
- NC：未传入`cu_seqlens`时为每条序列的chunk数$\lceil S/C\rceil$；传入`cu_seqlens`时为所有逻辑序列的chunk总数。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| q | 输入 | 公式中的Query。`BSND`为(B, S, H, K)，`BNSD`为(B, H, S, K)，`TND`为(T, H, K)，`NTD`为(H, T, K)。 | FLOAT16、BFLOAT16 | ND |
| k | 输入 | 公式中的Key。shape和数据类型必须与`q`相同。 | FLOAT16、BFLOAT16 | ND |
| v | 输入 | 公式中的Value。`BSND`为(B, S, HV, V)，`BNSD`为(B, HV, S, V)，`TND`为(T, HV, V)，`NTD`为(HV, T, V)。数据类型必须与`q`相同。 | FLOAT16、BFLOAT16 | ND |
| g | 输入 | raw gate或已激活的自然对数gate。`BSND`为(B, S, HV, K)，`BNSD`为(B, HV, S, K)，`TND`为(T, HV, K)，`NTD`为(HV, T, K)。 | FLOAT、BFLOAT16 | ND |
| beta | 输入 | 公式中的Delta系数。`BSND`为(B, S, HV)，`BNSD`为(B, HV, S)，`TND`为(T, HV)，`NTD`为(HV, T)。 | FLOAT、BFLOAT16 | ND |
| a_log | 可选输入 | gate衰减参数，shape为(HV)。当`use_gate_in_kernel=true`时必须传入。 | FLOAT | ND |
| dt_bias | 可选输入 | gate偏置，shape为(HV×K)。 | FLOAT | ND |
| initial_state | 可选输入 | 初始状态。`state_v_first=false`时shape为(N, HV, K, V)，否则为(N, HV, V, K)。 | FLOAT | ND |
| cu_seqlens | 可选输入 | 变长序列累计长度，shape为(N+1)，首元素为0，末元素为T或S，元素单调不减。 | INT64 | ND |
| chunk_indices | 可选输入 | chunk索引，shape为(2×NC)，按`(seq_id, chunk_id)`保存，必须采用sequence-major canonical顺序。传入时必须同时传入`cu_seqlens`。 | INT64 | ND |
| attn_out | 输出 | 注意力输出。rank-4输入固定输出BSND格式(B, S, HV, V)；rank-3输入固定输出TND格式(T, HV, V)。 | FLOAT16、BFLOAT16 | ND |
| final_state | 可选输出 | 最终状态。`state_v_first=false`时shape为(N, HV, K, V)，否则为(N, HV, V, K)。 | FLOAT | ND |
| gk | 可选输出 | chunk-local log2累计gate。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)。 | FLOAT | ND |
| Aqk | 输出 | chunk内Query-Key系数矩阵。rank-4输入为(B, HV, S, C)，rank-3输入为(HV, T, C)。 | FLOAT16、BFLOAT16 | ND |
| Akk | 输出 | chunk内下三角求逆结果。rank-4输入为(B, HV, S, C)，rank-3输入为(HV, T, C)。 | FLOAT16、BFLOAT16 | ND |
| w | 可选输出 | 供反向使用的W中间量。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)。 | FLOAT16、BFLOAT16 | ND |
| u | 可选输出 | 供反向使用的U中间量。rank-4输入为(B, HV, S, V)，rank-3输入为(HV, T, V)。 | FLOAT16、BFLOAT16 | ND |
| qg | 可选输出 | 供反向使用的gate缩放Query。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)。 | FLOAT16、BFLOAT16 | ND |
| kg | 可选输出 | 供反向使用的gate缩放Key。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)。 | FLOAT16、BFLOAT16 | ND |
| v_new | 可选输出 | 供反向使用的Value中间量。rank-4输入为(B, HV, S, V)，rank-3输入为(HV, T, V)。 | FLOAT16、BFLOAT16 | ND |
| h | 可选输出 | 公开chunk状态。rank-4输入为(B, NC, HV, K, V)，rank-3输入为(NC, HV, K, V)；`state_v_first=true`时交换末两维。 | FLOAT16、BFLOAT16 | ND |
| layout | 可选属性 | 输入布局。支持`BSND`、`BNSD`、`TND`、`NTD`，默认值为`BSND`。该属性只描述输入布局。 | STRING | - |
| scale | 必选属性 | Query缩放系数。通常取$K^{-0.5}$。 | FLOAT | - |
| chunk_size | 必选属性 | chunk大小。支持64、128，默认值为64。 | INT | - |
| safe_gate | 必选属性 | 是否使用有界gate。默认值为false。 | BOOL | - |
| lower_bound | 可选属性 | 有界gate下界。`safe_gate=true`时取值范围为[-5, 0)，默认值为-5.0。 | FLOAT | - |
| use_gate_in_kernel | 必选属性 | 是否在kernel内由raw gate计算激活。默认值为false。 | BOOL | - |
| state_v_first | 可选属性 | 是否将状态张量末两维排列为(V, K)。默认值为false。 | BOOL | - |

## Python返回策略

`cann_ops_transformer.ops.chunk_kda_fwd`固定返回12项。每一项的返回条件如下：

| 返回项 | 返回条件 |
| :--- | :--- |
| attn_out | 始终返回。 |
| final_state | `output_final_state=true`时返回，否则为`None`。 |
| gk | `use_gate_in_kernel=false`或`disable_recompute=true`时返回，否则为`None`。 |
| Aqk | 始终返回。 |
| Akk | 始终返回。 |
| w | `disable_recompute=true`时返回，否则为`None`。 |
| u | `disable_recompute=true`时返回，否则为`None`。 |
| qg | `disable_recompute=true`时返回，否则为`None`。 |
| kg | `disable_recompute=true`时返回，否则为`None`。 |
| v_new | `disable_recompute=true`时返回，否则为`None`。 |
| h | `disable_recompute=true`或`return_intermediate_states=true`时返回，否则为`None`。 |
| initial_state | 原样返回Python输入`initial_state`，不是aclnn输出。 |

aclnn接口不接收`output_final_state`、`disable_recompute`或`return_intermediate_states`。L2只根据每个可选输出指针是否为空决定是否导出该输出。

## 约束说明

- H和HV必须为正整数，且满足HV≥H、HV≤128、HV % H=0。
- K和V取值范围为[16, 256]，且必须为16的倍数。
- `chunk_size`仅支持64和128。
- `q`、`k`、`v`的数据类型必须相同。
- `use_gate_in_kernel=true`时，`a_log`必须传入；`dt_bias`可以不传入。
- `safe_gate=true`且`use_gate_in_kernel=true`时，`lower_bound`取值范围为[-5, 0)。
- rank-4变长输入要求B=1。
- 变长输入最多支持1024条逻辑序列。
- `layout`只描述输入。`attn_out`固定为sequence-major；供反向继续计算的中间输出固定为head-major；`final_state`和公开`h`按参数表约定输出。
- 默认支持确定性计算。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [aclnnChunkKdaFwd](./docs/aclnnChunkKdaFwd.md) | 通过`aclnnChunkKdaFwdGetWorkspaceSize`和`aclnnChunkKdaFwd`调用。 |
| pytest | [ChunkKdaFwd测试框架](./tests/pytest/README.md) | 覆盖功能、跨布局一致性、确定性和性能验证。 |

阶段划分、平台模板和流水设计见[ChunkKdaFwd算子设计介绍](./docs/ChunkKdaFwd算子设计介绍.md)。

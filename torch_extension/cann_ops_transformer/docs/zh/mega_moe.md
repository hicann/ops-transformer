# mega_moe

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    Mega MoE算子 将 MoE 层的专家 FFN 的完整计算流程及前后数据通信（即 Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。
    该算子提供了 mega_moe 与 get_symm_buffer_for_mega_moe等接口，这些接口需配套使用。

    - get_symm_buffer_for_mega_moe：需与mega_moe配套使用，用于封装输入参数并创建SymmBuffer结构体，生成`context`、`ep_world_size`和`ccl_buffer_size`等mega_moe算子运行所需信息。

- 计算公式：
  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：激活矩阵，对应入参 `x`。$\text{total\_num\_tokens}$ 是全局总 token 数，$\text{hidden}$ 是隐藏层维度。
    - $\mathbf{E} \in \mathbb{Z}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家编号矩阵，对应入参 `topk_ids`。$\text{num\_topk}$ 是每个 token 选择的专家数量。
    - $\mathbf{G} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家的门控权重矩阵，对应入参 `topk_weights`。
    - $\mathbf{W}_1 \in \mathbb{R}^{\text{num\_experts} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$：Linear1 的权重矩阵，对应入参 `l1_weights`。$\text{num\_experts}$ 是专家数量，$\text{intermediate\_hidden}$ 是中间层维度。
    - $\mathbf{W}_2 \in \mathbb{R}^{\text{num\_experts} \times \text{intermediate\_hidden} \times \text{hidden}}$：Linear2 的权重矩阵，对应入参 `l2_weights`。
  - 输出：

    - $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：最终输出矩阵，对应出参 `y`。
  - 约定：
    - $⋅$ 表示矩阵乘法，$⊙$ 表示逐元素乘法。
    - $\left \lfloor z\right \rceil$ 表示将 $z$ 四舍五入到最近的整数，$\left \lfloor z\right \rfloor$ 表示将 $z$ 向下取整。
    - $|z|$ 表示取绝对值，$\max(z)$ 表示取最大值。
    - 全体 token 的集合为 $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{total\_num\_tokens} - 1\} \}$。
    - $\text{token}_i$ 的 token 表示（即隐藏状态向量）为 $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$，且 $\mathbf{x}_i = \mathbf{X}[i,:]$。
    - $\text{token}_i$ 的专家索引为 $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{num\_topk} - 1\},\quad e_{i,k} \in \{0,\dots,\text{num\_experts} - 1\}$。
    - 激活函数 $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}$，其中 $\sigma$ 为 Sigmoid 函数。
    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$。其中 $\mathbb{Z}_8^{\text{sym}}$ 的上标 $\text{sym}$ 表示对称量化值域区间：其值域关于 $-127$ 与 $127$ 对称取整，与标准 INT8 的 $[-128, 127]$ 值域不同，故以 $\text{sym}$ 上标区分。
    - 张量切片操作采用Python风格的 `start:stop:step` 表示法，例如$[0::2, :]$ 代表取偶数行、$[1::2, :]$ 代表取奇数行。
    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ 表示二进制重解释操作，将张量$\mathbf{Z}$的底层二进制数据按目标类型 $T$ 重新解释。
- 计算过程

    各产品支持的**Linear计算时**的激活矩阵A和权重矩阵W的数据类型如下：

    | 场景名    |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN、FLOAT8_E5M2 |FLOAT8_E4M3FN、FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持上表中A8W8-FP场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：不支持上表中A16W16 、A8W8-INT、A8W4-INT场景。

    <details>
    <summary> A16W16 非量化场景</summary>

    - **EP Dispatch**

        在 Dispatch 阶段，每个 $\text{token}_i$ 将其token表示 $\mathbf{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数，则有 $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的token表示 $\mathbf{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$。该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。整个计算过程分为如下三个子步骤。

        **1. Linear1 投影**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。其计算公式为
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}\bigr)
        $$
        **2. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e\bigl[:,\;\text{:intermediate\_hidden}\bigr], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e\bigl[:,\;\text{intermediate\_hidden:}\bigr]
        $$
        $$
        \mathbf{A}_e = \text{SiLU}\bigl(\mathbf{H}_{\text{gate}}\bigr) \;\odot\; \mathbf{H}_{\text{up}}
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{intermediate\_hidden}}\bigr)
        $$

        **3. Linear2 投影**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。
    - **Token Combine**

        Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。
    </details>

    <details>
    <summary> A8W8-INT 量化场景</summary>

    - **输入**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `l1_weights_sf`。
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `l2_weights_sf`。

    - **EP Dispatch**

        在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\mathbf{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到 INT8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$（$e_{i,k} = \mathbf{E}[i,k]$）。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 SwiGLU 结构以提升表达能力。在A8W8场景下，两个线性层都使用 INT8 输入和 INT8 权重进行矩阵乘，得到 INT32 中间结果并反量化。具体分为三个子步骤。

        **1. Linear1 投影（INT8 矩阵乘 + 反量化）**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。计算时执行 INT8 矩阵乘法，得到 INT32 计算结果：
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$
        然后反量化为预激活值 $\mathbf{H}_e$：
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        **2. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:, :\text{intermediate\_hidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:, \text{intermediate\_hidden}:]
        $$
        $$
        \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
        $$

        **3. Linear2 投影（量化 + INT8 矩阵乘 + 反量化）**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        在 A8W8 场景下，需要将激活值量化为 INT8，因此先对 $\mathbf{A}_e$ 的每一行（每个 token）计算缩放因子：
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        得到专家 $e$ 在 Linear2 计算时的激活缩放因子 $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$。然后量化：
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$
        再执行 INT8 矩阵乘法并反量化：
        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

    - **Token Combine**

        Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。

    </details>

    <details>
    <summary> A8W4-INT 量化场景</summary>

    - **输入**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `l1_weights_sf`。
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `l2_weights_sf`。
        - $\mathbf{B}_1 \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 的偏置矩阵，由 INT4 量化过程离线生成，对应入参 `l1_bias`。
        - $\mathbf{B}_2 \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 的偏置矩阵，由 INT4 量化过程离线生成，对应入参 `l2_bias`。

    - **EP Dispatch**

        在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\mathbf{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到 INT8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。在A8W4-INT场景下，两个线性层都采用MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案进行矩阵乘，通过将 INT8 值拆为高4位和低4位两个有符号 INT4，使得 INT8×INT4 的矩阵乘可分解为两个 INT4×INT4 的矩阵乘，从而利用硬件的 INT4 矩阵乘加速。该方案的数学原理和实现逻辑可参阅[GroupedMatmul W4A8量化与MSD方案](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81)。

        **1. 生成精度补偿的偏置矩阵（离线生成，在算子外完成，并作为算子输入）**

        MSD 方案的核心是将量化后的 INT8 激活值二进制重解释地拆分为两个 INT4 分量。由于低位 INT4 分量按 $(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$ 定义，其数值范围被映射到 $[-8, 7]$，这相当于在原始无符号低 4 位（0~15）的基础上减去了 8。当将高、低位的 INT4 分别与权重矩阵做矩阵乘并合并时，该偏移会引入一个常数项，需要在最终结果中予以补偿。

        记原始 INT8 激活矩阵为 $\mathbf{X}^{\text{int8}}$，拆分后的高位和低位 INT4 激活矩阵分别为 $\mathbf{X}_1^{\text{int4}}$ 和 $\mathbf{X}_2^{\text{int4}}$，权重矩阵为 $\mathbf{W}$。恢复关系为：

        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        其中 $\mathbf{1}_{\text{mat}}$ 为与 $\mathbf{X}_2^{\text{int4}}$ 形状相同的全 1 矩阵，用于逐元素加 8。令 $\mathbf{1}$ 为形状与输入特征维度一致的全 1 列向量，则矩阵乘展开为：

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        这里 $\mathbf{1}^\top \cdot \mathbf{W}$ 表示对权重矩阵 $\mathbf{W}$ 的每一列求和，其结果为一个行向量（形状与输出维度相同）。在具体实现中，该行向量会被广播到批次维度，加到所有 token 的输出上。可见，若只计算前两项，会缺失一个正数项。为了精确恢复结果，需预先计算补偿偏置：

        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        $\mathbf{B}_1$ 和 $\mathbf{B}_2$ 均按此法，分别使用对应的全 1 列向量（维度分别为 $\text{hidden}$ 和 $\text{intermediate\_hidden}$）与各自的权重矩阵相乘，得到形状匹配的偏置行向量。这些偏置在离线阶段计算完成后传入算子，在合并高低位结果时直接参与加法，从而消除偏移引入的误差。

        **2. Linear1 投影（INT4 × INT4 矩阵乘 + 反量化）**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。

        **(1) 激活重解释为INT4**

        将 INT8 激活张量 $\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 二进制重解释为两个 INT4 交替拼接的视图：

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        其中高位和低位分量直接由 $\mathbf{Q}_e^{\text{int4}}$ 的偶数行和奇数行给出：

        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        它们在数值上可以由原 INT8 经如下计算得到：

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        恢复关系为 $\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\mathbf{Q}_e^{\text{int4}}$ 与 $\mathbf{Q}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

        **(2) INT4 × INT4 矩阵乘与权重反量化**

        将重解释后的 INT4 激活视图 $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ 与权重矩阵 $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediate\_hidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W1}_e$ 进行反量化，得到结果：

        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        将结果按偶数行和奇数行分别记为如下的矩阵视图，它们即对应于$\mathbf{Q}_e^{\text{high}}$和$\mathbf{Q}_e^{\text{low}}$的计算结果：

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        **(3) 精度补偿与激活反量化**

        利用偏置 $\mathbf{B}_1[e]$ 和激活缩放因子 $\mathbf{s}^{X}_e$（行向量）分别进行精度补偿和激活反量化，最终预激活值为：

        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        其中 $\odot \mathbf{s}^{X}_e$ 表示将矩阵的每一行乘以 $\mathbf{s}^{X}_e$ 中对应的标量。

        **3. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:,\; :\!\text{intermediate\_hidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:,\; \text{intermediate\_hidden}:]
        $$
        $$
        \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
        \quad\in \mathbb{R}^{N_e \times \text{intermediate\_hidden}}
        $$

        **4. Linear2 投影（量化 + INT4 × INT4 矩阵乘 + 反量化）**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        在 A8W4 场景下，首先将激活值量化为 INT8，再通过二进制重解释为两个 INT4 的拼接视图，以一次矩阵乘完成计算，过程如下。

        **(1) 激活量化**

        对 $\mathbf{A}_e$ 的每一行计算缩放因子并量化至 INT8：
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        得到专家 $e$ 在 Linear2 计算时的激活缩放因子 $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$，并计算量化结果：
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$

        **(2) 激活重解释为 INT4**

        将 INT8 激活张量 $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}$ 二进制重解释为两个 INT4 交替拼接的视图：

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}
        $$

        其中高位和低位分量直接由 $\mathbf{A}_e^{\text{int4}}$ 的偶数行和奇数行给出：

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}
        $$

        它们在数值上可以由原 INT8 经如下计算得到：

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        恢复关系为 $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\mathbf{A}_e^{\text{int4}}$ 与 $\mathbf{A}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

        **(3) INT4 × INT4 矩阵乘与权重反量化**

        将重解释后的 INT4 激活视图 $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}$ 与权重矩阵 $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediate\_hidden} \times \text{hidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W2}_e$ 进行反量化，得到结果：

        $$
        \mathbf{D}_e = \bigl( \mathbf{A}_e^{\text{int4}} \cdot \mathbf{W}_2[e] \bigr) \odot \mathbf{s}^{W2}_e
        \;\in\; \mathbb{R}^{2N_e \times \text{hidden}}
        $$

        将结果按偶数行和奇数行分别记为如下的矩阵视图，它们即对应于 $\mathbf{A}_e^{\text{high}}$ 和 $\mathbf{A}_e^{\text{low}}$ 的计算结果：

        $$
        \mathbf{D}_e^{\text{high}} = \mathbf{D}_e[0::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}, \qquad
        \mathbf{D}_e^{\text{low}}  = \mathbf{D}_e[1::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        **(4) 精度补偿与激活反量化**

        利用偏置 $\mathbf{B}_2[e]$ 和激活缩放因子 $\mathbf{s}^{A_e}_e$（行向量）分别进行精度补偿和激活反量化，最终输出为：

        $$
        \mathbf{Y}_e = \Bigl( 16 \cdot \mathbf{D}_e^{\text{high}} + \mathbf{D}_e^{\text{low}} + \mathbf{B}_2[e] \Bigr) \odot \mathbf{s}^{A_e}_e
        \quad\in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        其中 $\odot \mathbf{s}^{A_e}_e$ 表示将矩阵的每一行乘以 $\mathbf{s}^{A_e}_e$ 中对应的标量。

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

    - **Token Combine**

        Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。

    </details>

    <details>
    <summary> A8W8-FP 量化场景</summary>

    第一阶段对输入 Token 按专家分组收集后做 MXFP8 量化，生成各专家的量化输入与缩放因子：

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    说明：根据 `topkIds` 将 Token 按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的 Token 索引集合，$E_{local}$表示当前专家收到的最大token数，每个专家数值可能不同，$X[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示 MX 逐组量化（group size = 32），对每组 32 个元素提取共享指数后量化为 FP8 目标类型（FLOAT8_E5M2 或 FLOAT8_E4M3FN），同时输出 FLOAT8_E8M0 缩放因子。量化后的数据将作为 GMM1 的输入。

    第二阶段对每个专家执行 GMM1 矩阵乘法（将 $W_1$ 沿列方向分为两半分别计算）、SwiGLU 激活和 MX 量化：

    $$
    Z_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(x)}, S_{1,e}^{(x)}), \quad Z_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(y)}, S_{1,e}^{(y)})
    $$

    $$
    U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
    $$

    $$
    \hat{U}_e,\ S_{U,e} = \mathrm{Q}_{\text{MX}}(U_e)
    $$

    说明：将 $W_1$ 的前 $N/2$ 列 $W_{1,e}^{(x)}$ 和后 $N/2$ 列 $W_{1,e}^{(y)}$ 分别与 MX 反量化后的输入做矩阵乘法，得到 Swish 分支 $Z_e^{(x)}$ 和门控分支 $Z_e^{(y)}$。SwiGLU 激活对两个分支做逐元素乘积 $x \cdot \sigma(x) \cdot y$，其中 $\sigma$ 为 Sigmoid 函数，将中间维度从 $N$ 减半为 $N/2$。随后对 SwiGLU 输出做 MX 量化，得到 GMM2 的量化输入 $\hat{U}_e$。

    第三阶段对每个专家执行 GMM2 矩阵乘法，并将结果按目标 Rank 分发：

    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    说明：将量化后的 SwiGLU 输出与第二组权重 $W_2$ 做 MX 反量化后的矩阵乘法，将 $N/2$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $O_e$。计算完成后通过 RDMA peermem 将结果按目标 Rank 的专家偏移地址写入远端，实现跨 Rank 聚合。

    第四阶段对所有 Token 按路由权重加权求和，恢复为与输入相同形状的输出：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
    $$

    说明：对每个 Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按 `topkWeights` 中的权重逐元素加权累加，得到最终输出 $Y$。

    其中，$X$ 表示参数 `x`，$W$ 表示参数 `topkWeights`，$W_1$ 表示参数 `weight1`，$W_2$ 表示参数 `weight2`，$Y$ 表示参数 `y`，$E_{\text{local}}$ 表示属性 `moeExpertNum / epWorldSize`（每个 Rank 的专家数），$K$ 表示 `topkIds` 的第二维度（top-K 值，取值 6 或 8）。

    局部变量说明：
    - $\mathcal{T}_e$：被路由到专家 $e$ 的 Token 索引集合，由 `topkIds` 排序后确定。
    - $\hat{X}_e,\ S_{X,e}$：专家 $e$ 的量化输入及其 MX 缩放因子，第一阶段中间结果。
    - $W_{1,e}^{(x)}$、$W_{1,e}^{(y)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由 `weight1` 按 SwiGLU 拆分推导。
    - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$：$W_{1,e}^{(x)}$ 和 $W_{1,e}^{(y)}$ 对应的 MX 缩放因子，从 `weightScales1` 按维度截取。
    - $S_{2,e}$：$W_{2,e}$ 对应的 MX 缩放因子，来自参数 `weightScales2`。
    - $Z_e^{(x)},\ Z_e^{(y)}$：GMM1 的两路矩阵乘法输出（Swish 分支和门控分支），中间结果。
    - $U_e$：SwiGLU 激活输出，维度 $m_e \times N/2$，中间结果。
    - $\hat{U}_e,\ S_{U,e}$：量化后的 SwiGLU 输出及其 MX 缩放因子，中间结果。
    - $O_e$：GMM2 的专家级输出，维度 $m_e \times H$，中间结果。
    - $\pi(i, k)$：Token $i$ 的第 $k$ 个 top-k 专家在展开排序后的行索引，由路由排序确定。
    - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX 逐组量化操作，block size = 32，输出 FP8 数据和 E8M0 缩放因子。
    - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX 逐组反量化操作，在 matmul 内部隐式执行。
    </details>

## 函数原型

- get_symm_buffer_for_mega_moe：

```python
get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, *, max_recv_token_num=0, dispatch_quant_mode=0, dispatch_quant_out_dtype=None, combine_quant_mode=0, comm_alg="") -> SymmBuffer
```

- mega_moe：

```python
mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, weight1_type=None, weight2_type=None) -> (Tensor, Tensor)
```

## 参数说明

### get_symm_buffer_for_mega_moe

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>str</td>
        <td>必选</td>
        <td>EP通信域名称（专家并行通信域）。</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE模型的总专家数量。</td>
    </tr>
    <tr>
        <td>num_max_tokens_per_rank</td>
        <td>int</td>
        <td>必选</td>
        <td>每张卡上的token数量，当每个rank的num_tokens不同时，为最大的num_tokens大小。</td>
    </tr>
    <tr>
        <td>num_topk</td>
        <td>int</td>
        <td>必选</td>
        <td>每个token发送的专家数。</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>每个token大小。</td>
    </tr>
    <tr>
        <td>intermediate_hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>中间层投影维度。</td>
    </tr>
    <tr>
        <td>max_recv_token_num</td>
        <td>int</td>
        <td>可选</td>
        <td>每个Rank最大可接收Token数，默认值为0表示自动计算。默认值为0。</td>
    </tr>
    <tr>
        <td>dispatch_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>dispatch通信时量化模式。0表示非量化（A16W16场景），2表示INT8量化（A8W8-INT、A8W4-INT场景），4表示MXFP量化（A8W8-FP场景）。各产品支持的取值见约束说明。默认值为0。</td>
    </tr>
    <tr>
        <td>dispatch_quant_out_dtype</td>
        <td>torch.dtype</td>
        <td>可选</td>
        <td>dispatch量化后输出的数据类型。支持 torch.int8、torch.float8_e5m2、torch.float8_e4m3fn。各产品支持的取值见约束说明。默认值为None。</td>
    </tr>
    <tr>
        <td>combine_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>combine通信时的量化模式。0表示非量化，3表示MXFP float8_e5m2类型，4表示MXFP float8_e4m3类型。各产品支持的取值见约束说明。默认值为0。</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为""。</td>
    </tr>
</tbody>
</table>

### mega_moe

上角标<sup>1</sup>表示Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持，上角标<sup>2</sup>表示Ascend 950PR/Ascend 950DT不支持，产品不支持的参数使用默认值即可。

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 320px">
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 260px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MoE层输入的token隐藏状态。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>topk_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>专家索引矩阵，表示每个token选择的num_topk个专家。元素取值范围为<code>[0, num_experts)</code>，且同一token选择的num_topk个专家不能重复。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示MoE模型的专家门控网络为当前输入Token选出的num_topk个专家所对应的门控权重系数。</td>
        <td>BFLOAT16<sup>2</sup>、FLOAT32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td rowspan="5">l1_weights</td>
        <td rowspan="5">list[Tensor]</td>
        <td rowspan="5">必选</td>
        <td rowspan="5">专家网络第一线性层的权重矩阵（包括门控与上投影），用于将输入映射至中间维度，输出供给激活函数。</td>
        <td>BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>(hidden, 2 × intermediate_hidden)</td>
    </tr>
    <tr>
        <td>INT8<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(hidden, 2 × intermediate_hidden)</td>
    </tr>
    <tr>
        <td>INT4(INT32)<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(hidden, 2 × intermediate_hidden // 8)</td>
    </tr>
    <tr>
        <td>FLOAT8_E5M2<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td rowspan="5">l2_weights</td>
        <td rowspan="5">list[Tensor]</td>
        <td rowspan="5">必选</td>
        <td rowspan="5">专家网络第二线性层的权重矩阵，负责将激活后的中间特征投影回隐藏维度。数据类型与l1_weights一致。</td>
        <td>BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>INT8<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>INT4<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E5M2<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>必选</td>
        <td>由<a href="#get_symm_buffer_for_mega_moe">get_symm_buffer_for_mega_moe</a>接口创建的结构体</td>
        <td>SymmBuffer</td>
        <td>SymmBuffer</td>
        <td>SymmBuffer</td>
    </tr>
    <tr>
        <td rowspan="2">l1_weights_sf</td>
        <td rowspan="2">list[Tensor]</td>
        <td rowspan="2">可选</td>
        <td rowspan="2">专家网络第一线性层的权重矩阵的量化缩放因子。</td>
        <td>UINT64<sup>2</sup></td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)</td>
    </tr>
    <tr>
        <td rowspan="2">l2_weights_sf</td>
        <td rowspan="2">list[Tensor]</td>
        <td rowspan="2">可选</td>
        <td rowspan="2">专家网络第二线性层的权重矩阵的量化缩放因子。</td>
        <td>UINT64<sup>2</sup></td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>l1_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>专家网络第一线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>l2_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>专家网络第二线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>x_active_mask<sup>2</sup></td>
        <td>Tensor</td>
        <td>可选</td>
        <td>表示token是否参与通信。</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(num_tokens, )</td>
    </tr>
    <tr>
        <td>activation</td>
        <td>str</td>
        <td>可选</td>
        <td>激活函数类型，默认值为"swiglu"。当前仅支持"swiglu"。</td>
        <td>str</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>activation_clamp</td>
        <td>float</td>
        <td>可选</td>
        <td>激活函数输入的对称截断阈值，将输入张量限制在 [-activation_clamp, activation_clamp] 区间内。None表示不截断，值需≥0。</td>
        <td>float</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>weight1_type</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，使用默认值即可。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>weight2_type</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，使用默认值即可。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
</tbody>
<tfoot>
<tr>
    <td colspan="7">表格中的<code>CeilDiv(<var>x</var>, <var>y</var>) = ⌈<var>x</var> / <var>y</var>⌉ = ⌊(<var>x</var> + <var>y</var> - 1) / <var>y</var>⌋</code></td>
</tr>
<tr>
    <td colspan="7">表格中用T1(T2)表示数据类型T1在传入前要求重解释为另一个数据类型T2再传入，例如，INT4(INT32)表示实际INT4的数据，在传入需重解释为INT32传入，其shape为重解释后的shape。</td>
</tr>
</tfoot>
</table>

## 返回值说明

### get_symm_buffer_for_mega_moe

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>必选</td>
        <td>用于封装输入参数并生成`context`、`ep_world_size`和`ccl_buffer_size`。</td>
        <td>SymmBuffer</td>
        <td>-</td>
    </tr>
</tbody>
</table>

### mega_moe

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡收到的token数据，对应公式中的Y，数据类型与输入`x`保持一致。要求为2维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡每个专家实际收到的token数量。要求为1维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>int32</td>
        <td>(num_experts_per_rank,)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 各张量参数的list[Tensor]长度、是否转置、是否支持非连续Tensor约束如下：

    <table style="undefined;table-layout: fixed; width:1000px"><colgroup>
    <col style="width: 160px">
    <col style="width: 320px">
    <col style="width: 160px">
    <col style="width: 220px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>list[Tensor]长度</th>
        <th>是否转置</th>
        <th>是否支持非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>不涉及</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>topk_ids</td>
            <td>不涉及</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>topk_weights</td>
            <td>不涉及</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>l1_weights</td>
            <td>num_experts_per_rank（BFLOAT16/INT8/INT4场景）或1（FLOAT8_E5M2/FLOAT8_E4M3FN场景）</td>
            <td>否（BFLOAT16/INT8/INT4场景）/是（FLOAT8_E5M2/FLOAT8_E4M3FN场景）</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l2_weights</td>
            <td>num_experts_per_rank（BFLOAT16/INT8/INT4场景）或1（FLOAT8_E5M2/FLOAT8_E4M3FN场景）</td>
            <td>否（BFLOAT16/INT8/INT4场景）/是（FLOAT8_E5M2/FLOAT8_E4M3FN场景）</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l1_weights_sf</td>
            <td>num_experts_per_rank（UINT64场景）或1（FLOAT8_E8M0场景）</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l2_weights_sf</td>
            <td>num_experts_per_rank（UINT64场景）或1（FLOAT8_E8M0场景）</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l1_bias</td>
            <td>num_experts_per_rank</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l2_bias</td>
            <td>num_experts_per_rank</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>不涉及</td>
            <td>否</td>
            <td>支持</td>
        </tr>
    </tbody>
    </table>

- **参数一致性约束**：
    - mega_moe 接口的所有输入参数及其对应的张量维度，必须与 get_symm_buffer_for_mega_moe 的同名参数（例如 `num_experts`、`hidden`、`intermediate_hidden` 等）保持一致。
    - 调用算子过程中使用的`num_experts`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_dtype`、`num_max_tokens_per_rank`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域和组网约束**：
    - 所有卡的`ep_world_size`、`ccl_buffer_size`参数取值需保持一致。
    - 各卡的通信域缓存区大小应当一致。`ccl_buffer_size` 为 HBM 上分配的 CCL 通信缓冲区**总大小**（Bytes），包含等大小的 **windowIn** 和 **windowOut** 两块空间，校验时以单个空间 `ccl_buffer_size / 2` 为准，需满足：

        $$ccl\_buffer\_size\ /\ 2 \ge \mathrm{offsetTokenPerExpert} + \mathrm{offsetTensor} + \mathrm{offsetFlag} + 10\,\mathrm{MB}$$

        **Atlas A2 训练系列产品/Atlas A2 推理系列产品：**

        ```Cpp
        offsetTokenPerExpert = ep_world_size × CeilAlign(ep_world_size × maxExpertPerRank + 1, 128) × 4B

        // winIn
        offsetAAfterDispatch = max_recv_token_num × (quant ? hidden + 512 : hidden × 2)
        offsetD              = num_max_tokens_per_rank × num_topk × hidden × 2B
        winInTensorSize      = offsetAAfterDispatch + offsetD

        // winOut
        offsetA              = num_max_tokens_per_rank × num_topk × (quant ? hidden + 512 : hidden × 2)
        offsetC              = max_recv_token_num × hidden × 2B
        winOutTensorSize     = offsetA + offsetC

        offsetTensor         = max(winInTensorSize, winOutTensorSize)
                            + (quant ? max_recv_token_num × 4B : 0)

        // sync flags
        offsetFlag           = ep_world_size × 512B
                            + ep_world_size × maxExpertPerRank × 64B
                            + ep_world_size × 64B
        ```

        **Atlas A3 训练系列产品/Atlas A3 推理系列产品：**

        ```Cpp
        offsetTokenPerExpert = ep_world_size × CeilAlign(ep_world_size × maxExpertPerRank + 1, 128) × 4B

        // winIn（A3 仅使用 winIn）
        offsetAAfterDispatch = num_max_tokens_per_rank × num_topk × (quant ? hidden + 512 : hidden × 2)
        offsetD              = num_max_tokens_per_rank × num_topk × hidden × 2B
        winInTensorSize      = offsetAAfterDispatch + offsetD

        offsetTensor         = winInTensorSize
                            + (quant ? num_max_tokens_per_rank × num_topk × 4B : 0)

        // sync flags（A3 仅 CrossRankSync）
        offsetFlag           = ep_world_size × 512B
        ```

        其中 `ep_world_size` 即通信域大小，$maxExpertPerRank$ 表示每张卡上可能专家数的最大值，$\mathrm{quant}$ 表示是否开启 dispatch 量化（`dispatch_quant_mode = 2`）。预留空间 10 MB 为内部元数据对齐与安全余量。
    - 通信域各节点的驱动版本应当相同。
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品：多机通信域要求交换机组网，不支持双机直连组网。
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：多机通信域要求在一个超节点内，不支持双机直连组网和跨超节点组网。

- **参数约束**：

    - **公共约束**：
        - `num_experts`：取值范围 $[\text{ep\_world\_size},\ 1024]$，且 `num_experts % ep_world_size == 0`。

    - **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品**：
        - 各卡 `num_tokens` 需保持一致，且每卡 `num_tokens`（即 `x.dim0`）必须小于或等于 `num_max_tokens_per_rank`。
        - `ep_world_size`：仅支持 2、4、8、16、32、64。
        - `num_experts_per_rank`：取值范围 $[1,\ 128]$，且 `num_experts_per_rank = num_experts / ep_world_size`。
        - `num_max_tokens_per_rank`：取值范围 $[1,\ 4096]$。
        - `max_recv_token_num` 需大于0，输入0表示自动计算，公式为 `num_tokens × ep_world_size × min(num_topk, num_experts_per_rank)`。
        - `num_topk`：取值范围 $[1,\ 16]$。
        - `hidden`：取值范围 $[1024,\ 8192]$，且 `hidden % 512 == 0`。
        - `intermediate_hidden`：取值范围 $[1024,\ 3072]$，且 `intermediate_hidden % 512 == 0`
        - `dispatch_quant_mode`：仅支持 0（非量化）、2（INT8 量化）。
        - `dispatch_quant_out_dtype`：仅支持 `torch.int8`（对应整数编码 1）。
        - A2/A3 产品的场景配套矩阵：

            <table>
              <thead>
                <tr>
                  <th>场景</th>
                  <th>x</th>
                  <th>l1_weights</th>
                  <th>l2_weights</th>
                  <th>l1_weights_sf</th>
                  <th>l2_weights_sf</th>
                  <th>l1_bias</th>
                  <th>l2_bias</th>
                  <th>y</th>
                  <th>dispatch_quant_mode</th>
                  <th>dispatch_quant_out_dtype</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>A16W16</td>
                  <td>BFLOAT16</td>
                  <td>BFLOAT16</td>
                  <td>BFLOAT16</td>
                  <td>–</td>
                  <td>–</td>
                  <td>–</td>
                  <td>–</td>
                  <td>BFLOAT16</td>
                  <td>0</td>
                  <td>–</td>
                </tr>
                <tr>
                  <td>A8W8-INT</td>
                  <td>BFLOAT16</td>
                  <td>INT8</td>
                  <td>INT8</td>
                  <td>UINT64</td>
                  <td>UINT64</td>
                  <td>–</td>
                  <td>–</td>
                  <td>BFLOAT16</td>
                  <td>2</td>
                  <td>torch.int8</td>
                </tr>
                <tr>
                  <td>A8W4-INT</td>
                  <td>BFLOAT16</td>
                  <td>INT4(INT32)</td>
                  <td>INT4(INT32)</td>
                  <td>UINT64</td>
                  <td>UINT64</td>
                  <td>FLOAT32</td>
                  <td>FLOAT32</td>
                  <td>BFLOAT16</td>
                  <td>2</td>
                  <td>torch.int8</td>
                </tr>
              </tbody>
            </table>

    - **Ascend 950PR/Ascend 950DT**：
        - `num_tokens`：取值范围 $[1,\ 512]$。
        - `hidden`：仅支持 4096、5120、7168。
        - `num_topk`：仅支持 6、8。
        - `num_experts_per_rank`：取值范围 $[1,\ 16]$，且 `num_experts_per_rank = num_experts / ep_world_size`。
        - `intermediate_hidden`：仅支持 1024、2048、3072、4096、7168。
        - `ep_world_size`：取值范围 $[2,\ 768]$。
        - `num_max_tokens_per_rank`：为 0 或满足 `num_tokens == num_max_tokens_per_rank`。
        - `dispatch_quant_out_dtype`：仅支持 `torch.float8_e5m2`（对应整数编码 23）或 `torch.float8_e4m3fn`（对应整数编码 24）。
        - `dispatch_quant_mode`：仅支持 4（MXFP 量化模式），dispatch 阶段使用 MX 逐组量化（group size = 32），量化缩放因子类型为 FLOAT8_E8M0。
        - `combine_quant_mode`：必须为 0。
        - `comm_alg`：预留参数，必须为空字符串 `""`。
        - `x_active_mask` 和 `scales`：当前版本必须传入空指针，不支持非空输入。
        - `y` 的数据类型与 `x` 相同。
        - `weight1` 的 dim1（`intermediate_hidden`）必须等于 `weight2` 的 dim2 的二倍。`weight1` shape 为 `(num_experts_per_rank, intermediate_hidden, hidden)`，`weight2` shape 为 `(num_experts_per_rank, hidden, intermediate_hidden / 2)`。MXFP 场景下，`dispatch_quant_out_dtype` 为 `torch.float8_e5m2` 时 `weight1` 和 `weight2` 必须为 FLOAT8_E5M2，为 `torch.float8_e4m3fn` 时 `weight1` 和 `weight2` 必须为 FLOAT8_E4M3FN。
        - `weight_scales1`、`weight_scales2`：不可为空指针。`weight_scales1` shape 为 `(num_experts_per_rank, intermediate_hidden, CeilDiv(hidden, 64), 2)`，`weight_scales2` shape 为 `(num_experts_per_rank, hidden, CeilDiv(intermediate_hidden / 2, 64), 2)`，两者 dim3 必须等于 2。其中 $CeilDiv(hidden,\ 64) = \lceil hidden\ /\ 64 \rceil = \lfloor (hidden + 63)\ /\ 64 \rfloor$，$CeilDiv(intermediate\_hidden / 2,\ 64) = \lceil (intermediate\_hidden / 2)\ /\ 64 \rceil = \lfloor (intermediate\_hidden / 2 + 63)\ /\ 64 \rfloor$。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  下面示例将两个接口按调用顺序串联：先初始化通信域，然后用 get_symm_buffer_for_mega_moe 构造 sym_buffer，最后调用 mega_moe 运行算子。
  - **Ascend 950PR/Ascend 950DT**：

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    import torchair

    E = 4
    num_tokens = 256
    H = 4096
    N = 1024
    topK = 6
    num_experts = 8

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def ceil(a, b):
        return (a + b - 1) // b

    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        # 创建HCCL通信链路并初始化EP域
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")

        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group

    def get_megamoe_kwargs(
        x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        x = x.to(torch.bfloat16).npu()
        expert_ids = expert_ids.to(torch.int32).npu()
        weights1 = weights1.to(torch.float8_e5m2).npu()
        weights_scales1 = weights_scales1.to(torch.float8_e8m0fnu).npu()
        weights2 = weights2.to(torch.float8_e5m2).npu()
        weights_scales2 = weights_scales2.to(torch.float8_e8m0fnu).npu()
        expert_scales = expert_scales.to(torch.bfloat16).npu()

        return {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
        }

    def run_megamoe_npu(
        queue, rank, x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
        megamoe_kwargs = get_megamoe_kwargs(
            x=x,
            expert_ids=expert_ids,
            weights1=weights1,
            weights_scales1=weights_scales1,
            weights2=weights2,
            weights_scales2=weights_scales2,
            expert_scales=expert_scales,
        )
        # 步骤1：初始化通信域
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        # 步骤2：构造distribute_buffer（SymmBuffer结构体）
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=4, dispatch_quant_out_dtype=torch.float8_e5m2
        )
        # 步骤3：运行mega_moe，传入上一步构造的sym_buffer
        y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank} npu finished! \n')

        queue.put([
            rank,
            [
                y.cpu(), expert_token_nums.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):

            ep_id = rank // 1

            if target_func == run_megamoe_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                    "weights1": server_kwargs["weights1_list"][ep_id],
                    "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                    "weights2": server_kwargs["weights2_list"][ep_id],
                    "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                    "expert_scales": server_kwargs["expert_scales_list"][ep_id]
                }


        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
        mp.set_start_method("forkserver", force=True)
        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            proc.start()
            proc_list.append(proc)


        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output


        for proc in proc_list:
            proc.join()

        if None in rank_outputs:
            print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
            exit(1)

        # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
        category_outputs = []
        category_num = len(rank_outputs[0])
        for category_id in range(category_num):
            specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
            category_outputs.append(specific_category_output)

        return category_outputs

    if __name__ == "__main__":
        x_shape = [num_tokens, H]
        expert_idx_shape = [num_tokens, topK]
        weight_shape = [E, N, H]
        weight_scale_shape = [E, N, ceil(H, 64), 2]
        output_shape = [num_tokens, N//2]
        weight2_shape = [E, H, N//2]
        weight2_scale_shape = [E, H, ceil(N//2, 64), 2]
        expert_scales_shape = [num_tokens, topK]
        # 构造输入
        x = torch.randn(x_shape, dtype=torch.bfloat16)
        expert_scales = torch.randn(expert_scales_shape, dtype=torch.bfloat16)
        expert_ids = torch.stack(
            [torch.randperm(num_experts)[:topK] for _ in range(num_tokens)]
        ).to(torch.int32)
        weight1 = torch.randn(weight_shape, dtype=torch.float32).to(torch.float8_e5m2)
        weight_scales1 = torch.randint(125, 130, weight_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weight2 = torch.randn(weight2_shape, dtype=torch.float32).to(torch.float8_e5m2)
        weight_scales2 = torch.randint(125, 130, weight2_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
        golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
        golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

        [y, expert_token_nums] = gen_npu(
            run_megamoe_npu,
            x_list=golden_x_list,
            expert_ids_list=golden_expert_ids_list,
            weights1_list=golden_weights1_list,
            weights_scales1_list=golden_weights_scales1_list,
            weights2_list=golden_weights2_list,
            weights_scales2_list=golden_weights_scales2_list,
            expert_scales_list=golden_expert_scales_list,
        )
    ```

  - **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品**：

    ```python
    import torch
    import torch_npu
    import torch.multiprocessing as mp
    import torch.distributed as dist
    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

    num_tokens = 64
    hidden = 4096
    intermediate_hidden = 1024
    num_topk = 6
    num_experts = 16
    num_max_tokens_per_rank = 256
    num_servers = 1
    num_ranks_per_server = 2
    server_index = 0  # 当前机器在集群中的编号
    master_addr = '127.0.0.1'
    master_port = 50001
    scene = 'A16W16'

    world_size = num_servers * num_ranks_per_server
    num_experts_per_rank = num_experts // world_size

    torch_npu.npu.config.allow_internal_format = True


    def init_hccl_comm(rank):
        print(f'device_{rank} start init_process_group')
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        # 使用时hccl_buffer_size的值需根据算子接口文档中给出的通信域缓存区大小的约束进行配置
        options.hccl_config = {'hccl_buffer_size': 200}
        dist.init_process_group(
            backend='hccl',
            rank=server_index * num_ranks_per_server + rank,
            world_size=world_size,
            init_method=f'tcp://{master_addr}:{master_port}',
            pg_options=options,
        )
        print(f'device_{rank} init_process_group success')
        ep_group = dist.new_group(backend='hccl', ranks=list(range(world_size)))
        _ = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        return ep_group


    def bf16_mega_moe(rank):
        torch_npu.npu.set_device(rank % num_ranks_per_server)

        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='npu')
        topk_weights, topk_ids = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_ids = topk_ids.to(torch.int32)
        l1_weights = [
            torch.randn((hidden, intermediate_hidden * 2), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]
        l2_weights = [
            torch.randn((intermediate_hidden, hidden), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]

        ep_group = init_hccl_comm(rank)

        sym_buffer = get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            dispatch_quant_mode=0,
            dispatch_quant_out_dtype=None,
        )
        y, expert_token_nums = mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer)

        dist.barrier()
        dist.destroy_process_group()


    def int8_int8_mega_moe(rank):
        torch_npu.npu.set_device(rank % num_ranks_per_server)

        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='npu')
        topk_weights, topk_ids = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_ids = topk_ids.to(torch.int32)
        l1_weights_bf16 = [
            torch.randn((hidden, intermediate_hidden * 2), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]
        l2_weights_bf16 = [
            torch.randn((intermediate_hidden, hidden), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]

        def per_channel_cast_to_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            max_abs = torch.amax(torch.abs(w), dim=0, keepdim=True)
            sf = torch.clamp(max_abs / 127.0, min=1e-8)
            q_weight = torch.round(w / sf).clamp(-127, 127).to(torch.int8)
            return q_weight, sf.squeeze().to(torch.float32)

        l1_weights_int8, l1_weights_sf_float = map(list, zip(*[per_channel_cast_to_int8(w) for w in l1_weights_bf16]))
        l2_weights_int8, l2_weights_sf_float = map(list, zip(*[per_channel_cast_to_int8(w) for w in l2_weights_bf16]))

        # Cast weights to NZ format for better performance.
        l1_weights = [torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ) for w in l1_weights_int8]
        l2_weights = [torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ) for w in l2_weights_int8]

        # Pack float32 scale factors into uint64, which is required for the process of fixpipe dequantization.
        l1_weights_sf = [sf.view(torch.int32).to(torch.int64).view(torch.uint64) for sf in l1_weights_sf_float]
        l2_weights_sf = [sf.view(torch.int32).to(torch.int64).view(torch.uint64) for sf in l2_weights_sf_float]

        ep_group = init_hccl_comm(rank)

        sym_buffer = get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            dispatch_quant_mode=2,
            dispatch_quant_out_dtype=torch.int8,
        )
        y, expert_token_nums = mega_moe(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
        )

        dist.barrier()
        dist.destroy_process_group()


    def int8_int4_mega_moe(rank):
        torch_npu.npu.set_device(rank % num_ranks_per_server)

        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='npu')
        topk_weights, topk_ids = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_ids = topk_ids.to(torch.int32)
        l1_weights_bf16 = [
            torch.randn((hidden, intermediate_hidden * 2), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]
        l2_weights_bf16 = [
            torch.randn((intermediate_hidden, hidden), dtype=torch.bfloat16, device='npu')
            for _ in range(num_experts_per_rank)
        ]

        def per_channel_cast_to_int4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            max_abs = torch.amax(torch.abs(w), dim=0, keepdim=True)
            sf = torch.clamp(max_abs / 7.0, min=1e-8)
            q_weight = torch.round(w / sf).clamp(-7, 7).to(torch.int8)
            return q_weight, sf.squeeze().to(torch.float)

        def pack_int4_to_int8(x: torch.Tensor) -> torch.Tensor:
            x = (x + 8).to(torch.uint8).reshape(x.shape[0], -1, 2)
            x = ((x[..., 1] << 4) | x[..., 0]).view(torch.int8)
            return x

        l1_weights_int8, l1_weights_sf_float = map(list, zip(*[per_channel_cast_to_int4(w) for w in l1_weights_bf16]))
        l2_weights_int8, l2_weights_sf_float = map(list, zip(*[per_channel_cast_to_int4(w) for w in l2_weights_bf16]))

        l1_weights_int4 = [pack_int4_to_int8(w) for w in l1_weights_int8]
        l2_weights_int4 = [pack_int4_to_int8(w) for w in l2_weights_int8]

        # Cast weights to NZ format for better performance.
        l1_weights = [torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ).view(torch.int32) for w in l1_weights_int4]
        l2_weights = [torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ).view(torch.int32) for w in l2_weights_int4]

        # Pack float32 scale factors into uint64, which is required for the process of fixpipe dequantization.
        l1_weights_sf = [sf.view(torch.int32).to(torch.int64).view(torch.uint64) for sf in l1_weights_sf_float]
        l2_weights_sf = [sf.view(torch.int32).to(torch.int64).view(torch.uint64) for sf in l2_weights_sf_float]

        l1_bias = [(w.float() * sf.unsqueeze(0)).sum(dim=0) * 8.0 for w, sf in zip(l1_weights_int8, l1_weights_sf_float)]
        l2_bias = [(w.float() * sf.unsqueeze(0)).sum(dim=0) * 8.0 for w, sf in zip(l2_weights_int8, l2_weights_sf_float)]

        ep_group = init_hccl_comm(rank)

        sym_buffer = get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            dispatch_quant_mode=2,
            dispatch_quant_out_dtype=torch.int8,
        )
        y, expert_token_nums = mega_moe(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
        )

        dist.barrier()
        dist.destroy_process_group()


    if __name__ == '__main__':
        if scene == 'A16W16':
            torch.multiprocessing.spawn(bf16_mega_moe, nprocs=num_ranks_per_server)
        elif scene == 'A8W8-INT':
            torch.multiprocessing.spawn(int8_int8_mega_moe, nprocs=num_ranks_per_server)
        elif scene == 'A8W4-INT':
            torch.multiprocessing.spawn(int8_int4_mega_moe, nprocs=num_ranks_per_server)
        else:
            raise ValueError(f"Unsupported scene: {scene}, please choose from ['A16W16', 'A8W8-INT', 'A8W4-INT']")
    ```
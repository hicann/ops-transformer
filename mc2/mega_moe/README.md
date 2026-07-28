# MegaMoE

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：MegaMoE算子 将 MoE 层的专家 FFN 的完整计算流程及前后数据通信（即 Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。

- 计算公式：
  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$：激活矩阵，对应入参 `x`。$\text{totalNumTokens}$ 是全局总 token 数，$\text{hidden}$ 是隐藏层维度。
    - $\mathbf{E} \in \mathbb{Z}^{\text{totalNumTokens} \times \text{topK}}$：token 选择的专家编号矩阵，对应入参 `topkIds`。$\text{topK}$ 是每个 token 选择的专家数量。
    - $\mathbf{G} \in \mathbb{R}^{\text{totalNumTokens} \times \text{topK}}$：token 选择的专家的门控权重矩阵，对应入参 `topkWeights`。
    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{hidden} \times (2 \text{intermediateHidden})}$：路由 MoE 专家的 Linear1 权重，对应入参 `weight1` 的 MoE 专家部分。
    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{intermediateHidden} \times \text{hidden}}$：路由 MoE 专家的 Linear2 权重，对应入参 `weight2` 的 MoE 专家部分。
    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{hidden} \times (2 \text{intermediateHidden})}$：共享专家的 Linear1 权重，对应入参 `sharedWeight1`。
    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{intermediateHidden} \times \text{hidden}}$：共享专家的 Linear2 权重，对应入参 `sharedWeight2`。
  - 输出：

    - $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$：最终输出矩阵，对应出参 `y`。
  - 约定：
    - $⋅$ 表示矩阵乘法，$⊙$ 表示逐元素乘法。
    - $\left \lfloor z\right \rceil$ 表示将 $z$ 四舍五入到最近的整数，$\left \lfloor z\right \rfloor$ 表示将 $z$ 向下取整。
    - $|z|$ 表示取绝对值，$\max(z)$ 表示取最大值。
    - 全体 token 的集合为 $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{totalNumTokens} - 1\} \}$。
    - $\text{token}_i$ 的 token 表示（即隐藏状态向量）为 $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$，且 $\mathbf{x}_i = \mathbf{X}[i,:]$。
    - $\text{token}_i$ 的专家索引为 $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{topK} - 1\},\quad e_{i,k} \in \{0,\dots,\text{moeExpertNum} - 1\}$。
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
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持上表中A8W8-FP场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：不支持上表中A16W16 、A8W8-INT、A8W4-INT场景。

    <details>
    <summary> A16W16 非量化场景</summary>

    - **EP Dispatch**

        在 Dispatch 阶段，每个 $\text{token}_i$ 将其token表示 $\mathbf{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数，则有 $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的token表示 $\mathbf{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$。该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。整个计算过程分为如下三个子步骤。

        **1. Linear1 投影**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。其计算公式为
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}\bigr)
        $$
        **2. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e\bigl[:,\;\text{:intermediateHidden}\bigr], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e\bigl[:,\;\text{intermediateHidden:}\bigr]
        $$
        $$
        \mathbf{A}_e = \text{SiLU}\bigl(\mathbf{H}_{\text{gate}}\bigr) \;\odot\; \mathbf{H}_{\text{up}}
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{intermediateHidden}}\bigr)
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
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$。
    </details>

    <details>
    <summary> A8W8-INT 量化场景</summary>

    - **输入**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `weightScales1`。
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `weightScales2`。

    - **EP Dispatch**

        在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\mathbf{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到 INT8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$（$e_{i,k} = \mathbf{E}[i,k]$）。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

    - **Expert Compute**

        在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 SwiGLU 结构以提升表达能力。在A8W8场景下，两个线性层都使用 INT8 输入和 INT8 权重进行矩阵乘，得到 INT32 中间结果并反量化。具体分为三个子步骤。

        **1. Linear1 投影（INT8 矩阵乘 + 反量化）**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。计算时执行 INT8 矩阵乘法，得到 INT32 计算结果：
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$
        然后反量化为预激活值 $\mathbf{H}_e$：
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **2. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:, :\text{intermediateHidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:, \text{intermediateHidden}:]
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
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
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
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$。

    </details>

    <details>
    <summary> A8W4-INT 量化场景</summary>

    - **输入**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `weightScales1`。
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `weightScales2`。
        - $\mathbf{B}_1 \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$：Linear1 的偏置矩阵，由 INT4 量化过程离线生成，对应入参 `bias1`。
        - $\mathbf{B}_2 \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$：Linear2 的偏置矩阵，由 INT4 量化过程离线生成，对应入参 `bias2`。

    - **EP Dispatch**

        在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\mathbf{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到 INT8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$

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

        $\mathbf{B}_1$ 和 $\mathbf{B}_2$ 均按此法，分别使用对应的全 1 列向量（维度分别为 $\text{hidden}$ 和 $\text{intermediateHidden}$）与各自的权重矩阵相乘，得到形状匹配的偏置行向量。这些偏置在离线阶段计算完成后传入算子，在合并高低位结果时直接参与加法，从而消除偏移引入的误差。

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

        将重解释后的 INT4 激活视图 $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ 与权重矩阵 $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediateHidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W1}_e$ 进行反量化，得到结果：

        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediateHidden}}
        $$

        将结果按偶数行和奇数行分别记为如下的矩阵视图，它们即对应于$\mathbf{Q}_e^{\text{high}}$和$\mathbf{Q}_e^{\text{low}}$的计算结果：

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **(3) 精度补偿与激活反量化**

        利用偏置 $\mathbf{B}_1[e]$ 和激活缩放因子 $\mathbf{s}^{X}_e$（行向量）分别进行精度补偿和激活反量化，最终预激活值为：

        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        其中 $\odot \mathbf{s}^{X}_e$ 表示将矩阵的每一行乘以 $\mathbf{s}^{X}_e$ 中对应的标量。

        **3. SwiGLU 激活**

        首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:,\; :\!\text{intermediateHidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:,\; \text{intermediateHidden}:]
        $$
        $$
        \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
        \quad\in \mathbb{R}^{N_e \times \text{intermediateHidden}}
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
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$

        **(2) 激活重解释为 INT4**

        将 INT8 激活张量 $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}$ 二进制重解释为两个 INT4 交替拼接的视图：

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}
        $$

        其中高位和低位分量直接由 $\mathbf{A}_e^{\text{int4}}$ 的偶数行和奇数行给出：

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}
        $$

        它们在数值上可以由原 INT8 经如下计算得到：

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        恢复关系为 $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\mathbf{A}_e^{\text{int4}}$ 与 $\mathbf{A}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

        **(3) INT4 × INT4 矩阵乘与权重反量化**

        将重解释后的 INT4 激活视图 $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}$ 与权重矩阵 $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediateHidden} \times \text{hidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W2}_e$ 进行反量化，得到结果：

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
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$。

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

    其中，$X$ 表示参数 `x`，$W$ 表示参数 `topkWeights`，$W_1$ 表示参数 `weight1`，$W_2$ 表示参数 `weight2`，$Y$ 表示参数 `y`，$E_{\text{local}}$ 表示 `localMoeExpertNum = moeExpertNum / epWorldSize`（每个 Rank 的路由 MoE 专家数），$K$ 表示 `topkIds` 的第二维度。

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

    <details>
    <summary> A8W4-FP 量化场景</summary>

    第一阶段（Token 选择、量化与 Dispatch）：

    对本 rank 的输入 Token $X \in \mathbb{R}^{B\times H}$，根据 `topkIds` 得到每个专家 $e$ 对应的 Token 下标集合 $T_e$，并将选中的 BF16 Token 按 32 个元素一组量化为 MXFP8 E4M3：

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    其中，$\hat{X}_e$ 的数据类型为 `FLOAT8_E4M3FN`，$S_{X,e}$ 的数据类型为 `FLOAT8_E8M0`。随后将量化后的 Token 及其缩放因子发送到专家所在 rank。

    路由 MoE 专家的第一层和第二层权重 $W_{1,e}$、$W_{2,e}$ 均为 MXFP4 E2M1 数据，缩放因子为 E8M0，$e$ 的范围为 $[0,\text{localMoeExpertNum})$。共享专家权重单独由 `sharedWeight1`、`sharedWeight2` 提供，其第一维为 `sharedExpertNumPerRank`。A8W4 kernel 在矩阵乘 Prologue 中处理 FP4 权重，送入矩阵乘的逻辑数据流为 FP8 激活乘 FP4 权重。

    第二阶段（GMM1、SwiGLU 与再次量化）：

    对专家 $e$ 收到的 Token，第一层分组矩阵乘和 SwiGLU 计算为：

    $$
    Z_x = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(x)},S_{1,e}^{(x)}\right),
    $$

    $$
    Z_y = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(y)},S_{1,e}^{(y)}\right),
    $$

    $$
    U_e = Z_x \odot \sigma(Z_x) \odot Z_y.
    $$

    SwiGLU 输出继续按 32 个元素一组量化为 MXFP8 E4M3，供第二层矩阵乘使用：

    $$
    \hat{U}_e,\;S_{U,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(U_e\right).
    $$

    第三阶段（GMM2）：

    第二层矩阵乘仍为 A8W4，即 FP8 E4M3 激活乘 FP4 E2M1 权重：

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{U}_e,S_{U,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    第四阶段（Combine 与加权合并）：

    将各专家输出送回原 rank，并根据 `topkWeights` 做加权合并：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)],
    $$

    最终输出 $Y$ 的数据类型为 BF16。A8W4-FP 的主要数据类型流为：`BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`。
    </details>

    <details>
    <summary> A4W4-FP 量化场景</summary>

    第一阶段（Token 选择、量化与 Dispatch）：

    对每个专家 $e$ 对应的 Token 集合 $T_e$，将选中的 BF16 Token 按 32 个元素一组量化为 MXFP4 E2M1：

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    其中，$\hat{X}_e$ 的数据类型为 `FLOAT4_E2M1`，$S_{X,e}$ 的数据类型为 `FLOAT8_E8M0`。路由 MoE 专家的第一层和第二层权重 $W_{1,e}$、$W_{2,e}$ 均为 MXFP4 E2M1，$e$ 的范围为 $[0,\text{localMoeExpertNum})$；共享专家权重的第一维为 `sharedExpertNumPerRank`。权重缩放因子为 E8M0。

    第二阶段（A4W4 GMM1、SwiGLU 与输出类型提升）：

    第一层分组矩阵乘为 A4W4：

    $$
    Z_x = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(x)},S_{1,e}^{(x)}\right),
    $$

    $$
    Z_y = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(y)},S_{1,e}^{(y)}\right),
    $$

    $$
    U_e = Z_x \odot \sigma(Z_x) \odot Z_y.
    $$

    这里不能继续把 SwiGLU 输出量化为 FP4。kernel 在 `QuantMode == E2M1_QUANT` 时，将 `SwigluQuantOutType` 指定为 `fp8_e4m3fn_t`，因此输出会提升为 MXFP8 E4M3：

    $$
    \hat{U}_e,\;S_{U,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(U_e\right).
    $$

    第三阶段（A8W4 GMM2）：

    由于 SwiGLU 量化输出为 FP8 E4M3，而第二层权重仍为 FP4 E2M1，因此第二层矩阵乘实际为 A8W4，而不是 A4W4：

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{U}_e,S_{U,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    第四阶段（Combine 与加权合并）：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)],
    $$

    最终输出 $Y$ 的数据类型为 BF16。A4W4-FP 的完整数据类型流为：`BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`。其中所有 MX 缩放因子的类型均为 `FLOAT8_E8M0`，量化粒度均为 32 个连续元素。
    </details>

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>MoE层输入的token隐藏状态。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkIds</td>
   <td>输入</td>
   <td>专家索引矩阵，表示每个token选择的topK个专家。元素取值范围为[0, moeExpertNum)，且同一token选择的topK个专家不能重复。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkWeights</td>
   <td>输入</td>
   <td>表示MoE模型的专家门控网络为当前输入Token选出的topK个专家所对应的门控权重系数。</td>
   <td>FP32、BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td>输入</td>
   <td>MoE专家网络第一线性层的权重矩阵（包括门控与上投影），用于将输入映射至中间维度，输出供给激活函数。路由 MoE 专家数为 <code>localMoeExpertNum</code>。</td>
   <td>BF16、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1</td>
   <td>ND、FRACTAL_NZ、FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>输入</td>
   <td>MoE专家网络第二线性层的权重矩阵，负责将激活后的中间特征投影回隐藏维度。数据类型与weight1一致。路由 MoE 专家数为 <code>localMoeExpertNum</code>。</td>
   <td>BF16、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1</td>
   <td>ND、FRACTAL_NZ、FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weightScales1</td>
   <td>可选输入</td>
   <td>MoE专家网络第一线性层的权重矩阵的量化缩放因子。</td>
   <td>FLOAT8_E8M0、UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScales2</td>
   <td>可选输入</td>
   <td>MoE专家网络第二线性层的权重矩阵的量化缩放因子。</td>
   <td>FLOAT8_E8M0、UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias1</td>
   <td>可选输入</td>
   <td>MoE专家网络第一线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias2</td>
   <td>可选输入</td>
   <td>MoE专家网络第二线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td>可选输入</td>
   <td>表示token是否参与通信。</td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
    <tr>
   <td>sharedWeight1</td>
   <td>可选输入</td>
   <td>共享专家网络第一线性层的权重矩阵（包括门控与上投影），用于将输入映射至中间维度，输出供给激活函数。</td>
   <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1</td>
   <td>ND、FRACTAL_NZ、FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeight2</td>
   <td>可选输入</td>
   <td>共享专家网络第二线性层的权重矩阵，负责将激活后的中间特征投影回隐藏维度。数据类型与weight1一致。</td>
   <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1</td>
   <td>ND、FRACTAL_NZ、FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeightScales1</td>
   <td>可选输入</td>
   <td>共享专家网络第一线性层的权重矩阵的量化缩放因子。</td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeightScales2</td>
   <td>可选输入</td>
   <td>共享专家网络第二线性层的权重矩阵的量化缩放因子。</td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias1</td>
   <td>可选输入</td>
   <td>共享专家网络第一线性层的偏置，暂不支持。</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias2</td>
   <td>可选输入</td>
   <td>共享专家网络第二线性层的偏置，暂不支持。</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>属性</td>
   <td>MoE模型的总专家数量。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>属性</td>
   <td>专家并行通信域大小。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>cclBufferSize</td>
   <td>属性</td>
   <td>CCL通信缓冲区大小。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>maxRecvTokenNum</td>
   <td>可选属性</td>
   <td>每个Rank最大可接收Token数。默认值为0。该值为0时，会按最大值bs*epWorldSize*min(topK,localMoeExpertNum)预留内存大小；不为0时，将按照输入值大小预留内存，该场景下需用户保证填入值大于等于每个rank最大可接收的Token数。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantMode</td>
   <td>可选属性</td>
   <td>dispatch通信时量化模式。0表示非量化（A16W16场景），2表示INT8量化（A8W8-INT、A8W4-INT场景），4表示MXFP量化（A8W8-FP、A8W4-FP、A4W4-FP场景）。默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantOutDtype</td>
   <td>可选属性</td>
   <td>dispatch量化后输出的数据类型。支持1（INT8）、23（FLOAT8_E5M2）、24（FLOAT8_E4M3FN）、296（FLOAT4_E2M1）。默认值为DT_UNDEFINED。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combineQuantMode</td>
   <td>可选属性</td>
   <td>combine通信时的量化模式。0表示非量化，3表示MXFP float8_e5m2类型，4表示MXFP float8_e4m3类型, 默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>可选属性</td>
   <td>预留参数，暂不支持。默认值为""。</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>numMaxTokensPerRank</td>
   <td>可选属性</td>
   <td>每张卡上的token数量。当每个rank的numTokens不同时，为最大的numTokens大小。默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>activation</td>
   <td>可选属性</td>
   <td>激活函数类型。默认值为"swiglu"。当前仅支持"swiglu"。</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>activationClamp</td>
   <td>可选属性</td>
   <td>激活函数输入的对称截断阈值，将输入张量限制在[-activationClamp, activationClamp]区间内。默认值为float最大值，表示不截断，值需≥0。</td>
   <td>FLOAT</td>
   <td></td>
  </tr>
  <tr>
   <td>activationOutDtype</td>
   <td>可选属性</td>
   <td>激活函数输出的数据类型。默认值为DT_UNDEFINED。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight1</td>
   <td>可选属性</td>
   <td>weight1是否转置。默认值为false。</td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight2</td>
   <td>可选属性</td>
   <td>weight2是否转置。默认值为false。</td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>weight1Interleave</td>
   <td>可选属性</td>
   <td>weight1交错参数。预留参数，默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>计算输出结果，数据类型与输入x相同。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNums</td>
   <td>输出</td>
   <td>本卡每个专家实际收到的token数量。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- **预留和非对外参数说明**：
  - 参数表格中的部分参数、部分数据类型暂未对外提供，为预留或内部实现使用。接口参数的介绍及其约束在接口文档[MegaMoE算子接口文档](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)中详细说明。

- **参数一致性约束**：
  - 调用算子过程中使用的`moeExpertNum`、`maxRecvTokenNum`、`dispatchQuantMode`、`dispatchQuantOutDtype`、`numMaxTokensPerRank`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域和组网约束**：
  - 所有卡的`epWorldSize`、`cclBufferSize`参数取值需保持一致。
  - 通信域各节点的驱动版本应当相同。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机通信域要求交换机组网，不支持双机直连组网。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：多机通信域要求在一个超节点内，不支持双机直连组网和跨超节点组网。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持UB Memory通信协议。
- **参数约束**：
  - **<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>**：
    - 场景配套矩阵：

      | 场景 | x | weight1 | weight2 | weightScales1 | weightScales2 | bias1 | bias2 | y | dispatchQuantMode | dispatchQuantOutDtype |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | A16W16 | BF16 | BF16 | BF16 | – | – | – | – | BF16 | 0 | – |
      | A8W8-INT | BF16 | INT8 | INT8 | UINT64 | UINT64 | – | – | BF16 | 2 | 1（INT8） |
      | A8W4-INT | BF16 | INT4(INT32) | INT4(INT32) | UINT64 | UINT64 | FP32 | FP32 | BF16 | 2 | 1（INT8） |

  - **<term>Ascend 950PR/Ascend 950DT</term>**：
    - BS（x.dim0）支持[1, maxBS], maxBS = (8 * (totalUbSize - 48K)) / (topK * (64 + epWorldSize)), 其中totalUbSize为UB总大小，950系列产品该值支持到当前环境能申请的最大内存。
    - H（x.dim1）支持1024、2048、3072、4096、5120、6144、7168、8192。
    - topK（topkIds.dim1）支持[1, 32]。
    - expertPerRank 范围 [1, 1024]。
    - hiddenDim（weight1.dim1）仅支持1024、2048、3072、4096、7168。
    - epWorldSize范围 [2, 1024]。
    - moeExpertNum范围 [epWorldSize, 2048]，且moeExpertNum % epWorldSize == 0。
    - maxRecvTokenNum范围 [0, BS × epWorldSize × min(topK, localMoeExpertNum)]。
    - dispatchQuantOutDtype仅支持23（FLOAT8_E5M2）或24（FLOAT8_E4M3FN）或296（FLOAT4_E2M1）。
    - 当前版本仅支持MXFP量化模式（dispatchQuantMode = 4），dispatch阶段使用MX逐组量化（group size = 32），量化缩放因子类型为FLOAT8_E8M0。
    - combineQuantMode取值为0、3、4，0表示非量化，3表示MXFP float8_e5m2类型，4表示MXFP float8_e4m3类型
    - commAlg必须为空字符串""。
    - y的数据类型与x相同。
    - weight1的dim1（hiddenDim）必须等于weight2的dim2的二倍，这是因为SwiGLU激活需要将中间维度从hiddenDim减半为hiddenDim/2。
    - localMoeExpertNum = moeExpertNum / epWorldSize；sharedExpertNumPerRank = sharedWeight1.dim0（未启用共享专家时为0）；expertPerRank = sharedExpertNumPerRank + localMoeExpertNum。

  - **MXFP量化场景约束**：
      - weight1 shape为(localMoeExpertNum, hiddenDim, H)，weight2 shape为(localMoeExpertNum, H, hiddenDim/2)。
      - weightScales1 shape为(localMoeExpertNum, hiddenDim, CeilDiv(H, 64), 2)。
      - weightScales2 shape为(localMoeExpertNum, H, CeilDiv(hiddenDim/2, 64), 2)。
      - sharedWeight1 shape为(sharedExpertNumPerRank, hiddenDim, H)，sharedWeight2 shape为(sharedExpertNumPerRank, H, hiddenDim/2)。
      - sharedWeightScales1 shape为(sharedExpertNumPerRank, hiddenDim, CeilDiv(H, 64), 2)，sharedWeightScales2 shape为(sharedExpertNumPerRank, H, CeilDiv(hiddenDim/2, 64), 2)。
      - weightScales1的dim3和weightScales2的dim3必须等于2。
      - A8W4-FP场景下，FLOAT4_E2M1类型的weight1必须使用FORMAT_FRACTAL_NZ_C0_32格式。
      - MXFP场景下，dispatchQuantOutDtype=23时weight1和weight2必须为FLOAT8_E5M2，dispatchQuantOutDtype=24时必须为FLOAT8_E4M3FN，dispatchQuantOutDtype=296时必须为FLOAT4_E2M1。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | - | 通过[mega_moe](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)PyTorch接口方式调用mega_moe算子。 |

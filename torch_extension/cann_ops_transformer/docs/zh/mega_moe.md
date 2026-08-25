# mega_moe

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  - Mega MoE算子将MoE层的专家FFN的完整计算流程及前后数据通信（即Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。
  - 该算子提供了mega_moe与get_symm_buffer_for_mega_moe等接口配套使用。
  - get_symm_buffer_for_mega_moe：用于封装输入参数并创建SymmBuffer结构体，生成`context`、`ep_world_size`和`ccl_buffer_size`等mega_moe算子运行所需信息。

- **计算公式**：

  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：激活矩阵，对应入参 `x`。$\text{total\_num\_tokens}$ 是全局总 token 数，$\text{hidden}$ 是隐藏层维度。
    - $\mathbf{E} \in \mathbb{Z}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家编号矩阵，对应入参 `topk_ids`。$\text{num\_topk}$ 是每个 token 选择的专家数量。
    - $\mathbf{G} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家的门控权重矩阵，对应入参 `topk_weights`。
    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{local\_moe\_expert\_num} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$：路由 MoE 专家的 Linear1 权重，对应入参 `l1_weights` 的 MoE 专家部分。
    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{local\_moe\_expert\_num} \times \text{intermediate\_hidden} \times \text{hidden}}$：路由 MoE 专家的 Linear2 权重，对应入参 `l2_weights` 的 MoE 专家部分。
    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{shared\_expert\_num\_per\_rank} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$：共享专家的 Linear1 权重，对应入参 `shared_l1_weights`。
    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{shared\_expert\_num\_per\_rank} \times \text{intermediate\_hidden} \times \text{hidden}}$：共享专家的 Linear2 权重，对应入参 `shared_l2_weights`。
  - 输出：

    - $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：最终输出矩阵，对应出参 `y`。
  - 约定：
    - $⋅$ 表示矩阵乘法，$⊙$ 表示逐元素乘法。
    - $\left \lfloor z\right \rceil$ 表示将 $z$ 四舍五入到最近的整数，$\left \lfloor z\right \rfloor$ 表示将 $z$ 向下取整。
    - $|z|$ 表示取绝对值，$\max(z)$ 表示取最大值。
    - 全体token的集合为 $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{total\_num\_tokens} - 1\} \}$。
    - $\text{token}_i$ 的token表示（即隐藏状态向量）为 $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$，且 $\mathbf{x}_i = \mathbf{X}[i,:]$。
    - $\text{token}_i$ 的专家索引为 $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{num\_topk} - 1\},\quad e_{i,k} \in \{0,\dots,\text{num\_experts} - 1\}$。
    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$。其中 $\mathbb{Z}_8^{\text{sym}}$ 的上标 $\text{sym}$ 表示对称量化值域区间：其值域关于 $-127$ 与 $127$ 对称取整，与标准int8 的 $[-128, 127]$ 值域不同，故以 $\text{sym}$ 上标区分。
    - 张量切片操作采用Python风格的 `start:stop:step` 表示法，例如$[0::2, :]$ 代表取偶数行、$[1::2, :]$ 代表取奇数行。
    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ 表示二进制重解释操作，将张量$\mathbf{Z}$的底层二进制数据按目标类型 $T$ 重新解释。
  - <span id="activation-formulas">激活函数公式：</span>
    - 记 Linear1 输出拆分后的 gate 分支为 $\mathbf{G}$，up 分支为 $\mathbf{U}$，激活输出为 $\mathbf{A}$。Sigmoid 函数和 SiLU 函数定义为：
      $$
      \sigma(z)=\frac{1}{1+e^{-z}}, \qquad
      \operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}.
      $$
      记截断值为 $c$；未启用截断（未配置或配置为 $0$）时，数学上取 $c=+\infty$。定义
      $\mathbf{G}_c=\min(\mathbf{G},c)$，$\mathbf{U}_c=\operatorname{clip}(\mathbf{U},-c,c)$。
    - `swiglu`：
      $$
      \mathbf{A}=\operatorname{SwiGLU}(\mathbf{G},\mathbf{U})
      =\operatorname{Swish}_1(\mathbf{G}_c)\odot\mathbf{U}_c
      =\operatorname{SiLU}(\mathbf{G}_c)\odot\mathbf{U}_c.
      $$
    <!-- npu="950" id25 -->
    - `swiglustep`：
      $$
      \mathbf{A}=\operatorname{SwiGLUStep}(\mathbf{G},\mathbf{U})=\min\!\left(\operatorname{SiLU}(\mathbf{G}),c\right)\odot\mathbf{U}_c.
      $$
    - `swigluoai`：
      $$
      \mathbf{A}=\operatorname{SwiGLUOAI}(\mathbf{G},\mathbf{U})=\mathbf{G}_c\odot\sigma(\alpha\mathbf{G}_c)\odot(\mathbf{U}_c+\beta).
      $$
    - `situglu`：
      $$
      \mathbf{A}=\operatorname{SiTUGLU}(\mathbf{G},\mathbf{U})=\beta\tanh\!\left(\frac{\mathbf{G}}{\beta}\right)\odot\sigma(\mathbf{G})\odot L(\mathbf{U};\beta_{\mathrm{linear}}),
      $$
      其中
      $$
      L(\mathbf{U};\beta_{\mathrm{linear}})=
      \begin{cases}
      \mathbf{U}, & \text{未配置 linear\_beta},\\
      \beta_{\mathrm{linear}}\tanh\!\left(\dfrac{\mathbf{U}}{\beta_{\mathrm{linear}}}\right),
      & \text{已配置 linear\_beta}.
      \end{cases}
      $$
    <!-- end id25 -->

- 计算说明：

    各产品支持的**Linear计算时**的激活矩阵A和权重矩阵W的数据类型如下：

    | 场景名    |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN、FLOAT8_E5M2 |FLOAT8_E4M3FN、FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持A16W16、A8W8-INT和A8W4-INT场景。
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持A8W8-FP、A8W4-FP和A4W4-FP场景。
    <!-- end id8 -->

    <details>
    <summary> A16W16 非量化场景</summary>

    - **EP Dispatch**

        在Dispatch阶段，每个 $\text{token}_i$ 将其token表示 $\mathbf{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的token索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的token总数，则有 $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的token表示 $\mathbf{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过Dispatch后收到的全部token表示。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$。该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。

    - **Expert Compute**

        在MoE层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。整个计算过程分为如下三个子步骤。

        **1. Linear1 投影**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate部分** 和 **up部分** 所需的预激活值。其计算公式为
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}\bigr)
        $$
        **2. 激活**

        将 $\mathbf{H}_e$ 沿列维度拆分为 gate 分支 $\mathbf{G}_e$ 和 up 分支 $\mathbf{U}_e$，根据 `activation` 计算中间激活表示 $\mathbf{A}_e$；各可选激活类型的计算方式参见[激活函数公式](#activation-formulas)。

        **3. Linear2 投影**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

    - **Token Combine**

        Combine负责收集所有专家计算出的输出向量，按照每个token原先分配到的专家权重进行加权求和，最终为每个token生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有token的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。
    </details>

    <details>
    <summary> A8W8-INT量化场景</summary>

    - **输入**

      - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `l1_weights_sf`。
      - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `l2_weights_sf`。

    - **EP Dispatch**

        在Dispatch通信之前，首先将原始BF16 激活矩阵 $\mathbf{X}$ 量化为int8。对每个 $\text{token}_i$，计算其逐token缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到int8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在Dispatch通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$（$e_{i,k} = \mathbf{E}[i,k]$）。

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的token索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的token总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过Dispatch后收到的全部token表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

    - **Expert Compute**

        在MoE层中，每个专家本质上是一个独立的前馈网络（FFN），采用SwiGLU结构以提升表达能力。在A8W8场景下，两个线性层都使用int8 输入和int8 权重进行矩阵乘，得到int32 中间结果并反量化。具体分为三个子步骤。

        **1. Linear1 投影（int8 矩阵乘 + 反量化）**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate部分** 和 **up部分** 所需的预激活值。计算时执行int8 矩阵乘法，得到int32 计算结果：
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$
        然后反量化为预激活值 $\mathbf{H}_e$：
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        **2. 激活**

        将 $\mathbf{H}_e$ 沿列维度拆分为 gate 分支 $\mathbf{G}_e$ 和 up 分支 $\mathbf{U}_e$，根据 `activation` 计算中间激活表示 $\mathbf{A}_e$；各可选激活类型的计算方式参见[激活函数公式](#activation-formulas)。

        **3. Linear2 投影（量化 + int8 矩阵乘 + 反量化）**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        在A8W8 场景下，需要将激活值量化为int8，因此先对 $\mathbf{A}_e$ 的每一行（每个token）计算缩放因子：
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        得到专家 $e$ 在Linear2 计算时的激活缩放因子 $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$。然后量化：
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$
        再执行int8 矩阵乘法并反量化：
        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

    - **Token Combine**

        Combine负责收集所有专家计算出的输出向量，按照每个token原先分配到的专家权重进行加权求和，最终为每个token生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有token的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。

    </details>

    <details>
    <summary> A8W4-INT量化场景</summary>

    - **输入**
      - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `l1_weights_sf`。
      - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `l2_weights_sf`。
      - $\mathbf{B}_1 \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 的偏置矩阵，由int4 量化过程离线生成，对应入参 `l1_bias`。
      - $\mathbf{B}_2 \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 的偏置矩阵，由int4 量化过程离线生成，对应入参 `l2_bias`。

    - **EP Dispatch**

        在Dispatch通信之前，首先将原始BF16 激活矩阵 $\mathbf{X}$ 量化为int8。对每个 $\text{token}_i$，计算其逐token缩放因子：
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        然后量化得到int8 表示：
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        在Dispatch通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$

        记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的token索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的token总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过Dispatch后收到的全部token表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

        对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

    - **Expert Compute**

        在MoE层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。在A8W4-INT场景下，两个线性层都采用MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案进行矩阵乘，通过将int8 值拆为高4位和低4位两个有符号int4，使得int8×int4 的矩阵乘可分解为两个int4×int4 的矩阵乘，从而利用硬件的int4 矩阵乘加速。该方案的数学原理和实现逻辑可参阅[GroupedMatmul W4A8量化与MSD方案](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81)。

        **1. 生成精度补偿的偏置矩阵（离线生成，在算子外完成，并作为算子输入）**

        MSD方案的核心是将量化后的int8 激活值二进制重解释地拆分为两个int4 分量。由于低位int4 分量按 $(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$ 定义，其数值范围被映射到 $[-8, 7]$，这相当于在原始无符号低4 位（0~15）的基础上减去了8。当将高、低位的int4 分别与权重矩阵做矩阵乘并合并时，该偏移会引入一个常数项，需要在最终结果中予以补偿。

        记原始int8 激活矩阵为 $\mathbf{X}^{\text{int8}}$，拆分后的高位和低位int4 激活矩阵分别为 $\mathbf{X}_1^{\text{int4}}$ 和 $\mathbf{X}_2^{\text{int4}}$，权重矩阵为 $\mathbf{W}$。恢复关系为：

        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        其中 $\mathbf{1}_{\text{mat}}$ 为与 $\mathbf{X}_2^{\text{int4}}$ 形状相同的全1 矩阵，用于逐元素加8。令 $\mathbf{1}$ 为形状与输入特征维度一致的全1 列向量，则矩阵乘展开为：

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        这里 $\mathbf{1}^\top \cdot \mathbf{W}$ 表示对权重矩阵 $\mathbf{W}$ 的每一列求和，其结果为一个行向量（形状与输出维度相同）。在具体实现中，该行向量会被广播到批次维度，加到所有token的输出上。可见，若只计算前两项，会缺失一个正数项。为了精确恢复结果，需预先计算补偿偏置：

        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        $\mathbf{B}_1$ 和 $\mathbf{B}_2$ 均按此法，分别使用对应的全1 列向量（维度分别为 $\text{hidden}$ 和 $\text{intermediate\_hidden}$）与各自的权重矩阵相乘，得到形状匹配的偏置行向量。这些偏置在离线阶段计算完成后传入算子，在合并高低位结果时直接参与加法，从而消除偏移引入的误差。

        **2. Linear1 投影（int4 × int4 矩阵乘 + 反量化）**

        Linear1 投影是专家网络的第一层线性变换，同时产生 **gate部分** 和 **up部分** 所需的预激活值。

        **(1) 激活重解释为int4**

        将int8 激活张量 $\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 二进制重解释为两个int4 交替拼接的视图：

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        其中高位和低位分量直接由 $\mathbf{Q}_e^{\text{int4}}$ 的偶数行和奇数行给出：

        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        它们在数值上可以由原int8 经如下计算得到：

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        恢复关系为 $\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\mathbf{Q}_e^{\text{int4}}$ 与 $\mathbf{Q}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

        **(2) int4 × int4 矩阵乘与权重反量化**

        将重解释后的int4 激活视图 $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ 与权重矩阵 $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediate\_hidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W1}_e$ 进行反量化，得到结果：

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

        **3. 激活**

        将 $\mathbf{H}_e$ 沿列维度拆分为 gate 分支 $\mathbf{G}_e$ 和 up 分支 $\mathbf{U}_e$，根据 `activation` 计算中间激活表示 $\mathbf{A}_e$；各可选激活类型的计算方式参见[激活函数公式](#activation-formulas)。

        **4. Linear2 投影（量化 + int4 × int4 矩阵乘 + 反量化）**

        Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
        在A8W4 场景下，首先将激活值量化为int8，再通过二进制重解释为两个int4 的拼接视图，以一次矩阵乘完成计算，过程如下。

        **(1) 激活量化**

        对 $\mathbf{A}_e$ 的每一行计算缩放因子并量化至int8：
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        得到专家 $e$ 在Linear2 计算时的激活缩放因子 $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$，并计算量化结果：
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$

        **(2) 激活重解释为int4**

        将int8 激活张量 $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}$ 二进制重解释为两个int4 交替拼接的视图：

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}
        $$

        其中高位和低位分量直接由 $\mathbf{A}_e^{\text{int4}}$ 的偶数行和奇数行给出：

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}
        $$

        它们在数值上可以由原int8 经如下计算得到：

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        恢复关系为 $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\mathbf{A}_e^{\text{int4}}$ 与 $\mathbf{A}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

        **(3) int4 × int4 矩阵乘与权重反量化**

        将重解释后的int4 激活视图 $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}$ 与权重矩阵 $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediate\_hidden} \times \text{hidden}}$ 执行矩阵乘，并应用权重缩放因子 $\mathbf{s}^{W2}_e$ 进行反量化，得到结果：

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

        经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

    - **Token Combine**

        Combine负责收集所有专家计算出的输出向量，按照每个token原先分配到的专家权重进行加权求和，最终为每个token生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

        所有token的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。

    </details>

    <!-- npu="950" id23 -->
    <details>
    <summary> A8W8-FP量化场景</summary>

    第一阶段对输入Token按专家分组收集后做MXFP8 量化，生成各专家的量化输入与缩放因子：

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    说明：根据 `topk_ids` 将 Token 按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的 Token 索引集合，$E_{local}$表示当前专家收到的最大token数，每个专家数值可能不同，$X[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示 MX 逐组量化（group size = 32），对每组 32 个元素提取共享指数后量化为 FP8 目标类型（FLOAT8_E5M2 或 FLOAT8_E4M3FN），同时输出 FLOAT8_E8M0 缩放因子。量化后的数据将作为 GMM1 的输入。

    第二阶段对每个专家执行GMM1 矩阵乘法（将 $W_1$ 沿列方向分为两半分别计算）、SwiGLU激活和MX量化：

    $$
    G_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(G)}, S_{1,e}^{(G)}), \quad U_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(U)}, S_{1,e}^{(U)})
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e)
    $$

    $$
    \hat{A}_e,\ S_{A,e} = \mathrm{Q}_{\text{MX}}(A_e)
    $$

    说明：将 $W_1$ 的前 $N/2$ 列 $W_{1,e}^{(G)}$ 和后 $N/2$ 列 $W_{1,e}^{(U)}$ 分别与MX反量化后的输入做矩阵乘法，得到gate分支 $G_e$ 和up分支 $U_e$。SwiGLU的计算方式参见[激活函数公式](#activation-formulas)，其输出维度为 $N/2$。随后对激活输出做MX量化，得到GMM2 的量化输入 $\hat{A}_e$。

    第三阶段对每个专家执行GMM2 矩阵乘法，并将结果按目标Rank分发：

    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{A}_e, S_{A,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    说明：将量化后的SwiGLU输出与第二组权重 $W_2$ 做MX反量化后的矩阵乘法，将 $N/2$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $O_e$。计算完成后通过RDMA peermem将结果按目标Rank的专家偏移地址写入远端，实现跨Rank聚合。

    第四阶段对所有 Token 按路由权重加权求和，并叠加共享专家输出，恢复为与输入相同形状的输出：

    当启用共享专家（`shared_expert_num_per_rank` > 0）时，共享专家在每张卡上对本卡全部 token 本地执行与路由专家相同的 GMM1 + SwiGLU + GMM2 计算，使用 `shared_l1_weights`、`shared_l2_weights`、`shared_l1_weights_sf`、`shared_l2_weights_sf`，无需参与 Dispatch 通信。各共享专家的输出记为 $O^{\mathrm{shared}}_s$，$s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$。

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i]
    $$

    说明：对每个 Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按 `topk_weights` 中的权重逐元素加权累加，再直接加上各共享专家对当前 token 的输出，得到最终输出 $Y$。未启用共享专家时，共享专家求和项为零。

    其中，$X$ 表示参数 `x`，$W$ 表示参数 `topk_weights`，$W_1$ 表示参数 `l1_weights`，$W_2$ 表示参数 `l2_weights`，$Y$ 表示参数 `y`，$E_{\text{local}}$ 表示 `local_moe_expert_num = num_experts / ep_world_size`（每个 Rank 的路由 MoE 专家数），$K$ 表示 `topk_ids` 的第二维度。

    局部变量说明：
    - $\mathcal{T}_e$：被路由到专家 $e$ 的 Token 索引集合，由 `topk_ids` 排序后确定。
    - $\hat{X}_e,\ S_{X,e}$：专家 $e$ 的量化输入及其 MX 缩放因子，第一阶段中间结果。
    - $W_{1,e}^{(G)}$、$W_{1,e}^{(U)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由 `l1_weights` 按gate分支和up分支拆分推导。
    - $S_{1,e}^{(G)}$、$S_{1,e}^{(U)}$：$W_{1,e}^{(G)}$ 和 $W_{1,e}^{(U)}$ 对应的 MX 缩放因子，从 `l1_weights_sf` 按维度截取。
    - $S_{2,e}$：$W_{2,e}$ 对应的 MX 缩放因子，来自参数 `l2_weights_sf`。
    - $G_e,\ U_e$：GMM1 的gate分支和up分支输出，中间结果。
    - $A_e$：SwiGLU激活输出，维度 $m_e \times N/2$，中间结果。
    - $\hat{A}_e,\ S_{A,e}$：量化后的SwiGLU输出及其MX缩放因子，中间结果。
    - $O_e$：GMM2 的专家级输出，维度 $m_e \times H$，中间结果。
    - $\pi(i, k)$：Token $i$ 的第 $k$ 个top-k专家在展开排序后的行索引，由路由排序确定。
    - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX逐组量化操作，block size = 32，输出FP8 数据和E8M0 缩放因子。
    - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX逐组反量化操作，在matmul内部隐式执行。
    </details>

    <details>
    <summary> A8W4-FP 量化场景</summary>

    第一阶段（Token 选择、量化与 Dispatch）：

    对本 rank 的输入 Token \(X \in \mathbb{R}^{B\times H}\)，根据 `topk_ids` 得到每个专家 \(e\) 对应的 Token 下标集合 \(T_e\)，并将选中的 BF16 Token 按 32 个元素一组量化为 MXFP8 E4M3：

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    其中，\(\hat{X}_e\) 的数据类型为 `FLOAT8_E4M3FN`，\(S_{X,e}\) 的数据类型为 `FLOAT8_E8M0`。随后将量化后的 Token 及其缩放因子发送到专家所在 rank。

    路由 MoE 专家的第一层和第二层权重 $W_{1,e}$、$W_{2,e}$ 均为 MXFP4 E2M1 数据，缩放因子为 E8M0，$e$ 的范围为 $[0,\text{local\_moe\_expert\_num})$。共享专家权重单独由 `shared_l1_weights`、`shared_l2_weights` 提供，其第一维为 `shared_expert_num_per_rank`。A8W4 kernel 在矩阵乘 Prologue 中处理 FP4 权重，送入矩阵乘的逻辑数据流为 FP8 激活乘 FP4 权重。

    第二阶段（GMM1、SwiGLU 与再次量化）：

    对专家 \(e\) 收到的 Token，第一层分组矩阵乘和 SwiGLU 计算为：

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e).
    $$

    SwiGLU的计算方式参见[激活函数公式](#activation-formulas)。其输出继续按 32 个元素一组量化为 MXFP8 E4M3，供第二层矩阵乘使用：

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    第三阶段（GMM2）：

    第二层矩阵乘仍为 A8W4，即 FP8 E4M3 激活乘 FP4 E2M1 权重：

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    当启用共享专家（`shared_expert_num_per_rank` > 0）时，共享专家在每张卡上对本卡全部 token 本地执行与路由专家相同的 GMM1 + SwiGLU + GMM2 计算，使用 `shared_l1_weights`、`shared_l2_weights`、`shared_l1_weights_sf`、`shared_l2_weights_sf`，无需参与 Dispatch 通信。各共享专家的输出记为 $O^{\mathrm{shared}}_s$，$s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$。

    第四阶段（Combine 与加权合并）：

    将各专家输出送回原 rank，并根据 `topk_weights` 做加权合并，并叠加共享专家输出：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i],
    $$

    未启用共享专家时，共享专家求和项为零。最终输出 $Y$ 的数据类型为 BF16。A8W4-FP 的主要数据类型流为：`BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`。
    </details>

    <details>
    <summary> A4W4-FP 量化场景</summary>

    第一阶段（Token 选择、量化与 Dispatch）：

    对每个专家 \(e\) 对应的 Token 集合 \(T_e\)，将选中的 BF16 Token 按 32 个元素一组量化为 MXFP4 E2M1：

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    其中，\(\hat{X}_e\) 的数据类型为 `FLOAT4_E2M1`，\(S_{X,e}\) 的数据类型为 `FLOAT8_E8M0`。路由 MoE 专家的第一层和第二层权重 \(W_{1,e}\)、\(W_{2,e}\) 均为 MXFP4 E2M1，\(e\) 的范围为 \([0,\text{local\_moe\_expert\_num})\)；共享专家权重的第一维为 `shared_expert_num_per_rank`。权重缩放因子为 E8M0。

    第二阶段（A4W4 GMM1、SwiGLU 与输出类型提升）：

    第一层分组矩阵乘为 A4W4：

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{SwiGLU}(G_e,U_e).
    $$

    SwiGLU的计算方式参见[激活函数公式](#activation-formulas)。这里不能继续把SwiGLU输出量化为FP4。kernel在 `QuantMode == E2M1_QUANT` 时，将 `SwigluQuantOutType` 指定为 `fp8_e4m3fn_t`，因此输出会提升为MXFP8 E4M3：

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    第三阶段（A8W4 GMM2）：

    由于 SwiGLU 量化输出为 FP8 E4M3，而第二层权重仍为 FP4 E2M1，因此第二层矩阵乘实际为 A8W4，而不是 A4W4：

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    当启用共享专家（`shared_expert_num_per_rank` > 0）时，共享专家在每张卡上对本卡全部 token 本地执行与路由专家相同的 GMM1 + SwiGLU + GMM2 计算，使用 `shared_l1_weights`、`shared_l2_weights`、`shared_l1_weights_sf`、`shared_l2_weights_sf`，无需参与 Dispatch 通信。各共享专家的输出记为 $O^{\mathrm{shared}}_s$，$s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$。

    第四阶段（Combine 与加权合并）：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i],
    $$

    未启用共享专家时，共享专家求和项为零。最终输出 $Y$ 的数据类型为 BF16。A4W4-FP 的完整数据类型流为：`BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`。其中所有 MX 缩放因子的类型均为 `FLOAT8_E8M0`，量化粒度均为 32 个连续元素。
    </details>
    <!-- end id23 -->

## 函数原型

先调用get_symm_buffer_for_mega_moe接口封装输入参数并创建SymmBuffer结构体，再调用mega_moe接口进行计算。

```python
get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, *, max_recv_token_num=0, dispatch_quant_mode=0, dispatch_quant_out_dtype=None, combine_quant_mode=0, comm_alg="", topk_weights_type=0) -> SymmBuffer
```

```python
mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, activation_params=None, weight1_type=None, weight2_type=None, shared_l1_weights=None, shared_l2_weights=None, shared_l1_weights_sf=None, shared_l2_weights_sf=None, shared_l1_bias=None, shared_l2_bias=None) -> (Tensor, Tensor)
```

### 弹性扩缩容接口

```python
sym_buffer.query_mask_buffer(mask_status) -> None
sym_buffer.update_mask_buffer(rank, masked) -> None
sym_buffer.clean_mask_buffer() -> None
sym_buffer.get_local_buffer_tensor(dtype, size=None, offset=0) -> Tensor
sym_buffer.update_group(group) -> None
```

`mask_buffer` 及以上五个 `SymmBuffer` 弹性扩缩容接口当前仅支持 <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。`SymmBuffer` 默认不创建 `mask_buffer`，因此各代际默认均不会向算子传入该参数。第一次调用 `query_mask_buffer`、`update_mask_buffer` 或 `clean_mask_buffer` 时，才会在 NPU 上创建 shape 为 `[ep_world_size]` 的全 0 int32 掩码；0 表示正常 rank，1 表示失能 rank，失能 rank 会在后续计算和卡间通信中被跳过。

- `query_mask_buffer` 接收调用者预先创建的 `mask_status`，其 dtype 必须为 `torch.int32`，shape、所在 NPU 必须与内部 `mask_buffer` 一致。接口通过 D2D 拷贝将当前掩码写入 `mask_status`，不返回 Tensor。
- `update_mask_buffer` 只更新当前进程中的本地掩码，各 rank 的掩码一致性由调用者保证。当恢复 rank，即 `masked` 输入为 `False` 时，必须调用 `get_local_buffer_tensor` 清空算子现存通信缓存区标志位。样例如下：

  ```python
  symm_buffer.update_mask_buffer(0, False)  # 恢复 rank 0 通信
  symm_buffer.update_mask_buffer(1, False)  # 恢复 rank 1 通信
  local_buffer = symm_buffer.get_local_buffer_tensor(torch.uint8)
  local_buffer.zero_()
  ```

- `clean_mask_buffer` 在当前 NPU stream 上将本地掩码的所有元素清零；各 rank 需要分别调用，跨 rank 的一致性仍由调用者保证。掩码尚未创建时，接口会创建全 0 掩码。
- `get_local_buffer_tensor` 将当前 Rank 的本地 CCL Buffer 零拷贝包装为 NPU Tensor。`offset` 以 `dtype` 元素为单位，`size` 为 `None` 时返回从 `offset` 到 Buffer 末尾的一维视图，否则返回指定 shape。返回 Tensor 不持有底层内存，其生命周期不得超过 `SymmBuffer`；调用 `update_group` 后旧视图失效，必须重新获取。调用者写入原始通信 Buffer 前必须保证相关算子已经执行完成，并确保写入范围正确。
- `update_group` 使用新 group 完整重建通信链路；旧通信链路的算子执行完成以及新 group 与现有 mask shape 的一致性需要调用者保证。如需清除失能状态，应显式调用 `clean_mask_buffer`。

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
        <td>通信域内各Rank可能出现的最大单卡token数。A5支持各Rank的实际token数不同，每次调用需满足x.shape[0]不大于该值；所有Rank必须配置相同的上界。</td>
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
        <td>SwiGLU激活后的中间特征维度。Linear1同时生成gate和up两个分支，因此Linear1的完整输出宽度为2 × intermediate_hidden。</td>
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
        <td>dispatch通信时量化模式。0表示非量化（A16W16场景），2表示int8量化（A8W8-INT、A8W4-INT场景），4表示MXFP量化（A8W8-FP、A8W4-FP、A4W4-FP场景）。各产品支持的取值见约束说明。默认值为0。</td>
    </tr>
    <tr>
        <td>dispatch_quant_out_dtype</td>
        <td>torch.dtype</td>
        <td>可选</td>
        <td>dispatch量化后输出的数据类型。支持torch.int8、torch.float8_e5m2、torch.float8_e4m3fn、torch.float4_e2m1。各产品支持的取值见约束说明。默认值为None。</td>
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
    <tr>
        <td>topk_weights_type</td>
        <td>int</td>
        <td>可选</td>
        <td>topkWeights前移开关。0表示关闭，1表示开启（将topkWeights随token数据一起在dispatch阶段提前发送至目标rank，减少combine阶段通信量）。默认值为0。</td>
    </tr>
</tbody>
</table>

### mega_moe

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持参数表上角标<sup>1</sup>的描述，不支持的参数使用默认值即可。
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持参数表上角标<sup>2</sup>的描述，不支持的参数使用默认值即可。
<!-- end id10 -->

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
        <td>bfloat16</td>
        <td>ND</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>topk_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>专家索引矩阵，表示每个token选择的num_topk个专家。元素取值范围为<code>[0, num_experts)</code>，且同一token选择的num_topk个专家不能重复。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示MoE模型的专家门控网络为当前输入Token选出的num_topk个专家所对应的门控权重系数。</td>
        <td>bfloat16、float32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td rowspan="6">l1_weights</td>
        <td rowspan="6">list[Tensor]</td>
        <td rowspan="6">必选</td>
        <td rowspan="6">MoE专家网络第一线性层的权重矩阵（包括门控与上投影），用于将输入映射至中间维度，输出供给激活函数。单卡MoE 专家数为 <code>local_moe_expert_num</code>。</td>
        <td>bfloat16<sup>2</sup></td>
        <td>ND</td>
        <td>(hidden, 2 × intermediate_hidden)</td>
    </tr>
    <tr>
        <td>int8<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(hidden, 2 × intermediate_hidden)</td>
    </tr>
    <tr>
        <td>int4(int32)<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(hidden, 2 × intermediate_hidden // 8)</td>
    </tr>
    <tr>
        <td>float8_e5m2<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>float8_e4m3fn<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>float4_E2M1<sup>1</sup></td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td rowspan="6">l2_weights</td>
        <td rowspan="6">list[Tensor]</td>
        <td rowspan="6">必选</td>
        <td rowspan="6">MoE专家网络第二线性层的权重矩阵，负责将激活后的中间特征投影回隐藏维度。数据类型与l1_weights一致。单卡 MoE 专家数为 <code>local_moe_expert_num</code>。</td>
        <td>bfloat16<sup>2</sup></td>
        <td>ND</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>int8<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>int4<sup>2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>(intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>float8_e5m2<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>float8_e4m3fn<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>float4_E2M1<sup>1</sup></td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
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
        <td rowspan="2">MoE专家网络第一线性层的权重矩阵的量化缩放因子。</td>
        <td>uint64<sup>2</sup></td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>float8_e8m0<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)</td>
    </tr>
    <tr>
        <td rowspan="2">l2_weights_sf</td>
        <td rowspan="2">list[Tensor]</td>
        <td rowspan="2">可选</td>
        <td rowspan="2">MoE专家网络第二线性层的权重矩阵的量化缩放因子。</td>
        <td>uint64<sup>2</sup></td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>float8_e8m0<sup>1</sup></td>
        <td>ND</td>
        <td>(num_experts_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>l1_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>MoE专家网络第一线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
        <td>float32</td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>l2_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>MoE专家网络第二线性层的偏置，仅于A8W4-INT量化场景下需要该参数，用于精度补偿。</td>
        <td>float32</td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>x_active_mask<sup>2</sup></td>
        <td>Tensor</td>
        <td>可选</td>
        <td>表示token是否参与通信。</td>
        <td>int8</td>
        <td>ND</td>
        <td>(num_tokens, )</td>
    </tr>
    <tr>
        <td>activation</td>
        <td>str</td>
        <td>可选</td>
        <td>激活函数类型，默认值为"swiglu"。可选值为"swiglu"、"swiglustep"、"swigluoai"和"situglu"。</td>
        <td>str</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>activation_clamp</td>
        <td>float</td>
        <td>可选</td>
        <td>"swiglu"、"swiglustep"和"swigluoai"的截断值，未配置时使用float最大值。"situglu"忽略该值。值需≥0且不能为NaN。</td>
        <td>float</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>activation_params</td>
        <td>dict[str, float]</td>
        <td>可选</td>
        <td>激活函数参数字典，默认值为None。"swiglu"和"swiglustep"无需设置；"swigluoai"支持"alpha"和"beta"，默认值分别为1.702和1.0；"situglu"支持"beta"和"linear_beta"，"beta"默认值为1.0，"linear_beta"未配置时保持up分支不变。</td>
        <td>float</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <!-- npu="950" id22 -->
    <tr>
        <td>weight1_type</td>
        <td>int</td>
        <td>可选</td>
        <td>权重1的逻辑数据类型。A8W8-FP场景可选，默认从l1_weights的Tensor数据类型推导；A8W4-FP和A4W4-FP场景必须设置为float4_e2m1fn_x2对应的类型枚举。取值必须与weight2_type一致。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>weight2_type</td>
        <td>int</td>
        <td>可选</td>
        <td>权重2的逻辑数据类型。A8W8-FP场景可选，默认从l2_weights的Tensor数据类型推导；A8W4-FP和A4W4-FP场景必须设置为float4_e2m1fn_x2对应的类型枚举。取值必须与weight1_type一致。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <!-- end id22 -->
    <tr>
        <td rowspan="3">shared_l1_weights<sup>1</sup></td>
        <td rowspan="3">list[Tensor]</td>
        <td rowspan="3">可选</td>
        <td rowspan="3">共享专家网络第一线性层的权重矩阵（包括门控与上投影），用于将输入映射至中间维度，输出供给激活函数。</td>
        <td>FLOAT8_E5M2</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT4_E2M1</td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td rowspan="3">shared_l2_weights<sup>1</sup></td>
        <td rowspan="3">list[Tensor]</td>
        <td rowspan="3">可选</td>
        <td rowspan="3">共享专家网络第二线性层的权重矩阵，负责将激活后的中间特征投影回隐藏维度。数据类型与l1_weights一致。</td>
        <td>FLOAT8_E5M2</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>FLOAT4_E2M1</td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>shared_l1_weights_sf<sup>1</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>共享专家网络第一线性层的权重矩阵的量化缩放因子。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>shared_l2_weights_sf<sup>1</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>共享专家网络第二线性层的权重矩阵的量化缩放因子。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>shared_l1_bias<sup>1，2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>共享专家网络第一线性层的偏置，暂不支持。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>shared_l2_bias<sup>1，2</sup></td>
        <td>list[Tensor]</td>
        <td>可选</td>
        <td>共享专家网络第二线性层的偏置，暂不支持。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
</tbody>
<tfoot>
<tr>
    <td colspan="7">表格中的<code>CeilDiv(<var>x</var>, <var>y</var>) = ⌈<var>x</var> / <var>y</var>⌉ = ⌊(<var>x</var> + <var>y</var> - 1) / <var>y</var>⌋</code></td>
</tr>
<tr>
    <td colspan="7">表格中用T1(T2)表示数据类型T1在传入前要求重解释为另一个数据类型T2再传入，例如，int4(int32)表示实际int4的数据，在传入需重解释为int32传入，其shape为重解释后的shape。</td>
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
        <td>用于封装输入参数并生成 <code>context</code>、<code>ep_world_size</code> 和 <code>ccl_buffer_size</code>。</td>
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
        <td>本卡收到的token数据，对应公式中的Y，数据类型与输入 <code>x</code> 保持一致。要求为2维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡每个专家实际收到的token数量。要求为1维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>int32</td>
        <td>(local_moe_expert_num,)</td>
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
            <td>num_experts_per_rank（Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品及Ascend 950PR/Ascend 950DT的逐专家二维Tensor布局）或1（Ascend 950PR/Ascend 950DT的单个三维堆叠Tensor布局）</td>
            <td>否（bfloat16/int8/int4场景）/是（float8_e5m2/float8_e4m3fn/float4_E2M1场景）</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l2_weights</td>
            <td>num_experts_per_rank（Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品及Ascend 950PR/Ascend 950DT的逐专家二维Tensor布局）或1（Ascend 950PR/Ascend 950DT的单个三维堆叠Tensor布局）</td>
            <td>否（bfloat16/int8/int4场景）/是（float8_e5m2/float8_e4m3fn/float4_E2M1场景）</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l1_weights_sf</td>
            <td>与对应权重的TensorList长度和布局一致；Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品为num_experts_per_rank</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>l2_weights_sf</td>
            <td>与对应权重的TensorList长度和布局一致；Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品为num_experts_per_rank</td>
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

  <!-- npu="950" id21 -->
  - Ascend 950PR/Ascend 950DT的MXFP场景支持两种TensorList布局：逐专家布局使用`local_moe_expert_num`个二维Tensor；堆叠布局使用仅含一个三维Tensor的list，三维Tensor的dim0为`local_moe_expert_num`。`l1_weights`、`l2_weights`、`l1_weights_sf`和`l2_weights_sf`必须采用同一种布局；启用共享专家时，对应的四个共享专家输入也必须采用与MoE专家相同的布局。
  <!-- end id21 -->

- **参数一致性约束**：
  - mega_moe接口的所有输入参数及其对应的张量维度，必须与get_symm_buffer_for_mega_moe的同名参数（例如 `num_experts`、`hidden`、`intermediate_hidden` 等）保持一致。
  - 调用算子过程中使用的`num_experts`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_dtype`、`num_max_tokens_per_rank`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域和组网约束**：
    - 所有卡的`ep_world_size`参数取值需保持一致。
    - Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品的通信域缓存区大小应当一致。`ccl_buffer_size` 为 HBM 上分配的 CCL 通信缓冲区**总大小**（Bytes），包含等大小的 **windowIn** 和 **windowOut** 两块空间，校验时以单个空间 `ccl_buffer_size / 2` 为准，需满足：

        $$ccl\_buffer\_size\ /\ 2 \ge \mathrm{offsetTokenPerExpert} + \mathrm{offsetTensor} + \mathrm{offsetFlag} + 10\,\mathrm{MB}$$
    <!-- npu="910b" id11 -->
     **Atlas A2 训练系列产品/Atlas A2 推理系列产品：**

    ```text
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
    <!-- end id11 -->

    <!-- npu="A3" id12 -->
     **Atlas A3 训练系列产品/Atlas A3 推理系列产品：**

    ```text
    offsetTokenPerExpert = ep_world_size × CeilAlign(ep_world_size × maxExpertPerRank + 1, 128) × 4B

    // winIn（仅winIn，无winOut）
    offsetAAfterDispatch = num_max_tokens_per_rank × num_topk × (quant ? hidden + 512 : hidden × 2)
    offsetD              = num_max_tokens_per_rank × num_topk × hidden × 2B
    winInTensorSize      = offsetAAfterDispatch + offsetD

    offsetTensor         = winInTensorSize
                        + (quant ? num_max_tokens_per_rank × num_topk × 4B : 0)

    // sync flags
    syncStateReservedSize = 512KB
    offsetFlag            = max(ep_world_size × 512B, syncStateReservedSize)
    ```
    <!-- end id12 -->

    <!-- npu="950" id13 -->
     **Ascend 950PR/Ascend 950DT：**

    通信buffer由`get_symm_buffer_for_mega_moe`根据通信域配置、`num_max_tokens_per_rank`等参数自动计算并申请，用户无需自行计算或设置`ccl_buffer_size`。`num_max_tokens_per_rank`越大，内部申请的通信内存越多，建议按预期最大单卡token数合理设置。
    <!-- end id13 -->
  - 通信域各节点的驱动版本应当相同。
  <!-- npu="910b" id14 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机通信域要求交换机组网，不支持双机直连组网。
  <!-- end id14 -->
  <!-- npu="A3" id15 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：多机通信域要求在一个超节点内，不支持双机直连组网和跨超节点组网。
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持UB Memory通信协议。
  <!-- end id16 -->

- **参数约束**：

  <!-- npu="A3,910b" id17 -->
  - **Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：**

    - 各卡 `num_tokens` 需保持一致。
    - `ep_world_size`：取值为 `2`、`4`、`8`、`16`、`32`。
    - `num_experts`：取值范围为`world_size ≤ num_experts ≤ 1024`，且 `num_experts % ep_world_size == 0`。
    - `num_experts_per_rank`：取值范围为 `1 ≤ num_experts_per_rank ≤ 128`，且 `num_experts_per_rank = num_experts / world_size`。
    - `num_max_tokens_per_rank`：取值范围为 `1 ≤ num_max_tokens_per_rank ≤ 4096`。
    - `max_recv_token_num` 需大于0，输入0表示自动计算，公式为 `num_tokens × ep_world_size × min(num_topk, local_moe_expert_num)`。
    - `num_topk`：取值范围为 `1 ≤ num_topk ≤ 16`。
    - `hidden`：取值范围为 `1024 ≤ hidden ≤ 8192`，且 `hidden % 512 == 0`。
    - `intermediate_hidden`：取值范围为 `512 ≤ intermediate_hidden ≤ 3072`，且 `intermediate_hidden % 512 == 0`。
    - `dispatch_quant_mode`：取值范围为 `0`（非量化）、`2`（pertoken量化）。
    - `dispatch_quant_out_dtype`：取值为 `torch.int8`。
    - 支持三种计算场景（A16W16、A8W8-INT、A8W4-INT），不同场景下可选入参（缩放因子、偏置等）的必需性及数据类型有严格配套要求。调用时必须根据所选场景完整提供对应参数，不可混用或遗漏，配套关系见下表。
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
            <td><strong>A16W16</strong></td>
            <td>bfloat16</td>
            <td>bfloat16</td>
            <td>bfloat16</td>
            <td>–</td>
            <td>–</td>
            <td>–</td>
            <td>–</td>
            <td>bfloat16</td>
            <td>0</td>
            <td>–</td>
            </tr>
            <tr>
            <td><strong>A8W8-INT</strong></td>
            <td>bfloat16</td>
            <td>int8</td>
            <td>int8</td>
            <td>uint64</td>
            <td>uint64</td>
            <td>–</td>
            <td>–</td>
            <td>bfloat16</td>
            <td>2</td>
            <td>torch.int8</td>
            </tr>
            <tr>
            <td><strong>A8W4-INT</strong></td>
            <td>bfloat16</td>
            <td>int4</td>
            <td>int4</td>
            <td>uint64</td>
            <td>uint64</td>
            <td>float32</td>
            <td>float32</td>
            <td>bfloat16</td>
            <td>2</td>
            <td>torch.int8</td>
            </tr>
        </tbody>
        <tfoot>
            <tr>
                <td colspan="11">“–”表示该场景下<strong>不需要</strong>提供该参数，传入 <code>None</code> 或保持默认即可。</td>
            </tr>
            <tr>
                <td colspan="11">直接填写数据类型（如 <code>uint64</code>）表示该场景下该参数为<strong>必选</strong>，且必须使用该数据类型</td>
            </tr>
        </tfoot>
        </table>

  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - **Ascend 950PR/Ascend 950DT：**
    - `activation`支持"swiglu"、"swiglustep"、"swigluoai"和"situglu"。`activation_clamp`用于配置前三种激活的截断值；`activation_params`用于配置"swigluoai"的`alpha`、`beta`以及"situglu"的`beta`、`linear_beta`。
    - num_tokens（x.dim0）范围[1, +∞)，每次调用必须不大于创建`sym_buffer`时配置的`num_max_tokens_per_rank`。不同Rank的实际num_tokens可以不同，同一个`sym_buffer`也可以用于多次不同num_tokens的调用。
    - `num_max_tokens_per_rank`必须大于等于1，所有Rank取值必须一致，建议设置为`sym_buffer`复用期间所有Rank可能出现的最大单卡token数。超过原上界时需使用更大的上界重新创建`sym_buffer`。
    - hidden（x.dim1）范围[1024, 8192]。普通MTE权重格式要求32对齐；FLOAT4_E2M1的FORMAT_FRACTAL_NZ_C0_32格式要求64对齐。
    - num_topk（topk_ids.dim1）支持[1, 32]。
    - num_experts_per_rank 范围 [1, 1024]。
    - intermediate_hidden表示SwiGLU激活后的中间特征维度，范围[256, 4096]且128对齐；Linear1的完整输出宽度为2 × intermediate_hidden。
    - ep_world_size范围 [2, 1024]。
    - num_experts范围 [ep_world_size, 2048]，且num_experts % ep_world_size == 0。
    - max_recv_token_num范围 [0, num_max_tokens_per_rank × ep_world_size × min(num_topk, local_moe_expert_num)]；建议保持默认值0，由接口自动计算接收容量。
    - dispatch_quant_out_dtype仅支持torch.float8_e5m2或torch.float8_e4m3fn或torch.float4_e2m1。
    - 当前版本仅支持MXFP量化模式（dispatch_quant_mode = 4），dispatch阶段使用MX逐组量化（group size = 32），量化缩放因子类型为FLOAT8_E8M0。
    - x_active_mask和scales参数当前版本必须传入None，不支持非空输入。
    - combine_quant_mode当前支持0（非量化），3（MX模式float8_e5m2类型），4（MX模式float8_e4m3类型）。
    - comm_alg预留参数，必须为空字符串""。
    - y的数据类型与x相同。
    - l1_weights的Linear1输出维必须等于2 × intermediate_hidden，l2_weights的输入维必须等于intermediate_hidden。
    - l1_weights_sf和l2_weights_sf不可为空指针。
    - local_moe_expert_num = num_experts / ep_world_size；启用共享专家时，shared_expert_num_per_rank = shared_l1_weights.dim0；num_experts_per_rank = shared_expert_num_per_rank + local_moe_expert_num，未启用共享专家时 shared_expert_num_per_rank = 0。
    - shared_expert_num_per_rank范围 [0, 4]。
    - topo_type由通信域上下文自动推导。0表示MTE拓扑，1表示URMA跨超拓扑。当前暂不支持URMA通信方式。
    - topk_weights_type取值为0或1，0表示关闭topkWeights前移，1表示开启。当前暂不支持URMA通信方式。
    - 通信buffer由`get_symm_buffer_for_mega_moe`自动计算并申请，用户无需自行计算或设置`ccl_buffer_size`。
    - l1_weights和l2_weights的数据类型必须一致，且仅支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1。
    - topk_weights数据类型仅支持BF16或FP32。

    - **MXFP量化场景约束**：
        - l1_weights shape为(local_moe_expert_num, 2 × intermediate_hidden, hidden)，l2_weights shape为(local_moe_expert_num, hidden, intermediate_hidden)。
        - l1_weights_sf shape为(local_moe_expert_num, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)，CeilDiv(hidden, 64) = ⌈hidden / 64⌉ = ⌊(hidden + 63) / 64⌋。
        - l2_weights_sf shape为(local_moe_expert_num, hidden, CeilDiv(intermediate_hidden, 64), 2)，CeilDiv(intermediate_hidden, 64) = ⌈intermediate_hidden / 64⌉ = ⌊(intermediate_hidden + 63) / 64⌋。
        - shared_l1_weights shape为(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)，shared_l2_weights shape为(shared_expert_num_per_rank, hidden, intermediate_hidden)。
        - shared_l1_weights_sf shape为(shared_expert_num_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)，shared_l2_weights_sf shape为(shared_expert_num_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2)。
        - l1_weights_sf的dim3和l2_weights_sf的dim3必须等于2。
        - A8W4-FP场景下，FLOAT4_E2M1类型的l1_weights必须使用FORMAT_FRACTAL_NZ_C0_32格式。
        - A8W8-FP场景下，l1_weights和l2_weights必须同为FLOAT8_E5M2或同为FLOAT8_E4M3FN，`weight1_type`和`weight2_type`可省略并从权重Tensor的数据类型推导。A8W4-FP和A4W4-FP场景下，两层权重均为FLOAT4_E2M1，必须显式将`weight1_type`和`weight2_type`设置为`float4_e2m1fn_x2`对应的类型枚举，且两者一致。
        - x_active_mask和scales必须为None。
    - 支持三种计算场景（A8W8-FP、A8W4-FP、A4W4-FP），不同场景下可选入参（缩放因子、偏置等）的必需性及数据类型有严格配套要求。调用时必须根据所选场景完整提供对应参数，不可混用或遗漏，配套关系见下表。
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
            <td><strong>A8W8-FP</strong></td>
            <td>BFLOAT16</td>
            <td>FLOAT8_E5M2</td>
            <td>FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>–</td>
            <td>–</td>
            <td>BFLOAT16</td>
            <td>4</td>
            <td>torch.float8_e5m2</td>
            </tr>
            <tr>
            <td><strong>A8W8-FP</strong></td>
            <td>BFLOAT16</td>
            <td>FLOAT8_E4M3FN</td>
            <td>FLOAT8_E4M3FN</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>–</td>
            <td>–</td>
            <td>BFLOAT16</td>
            <td>4</td>
            <td>torch.float8_e4m3fn</td>
            </tr>
            <tr>
            <td><strong>A8W4-FP</strong></td>
            <td>BFLOAT16</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>–</td>
            <td>–</td>
            <td>BFLOAT16</td>
            <td>4</td>
            <td>torch.float8_e4m3fn</td>
            </tr>
            <tr>
            <td><strong>A4W4-FP</strong></td>
            <td>BFLOAT16</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>–</td>
            <td>–</td>
            <td>BFLOAT16</td>
            <td>4</td>
            <td>torch.float4_e2m1</td>
            </tr>
        </tbody>
        <tfoot>
            <tr>
                <td colspan="11">“–”表示该场景下<strong>不需要</strong>提供该参数，传入 <code>None</code> 或保持默认即可。</td>
            </tr>
            <tr>
                <td colspan="11">直接填写数据类型（如 <code>uint64</code>）表示该场景下该参数为<strong>必选</strong>，且必须使用该数据类型</td>
            </tr>
        </tfoot>
        </table>

  <!-- end id18 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  下面示例将两个接口按调用顺序串联：先初始化通信域，再用get_symm_buffer_for_mega_moe构造sym_buffer，最后调用mega_moe运行算子。

  <!-- npu="950" id19 -->
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
    scene = "A8W8"  # 可选："A8W8"、"A8W4"、"A4W4"

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def ceil(a, b):
        return (a + b - 1) // b

    def pack_fp4(weight):
        # 示例使用随机uint8模拟两个FP4 E2M1打包后的一个字节；实际使用时传入量化后的权重。
        return torch.randint(0, 256, (*weight.shape[:-1], weight.shape[-1] // 2), dtype=torch.uint8)

    def prepare_weight(weight, gmm_x_dtype):
        if scene == "A8W8":
            return weight.to(torch.float8_e5m2).npu()
        packed_weight = pack_fp4(weight).npu()
        if gmm_x_dtype == torch_npu.float4_e2m1fn_x2:
            return torch_npu.npu_format_cast(packed_weight, 29)
        return torch_npu.npu_format_cast(
            packed_weight, 29,
            customize_dtype=gmm_x_dtype,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )

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
        gmm1_x_dtype = torch_npu.float4_e2m1fn_x2 if scene == "A4W4" else torch.float8_e4m3fn
        weights1 = prepare_weight(weights1, gmm1_x_dtype)
        weights_scales1 = weights_scales1.to(torch.float8_e8m0fnu).npu()
        weights2 = prepare_weight(weights2, torch.float8_e4m3fn)
        weights_scales2 = weights_scales2.to(torch.float8_e8m0fnu).npu()
        expert_scales = expert_scales.to(torch.bfloat16).npu()

        kwargs = {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
        }
        if scene != "A8W8":
            kwargs.update(
                weight1_type=torch_npu.float4_e2m1fn_x2,
                weight2_type=torch_npu.float4_e2m1fn_x2,
            )
        return kwargs

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
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        # 步骤1：构造distribute_buffer（SymmBuffer结构体）
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens, num_topk=topK,
            hidden=H, intermediate_hidden=N // 2,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=(
                torch_npu.float4_e2m1fn_x2 if scene == "A4W4" else
                torch.float8_e4m3fn if scene == "A8W4" else torch.float8_e5m2
            )
        )
        # 步骤2：运行mega_moe，传入上一步构造的sym_buffer
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
        l1_weights = torch.randn(weight_shape, dtype=torch.float32)
        weight_scales1 = torch.randint(125, 130, weight_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        l2_weights = torch.randn(weight2_shape, dtype=torch.float32)
        weight_scales2 = torch.randint(125, 130, weight2_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [l1_weights.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [l2_weights.clone() for _ in range(rank_per_dev)]
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
  <!-- end id19 -->

  <!-- npu="A3,910b" id20 -->
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
  <!-- end id20 -->

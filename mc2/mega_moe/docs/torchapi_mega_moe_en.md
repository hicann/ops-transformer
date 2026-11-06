# mega_moe

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:58:10.776Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**:

  - The Mega MoE operator fuses the complete computation flow of the expert FFN in the MoE layer and the data communication before and after it (namely Dispatch + Linear1 + Activation + Linear2 + Combine) into a single operator, achieving overlap between communication and computation.
  - This operator provides APIs such as mega_moe and get_symm_buffer_for_mega_moe, which are used together.
  - get_symm_buffer_for_mega_moe: encapsulates input parameters and creates the SymmBuffer structure, generating information required for running the mega_moe operator, such as `context`, `ep_world_size`, and `ccl_buffer_size`.

- **Formula**:

  - Input:
    - $\mathbf{X} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$: activation matrix, corresponding to the input parameter `x`. $\text{total\_num\_tokens}$ is the global total number of tokens, and $\text{hidden}$ is the hidden dimension.
    - $\mathbf{E} \in \mathbb{Z}^{\text{total\_num\_tokens} \times \text{num\_topk}}$: matrix of expert IDs selected by tokens, corresponding to the input parameter `topk_ids`. $\text{num\_topk}$ is the number of experts selected by each token.
    - $\mathbf{G} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{num\_topk}}$: matrix of gating weights of the experts selected by tokens, corresponding to the input parameter `topk_weights`.
    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{local\_moe\_expert\_num} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$: Linear1 weights of the routed MoE experts, corresponding to the MoE expert part of the input parameter `l1_weights`.
    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{local\_moe\_expert\_num} \times \text{intermediate\_hidden} \times \text{hidden}}$: Linear2 weights of the routed MoE experts, corresponding to the MoE expert part of the input parameter `l2_weights`.
    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{shared\_expert\_num\_per\_rank} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$: Linear1 weights of the shared expert, corresponding to the input parameter `shared_l1_weights`.
    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{shared\_expert\_num\_per\_rank} \times \text{intermediate\_hidden} \times \text{hidden}}$: Linear2 weights of the shared expert, corresponding to the input parameter `shared_l2_weights`.
  - Output:

    - $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$: final output matrix, corresponding to the output parameter `y`.
  - Conventions:
    - $⋅$ denotes matrix multiplication, and $⊙$ denotes element-wise multiplication.
    - $\left \lfloor z\right \rceil$ denotes rounding $z$ to the nearest integer, and $\left \lfloor z\right \rfloor$ denotes rounding $z$ down.
    - $|z|$ denotes the absolute value, and $\max(z)$ denotes the maximum value.
    - The set of all tokens is $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{total\_num\_tokens} - 1\} \}$.
    - The token representation of $\text{token}_i$ (i.e., the hidden state vector) is $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$, and $\mathbf{x}_i = \mathbf{X}[i,:]$.
    - The expert index of $\text{token}_i$ is $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{num\_topk} - 1\},\quad e_{i,k} \in \{0,\dots,\text{num\_experts} - 1\}$.
    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$. The superscript $\text{sym}$ of $\mathbb{Z}_8^{\text{sym}}$ denotes the symmetric quantization value range: the value range is symmetrically rounded about $-127$ and $127$, which differs from the $[-128, 127]$ value range of standard int8, hence the $\text{sym}$ superscript is used for distinction.
    - Tensor slicing operations use the Python-style `start:stop:step` notation. For example, $[0::2, :]$ selects even rows, and $[1::2, :]$ selects odd rows.
    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ denotes the binary reinterpretation operation, which reinterprets the underlying binary data of tensor $\mathbf{Z}$ as the target type $T$.
  - <span id="activation-formulas">Activation function formulas:</span>
    - Let the gate branch obtained by splitting the Linear1 output be $\mathbf{G}$, the up branch be $\mathbf{U}$, and the activation output be $\mathbf{A}$. Hereafter, the specific activation computation jointly determined by `activation`, `activation_clamp`, and `activation_params` is collectively referred to as Activation. The Sigmoid function and the SiLU function are defined as:
      $$
      \sigma(z)=\frac{1}{1+e^{-z}}, \qquad
      \operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}.
      $$
      Let the clamp value be $c$; when clamping is not enabled (not configured or set to $0$), $c=+\infty$ is taken mathematically. Define
      $\mathbf{G}_c=\min(\mathbf{G},c)$, $\mathbf{U}_c=\operatorname{clip}(\mathbf{U},-c,c)$.
    - `swiglu`:
      $$
      \mathbf{A}=\operatorname{SwiGLU}(\mathbf{G},\mathbf{U})
      =\operatorname{Swish}_1(\mathbf{G}_c)\odot\mathbf{U}_c
      =\operatorname{SiLU}(\mathbf{G}_c)\odot\mathbf{U}_c.
      $$
    <!-- npu="950" id25 -->
    - `swiglustep`:
      $$
      \mathbf{A}=\operatorname{SwiGLUStep}(\mathbf{G},\mathbf{U})=\min\!\left(\operatorname{SiLU}(\mathbf{G}),c\right)\odot\mathbf{U}_c.
      $$
    - `swigluoai`:
      $$
      \mathbf{A}=\operatorname{SwiGLUOAI}(\mathbf{G},\mathbf{U})=\mathbf{G}_c\odot\sigma(\alpha\mathbf{G}_c)\odot(\mathbf{U}_c+\beta).
      $$
    - `situglu`:
      $$
      \mathbf{A}=\operatorname{SiTUGLU}(\mathbf{G},\mathbf{U})=\beta\tanh\!\left(\frac{\mathbf{G}}{\beta}\right)\odot\sigma(\mathbf{G})\odot L(\mathbf{U};\beta_{\mathrm{linear}}),
      $$
      where
      $$
      L(\mathbf{U};\beta_{\mathrm{linear}})=
      \begin{cases}
      \mathbf{U}, & \text{ not configured linear\_beta},\\
      \beta_{\mathrm{linear}}\tanh\!\left(\dfrac{\mathbf{U}}{\beta_{\mathrm{linear}}}\right),
      & \text{ configured linear\_beta}.
      \end{cases}
      $$
    <!-- end id25 -->

- Computation notes:

    The data types of the activation matrix A and weight matrix W for **Linear computation** supported by each product are as follows:

    | Scenario Name    |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN,FLOAT8_E5M2 |FLOAT8_E4M3FN,FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: support the A16W16, A8W8-INT, and A8W4-INT use cases.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: supports the A8W8-FP, A8W4-FP, and A4W4-FP use cases.
    <!-- end id8 -->

    <details>
    <summary> A16W16 non-quantization use case</summary>

    - **EP Dispatch**

        In the Dispatch phase, each $\text{token}_i$ sends its token representation $\mathbf{x}_i$ to experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$. That is, for each $k$, expert $e_{i,k}$ receives a copy of $\mathbf{x}_i$.

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the index set of all tokens dispatched to expert $e$. The size of set $I_e$, $N_e = |I_e|$, is the total number of tokens that expert $e$ needs to process. Then $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ is the matrix formed by stacking the token representations $\mathbf{x}_i$ of all tokens that satisfy $i \in I_e$ **by rows in any fixed order**. This matrix is the complete set of token representations received by expert $e$ after Dispatch.

        For each $\text{token}_i$ and its selected $k$-th expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$. This mapping records the position of $\mathbf{x}_i$ in the input matrix $\mathbf{X}_{e_{i,k}}$ of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts a gated activation structure specified by `activation` to enhance expressive capacity. The entire computation consists of the following three substeps.

        **1. Linear1 Projection**

        Linear1 projection is the first linear transformation of the expert network. It simultaneously produces the pre-activation values required for **the gate part** and **the up part**. The formula is:
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}\bigr)
        $$
        **2. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, then compute the intermediate activation representation $\mathbf{A}_e$ based on `activation`. For the computation method of each available activation type, see [Activation Function Formulas](#activation-formulas).

        **3. Linear2 Projection**

        Linear2 projection is the second linear transformation. It projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, making the expert output compatible with subsequent operations such as residual connections.
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        After the preceding computation, each output row of expert $e$ corresponds to one token in its batch. For $\text{token}_i$, its output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs a weighted sum based on the expert weights originally assigned to each token, and finally generates a fused output for each token. Using the position index $\operatorname{row}(i,k)$ recorded earlier, retrieve the row belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiply it by the gating weight, and sum the results:

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in input order to form the final output $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$.
    </details>

    <details>
    <summary> A8W8-INT quantization use case</summary>

    - **Input**

      - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$: per-channel scale factors of the Linear1 weight matrix, corresponding to the input parameter `l1_weights_sf`.
      - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$: per-channel scale factors of the Linear2 weight matrix, corresponding to the input parameter `l2_weights_sf`.

    - **EP Dispatch**

        Before the Dispatch communication, first quantize the original BF16 activation matrix $\mathbf{X}$ to int8. For each $\text{token}_i$, compute its per-token scale factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        Then quantize it to obtain the int8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        During the Dispatch communication phase, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and scale factor $s^{X}_i$ to experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$ ($e_{i,k} = \mathbf{E}[i,k]$).

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the index set of all tokens dispatched to expert $e$. The size of the set $I_e$, $N_e = |I_e|$, is the total number of tokens that expert $e$ needs to process. Then $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is a matrix formed by **row-stacking in any fixed order** all $\mathbf{q}_i$ with $i \in I_e$. This matrix is the complete token representation received by expert $e$ after Dispatch. Similarly, the scaling factor vector received by expert $e$ is denoted as $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are obtained by stacking all $s^{X}_i$ with $i \in I_e$ in the same row order as $\mathbf{Q}_e$.

        For each $\text{token}_i$ and its selected $k$-th expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In a MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts the gated activation structure specified by `activation` to improve expressive capacity. In the A8W8 use case, both linear layers perform matrix multiplication with int8 inputs and int8 weights to obtain int32 intermediate results, which are then dequantized. The process consists of the following three substeps.

        **1. Linear1 projection (int8 matrix multiplication + dequantization)**

        Linear1 projection is the first linear transformation of the expert network. It produces the pre-activation values required by both the **gate part** and the **up part**. During computation, int8 matrix multiplication is performed to obtain the int32 result:
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$
        The result is then dequantized into the pre-activation values $\mathbf{H}_e$:
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        **2. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, and compute the intermediate activation representation $\mathbf{A}_e$ according to `activation`. For the computation method of each optional activation type, see [Activation Function Formulas](#activation-formulas).

        **3. Linear2 projection (quantization + int8 matrix multiplication + dequantization)**

        Linear2 projection is the second linear transformation. It projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, making the expert output compatible with subsequent operations such as residual connections.
        In the A8W8 use case, activation values need to be quantized to int8. Therefore, first compute a scaling factor for each row (each token) of $\mathbf{A}_e$:
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        This yields the activation scaling factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ of expert $e$ in the Linear2 computation. Then quantize:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$
        Then perform int8 matrix multiplication and dequantize:
        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        After the preceding calculations, each output row of expert $e$ corresponds to a token in its batch. For $\text{token}_i$, its output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs a weighted sum based on the expert weights originally assigned to each token, and finally generates a fused output for each token. Using the previously recorded position index $\operatorname{row}(i,k)$, it retrieves the rows belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiplies them by the gating weights, and sums the results:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in input order to form the final output $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$.

    </details>

    <details>
    <summary> A8W4-INT quantization use case</summary>

    - **Inputs**
      - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$: Per-channel scaling factors of the Linear1 weight matrix, corresponding to input parameter `l1_weights_sf`.
      - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$: Per-channel scaling factors of the Linear2 weight matrix, corresponding to input parameter `l2_weights_sf`.
      - $\mathbf{B}_1 \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$: Bias matrix of Linear1, generated offline during the int4 quantization process, corresponding to input parameter `l1_bias`.
      - $\mathbf{B}_2 \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$: Linear2's bias matrix, generated offline during the int4 quantization process. Corresponding input parameter `l2_bias`.

    - **EP Dispatch**

        Before the Dispatch communication, the original BF16 activation matrix $\mathbf{X}$ is first quantized to int8. For each $\text{token}_i$, compute its per-token scale factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        Then quantize it to obtain the int8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        During the Dispatch communication phase, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and the scale factor $s^{X}_i$ to the experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the token index set of all tokens dispatched to expert $e$. The size of set $I_e$, $N_e = |I_e|$, is the total number of tokens to be processed by expert $e$. Then, $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is a matrix **stacked row-wise in any fixed order** from all $\mathbf{q}_i$ with $i \in I_e$. This matrix is the complete set of token representations received by expert $e$ after Dispatch; similarly, the scale factor vector received by the corresponding expert $e$ is denoted as $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are stacked, in the same row order as $\mathbf{Q}_e$, from all $s^{X}_i$ with $i \in I_e$.

        For each $\text{token}_i$ and its selected $k$-th expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In an MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts a gated activation structure specified by `activation` to enhance expressive capacity. In the A8W4-INT use case, both linear layers use the MSD (Mixed-precision Split-activation Decomposition) scheme for matrix multiplication. By splitting an int8 value into the high 4 bits and the low 4 bits as two signed int4 values, an int8×int4 matrix multiplication can be decomposed into two int4×int4 matrix multiplications, leveraging the hardware int4 matrix multiplication acceleration. For the mathematical principle and implementation logic of this scheme, see [GroupedMatmul W4A8 Quantization and MSD Scheme](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81).

        **1. Generate the precision-compensation bias matrix (generated offline outside the operator and used as an operator input)**

        The core of the MSD scheme is to split the quantized int8 activation values into two int4 components through binary reinterpretation. The low 4-bit int4 component is defined as $(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$, so its value range is mapped to $[-8, 7]$, which is equivalent to subtracting 8 from the original unsigned low 4 bits (0-15). When the high and low int4 components are multiplied with the weight matrix separately and then merged, this offset introduces a constant term that must be compensated for in the final result.

        Denote the original int8 activation matrix as $\mathbf{X}^{\text{int8}}$, the high-bit and low-bit int4 activation matrices after splitting as $\mathbf{X}_1^{\text{int4}}$ and $\mathbf{X}_2^{\text{int4}}$ respectively, and the weight matrix as $\mathbf{W}$. The recovery relationship is:

        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        where $\mathbf{1}_{\text{mat}}$ is an all-ones matrix of the same shape as $\mathbf{X}_2^{\text{int4}}$, used for element-wise addition of 8. Let $\mathbf{1}$ be an all-ones column vector whose shape matches the input feature dimension. The matrix multiplication expands to:

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        Here, $\mathbf{1}^\top \cdot \mathbf{W}$ denotes the sum over each column of the weight matrix $\mathbf{W}$, and the result is a row vector (with the same shape as the output dimension). In the actual implementation, this row vector is broadcast to the batch dimension and added to the output of all tokens. As can be seen, computing only the first two terms misses a positive term. To accurately recover the result, the compensation bias must be precomputed:

        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        $\mathbf{B}_1$ and $\mathbf{B}_2$ are computed in the same way: the corresponding all-ones column vectors (with dimensions $\text{hidden}$ and $\text{intermediate\_hidden}$ respectively) are each multiplied by their own weight matrices, yielding bias row vectors of matching shapes. These biases are computed offline and then passed to the operator. During the merge of high-bit and low-bit results, they are directly added, eliminating the error introduced by the offset.

        **2. Linear1 projection (int4 × int4 matrix multiplication + dequantization)**

        The Linear1 projection is the first linear transformation of the expert network. It simultaneously produces the pre-activation values required by the **gate part** and the **up part**.

        **(1) Reinterpreting the activation as int4**

        Binary-reinterpret the int8 activation tensor $\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ as a view of two interleaved int4 components:

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        The high and low components are directly given by the even and odd rows of $\mathbf{Q}_e^{\text{int4}}$:

        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        Numerically, they can be obtained from the original int8 values as follows:

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        The recovery relationship is $\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the type view, $\mathbf{Q}_e^{\text{int4}}$ and $\mathbf{Q}_e^{\text{int8}}$ share the underlying physical memory, without any data rearrangement or copying.

        **(2) int4 × int4 matrix multiplication and weight dequantization**

        Perform matrix multiplication of the reinterpreted int4 activation view $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ with the weight matrix $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediate\_hidden}}$, and apply the weight scale factor $\mathbf{s}^{W1}_e$ for dequantization to obtain the result:

        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        Record the result as the following matrix views for even rows and odd rows, respectively. They correspond to the computation results of $\mathbf{Q}_e^{\text{high}}$ and $\mathbf{Q}_e^{\text{low}}$:

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        **(3) Precision compensation and activation dequantization**

        Perform precision compensation and activation dequantization using the bias $\mathbf{B}_1[e]$ and the activation scaling factor $\mathbf{s}^{X}_e$ (a row vector), respectively. The final pre-activation value is:

        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
        $$

        where $\odot \mathbf{s}^{X}_e$ indicates that each row of the matrix is multiplied by the corresponding scalar in $\mathbf{s}^{X}_e$.

        **3. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, and compute the intermediate activation representation $\mathbf{A}_e$ based on `activation`. For the computation method of each optional activation type, see [activation function formulas](#activation-formulas).

        **4. Linear2 projection (quantization + int4 × int4 matrix multiplication + dequantization)**

        The Linear2 projection, as the second linear transformation, projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, so that the expert output is compatible with subsequent operations such as residual connection.
        In the A8W4 use case, the activation values are first quantized to int8 and then reinterpreted in binary as a concatenated view of two int4 values, completing the computation with a single matrix multiplication. The process is as follows.

        **(1) Activation quantization**

        For each row of $\mathbf{A}_e$, compute the scale factor and quantize to int8:
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        This yields the activation scale factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ of expert $e$ in the Linear2 computation. Compute the quantization result:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
        $$

        **(2) Activation reinterpretation as int4**

        Binary-reinterpret the int8 activation tensor $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}$ as a view of two alternately concatenated int4 components:

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}
        $$

        The high-bit and low-bit components are directly given by the even rows and odd rows of $\mathbf{A}_e^{\text{int4}}$, respectively:

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediate\_hidden}}
        $$

        They can be numerically obtained from the original int8 as follows:

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        The recovery relation is $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the view type, $\mathbf{A}_e^{\text{int4}}$ and $\mathbf{A}_e^{\text{int8}}$ share the underlying physical memory, requiring no data rearrangement or copy.

        **(3) int4 × int4 matrix multiplication and weight dequantization**

        Perform matrix multiplication between the reinterpreted int4 activation view $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediate\_hidden}}$ and the weight matrix $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediate\_hidden} \times \text{hidden}}$, and apply the weight scaling factor $\mathbf{s}^{W2}_e$ for dequantization to obtain the result:

        $$
        \mathbf{D}_e = \bigl( \mathbf{A}_e^{\text{int4}} \cdot \mathbf{W}_2[e] \bigr) \odot \mathbf{s}^{W2}_e
        \;\in\; \mathbb{R}^{2N_e \times \text{hidden}}
        $$

        Denote the results of the even rows and the odd rows as the following matrix views, which correspond to the computation results of $\mathbf{A}_e^{\text{high}}$ and $\mathbf{A}_e^{\text{low}}$, respectively:

        $$
        \mathbf{D}_e^{\text{high}} = \mathbf{D}_e[0::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}, \qquad
        \mathbf{D}_e^{\text{low}}  = \mathbf{D}_e[1::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        **(4) Precision compensation and activation dequantization**

        Perform precision compensation and activation dequantization with the bias $\mathbf{B}_2[e]$ and the activation scaling factor $\mathbf{s}^{A_e}_e$ (row vector), respectively. The final output is:

        $$
        \mathbf{Y}_e = \Bigl( 16 \cdot \mathbf{D}_e^{\text{high}} + \mathbf{D}_e^{\text{low}} + \mathbf{B}_2[e] \Bigr) \odot \mathbf{s}^{A_e}_e
        \quad\in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        where $\odot \mathbf{s}^{A_e}_e$ denotes multiplying each row of the matrix by the corresponding scalar in $\mathbf{s}^{A_e}_e$.

        After the above computation, each output row of expert $e$ corresponds to a token in its batch. For $\text{token}_i$, the output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs weighted summation based on the expert weight originally assigned to each token, and finally generates a fused output for each token. Using the previously recorded position index $\operatorname{row}(i,k)$, it retrieves the rows belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiplies them by the gating weights, and sums them:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in the input order as the final output $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$.

    </details>

    <!-- npu="950" id23 -->
    <details>
    <summary> A8W8-FP quantization use case</summary>

    In the first phase, input tokens are grouped and collected by expert and then quantized using MXFP8, generating the quantized input and scale factor of each expert:

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    Note: Tokens are sorted and collected by expert according to `topk_ids`. $\mathcal{T}_e$ is the index set of tokens assigned to expert $e$, and $E_{\text{local}}$ denotes the number of routed MoE experts on this Rank. $X[\mathcal{T}_e]$ is the input submatrix corresponding to expert $e$, whose number of rows depends on the number of tokens actually received by that expert. $\mathrm{Q}_{\text{MX}}$ denotes MX per-group quantization (group size = 32), which extracts a shared exponent from each group of 32 elements, quantizes to the FP8 target type (FLOAT8_E5M2 or FLOAT8_E4M3FN), and outputs a FLOAT8_E8M0 scale factor. The quantized data serves as the input of GMM1.

    In the second phase, GMM1 matrix multiplication (with $W_1$ split into two halves along the column dimension and computed separately), Activation, and MX quantization are performed for each expert:

    $$
    G_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(G)}, S_{1,e}^{(G)}), \quad U_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(U)}, S_{1,e}^{(U)})
    $$

    $$
    A_e = \operatorname{Activation}(G_e,U_e)
    $$

    $$
    \hat{A}_e,\ S_{A,e} = \mathrm{Q}_{\text{MX}}(A_e)
    $$

    Note: Perform matrix multiplication between the first $N/2$ columns $W_{1,e}^{(G)}$ and the last $N/2$ columns $W_{1,e}^{(U)}$ of $W_1$ and the MX-dequantized input, respectively, to obtain the gate branch $G_e$ and the up branch $U_e$. For the computation method of the activation function, see [Activation Function Formulas](#activation-formulas). The output dimension is $N/2$. Then perform MX quantization on the activation output. The quantization type is the same as that in the Dispatch phase, that is, both are `FLOAT8_E5M2` or both are `FLOAT8_E4M3FN`, to obtain the quantized input of GMM2 $\hat{A}_e$.

    The third phase performs GMM2 matrix multiplication for each expert and dispatches the results to the target Rank:

    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{A}_e, S_{A,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    Note: Perform matrix multiplication between the quantized activation output and the second group of weights $W_2$ after MX dequantization, mapping the $N/2$-dimensional intermediate representation back to the $H$-dimensional hidden space to obtain the output $O_e$ of each expert. After the computation, write the results to the remote end via RDMA peermem at the expert offset addresses of the target Rank, achieving cross-Rank aggregation.

    The fourth phase performs a weighted sum over all tokens based on the routing weights, adds the shared expert outputs, and restores the output to the same shape as the input:

    When shared experts are enabled (`shared_expert_num_per_rank` > 0), each shared expert locally performs the same GMM1 + Activation + GMM2 computation as the routed experts on all tokens of the local device, using `shared_l1_weights`, `shared_l2_weights`, `shared_l1_weights_sf`, and `shared_l2_weights_sf`, without participating in the Dispatch communication. The output of each shared expert is denoted as $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$.

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i]
    $$

    Note: For each token $i$, retrieve the corresponding row from the aggregated expert results based on the sorted routing index $\pi(i,k)$, perform element-wise weighted accumulation using the weights in `topk_weights`, and then directly add the outputs of each shared expert for the current token to obtain the final output $Y$. If shared experts are not enabled, the shared expert summation term is zero.

    where $X$ denotes parameter `x`, $W$ denotes parameter `topk_weights`, $W_1$ denotes parameter `l1_weights`, $W_2$ denotes parameter `l2_weights`, $Y$ denotes parameter `y`, $E_{\text{local}}$ denotes `local_moe_expert_num = num_experts / ep_world_size` (the number of routed MoE experts per Rank), and $K$ denotes the second dimension of `topk_ids`.

    The local variables are described as follows:
    - $\mathcal{T}_e$: index set of tokens routed to expert $e$, determined by sorting `topk_ids`.
    - $m_e=|\mathcal{T}_e|$: number of tokens actually received by expert $e$ in this call; $m_e$ may vary across experts.
    - $\hat{X}_e,\ S_{X,e}$: quantized input of expert $e$ and its MX scale factor, an intermediate result of the first phase.
    - $W_{1,e}^{(G)}$, $W_{1,e}^{(U)}$: sub-matrices of $W_1$ for expert $e$, corresponding to the first $N/2$ columns and the last $N/2$ columns, derived by splitting `l1_weights` into the gate branch and up branch.
    - $S_{1,e}^{(G)}$, $S_{1,e}^{(U)}$: MX scale factors of $W_{1,e}^{(G)}$ and $W_{1,e}^{(U)}$, truncated from `l1_weights_sf` along the dimension.
    - $S_{2,e}$: MX scale factor of $W_{2,e}$, from parameter `l2_weights_sf`.
    - $G_e,\ U_e$: outputs of the gate branch and up branch of GMM1, intermediate results.
    - $A_e$: activation output, with dimension $m_e \times N/2$, an intermediate result.
    - $\hat{A}_e,\ S_{A,e}$: quantized activation output and its MX scale factors, intermediate result.
    - $O_e$: expert-level output of GMM2, dimension $m_e \times H$, intermediate result.
    - $\pi(i, k)$: row index of the $k$-th top-k expert of Token $i$ after the expansion sort, determined by the routing sort.
    - $\mathrm{Q}_{\text{MX}}(\cdot)$: MX per-group quantization operation with block size = 32, outputting FP8 data and E8M0 scale factors.
    - $\mathrm{DQ}_{\text{MX}}(\cdot)$: MX per-group dequantization operation, implicitly performed inside matmul.
    </details>

    <details>
    <summary> A8W4-FP quantization use case</summary>

    Phase 1 (Token selection, quantization, and Dispatch):

    For the input Token $X \in \mathbb{R}^{B\times H}$ of this rank, obtain the Token index set $T_e$ corresponding to each expert $e$ based on `topk_ids`, and quantize the selected BF16 Tokens to MXFP8 E4M3 in groups of 32 elements:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    Here, the data type of $\hat{X}_e$ is `FLOAT8_E4M3FN`, and the data type of $S_{X,e}$ is `FLOAT8_E8M0`. The quantized tokens and their scale factors are then sent to the rank where the expert resides.

    The first- and second-layer weights of the routed MoE experts, $W_{1,e}$ and $W_{2,e}$, are both MXFP4 E2M1 data with E8M0 scale factors, where $e$ ranges over $[0,\text{local\_moe\_expert\_num})$. The shared expert weights are separately provided by `shared_l1_weights` and `shared_l2_weights`, whose first dimension is `shared_expert_num_per_rank`. The A8W4 kernel processes FP4 weights in the matrix multiplication Prologue. The logical data flow fed into the matrix multiplication is FP8 activations multiplied by FP4 weights.

    Phase 2 (GMM1, Activation, and requantization):

    For the tokens received by expert $e$, the first-layer grouped matrix multiplication and activation function are computed as:

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{Activation}(G_e,U_e).
    $$

    For the specific computation of the activation function, see [activation function formulas](#activation-formulas). The output is further quantized into MXFP8 E4M3 in groups of 32 elements, with the same quantization type as the Dispatch phase, both `FLOAT8_E4M3FN`, for use in the second-layer matrix multiplication:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    Phase 3 (GMM2):

    The second-layer matrix multiplication is still A8W4, that is, FP8 E4M3 activations multiplied by FP4 E2M1 weights:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When shared experts are enabled (`shared_expert_num_per_rank` > 0), each shared expert locally performs the same GMM1 + Activation + GMM2 computation as the routed experts on all tokens of the current device, using `shared_l1_weights`, `shared_l2_weights`, `shared_l1_weights_sf`, and `shared_l2_weights_sf`, without participating in the Dispatch communication. The output of each shared expert is denoted $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$.

    Stage 4 (Combine and weighted merging):

    Send the outputs of all experts back to their original ranks, perform weighted merging based on `topk_weights`, and add the shared expert outputs:

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert summation term is zero. The final output $Y$ is of data type BF16. The main data type flow of A8W4-FP is: `BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`.
    </details>

    <details>
    <summary> A4W4-FP quantization use case</summary>

    Stage 1 (Token selection, quantization, and Dispatch):

    For the token set $T_e$ corresponding to each expert $e$, quantize the selected BF16 tokens into MXFP4 E2M1 in groups of 32 elements:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    where $\hat{X}_e$ has the data type `FLOAT4_E2M1` and $S_{X,e}$ has the data type `FLOAT8_E8M0`. The first-layer and second-layer weights of the routed MoE experts, $W_{1,e}$ and $W_{2,e}$, are both MXFP4 E2M1, and the range of $e$ is $[0,\text{local\_moe\_expert\_num})$. The first dimension of the shared expert weights is `shared_expert_num_per_rank`. The weight scaling factor is E8M0.

    Stage 2 (A4W4 GMM1, Activation, and post-activation quantization):

    The first-layer grouped matrix multiplication is A4W4:

    $$
    G_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(G)},S_{1,e}^{(G)}\right),
    $$

    $$
    U_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(U)},S_{1,e}^{(U)}\right),
    $$

    $$
    A_e = \operatorname{Activation}(G_e,U_e).
    $$

    For the computation method of each activation function, see [activation formulas](#activation-formulas). In the A4W4-FP use case, the activation output after the first-layer matrix multiplication is quantized into MXFP8 E4M3 in groups of 32 contiguous elements, with E8M0 scaling factors, and serves as the input to the second-layer matrix multiplication. This design avoids the large precision loss that would result if both matrix multiplications used A4W4:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    Stage 3 (A8W4 GMM2):

    Since the quantized activation output is FP8 E4M3 while the second-layer weights remain FP4 E2M1, the second-layer matrix multiplication is actually A8W4 rather than A4W4:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When shared experts are enabled (`shared_expert_num_per_rank` > 0), each shared expert locally performs the same GMM1 + Activation + GMM2 computation as the routed experts on all tokens of the device, using `shared_l1_weights`, `shared_l2_weights`, `shared_l1_weights_sf`, `shared_l2_weights_sf`, without participating in Dispatch communication. The output of each shared expert is denoted as $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{shared\_expert\_num\_per\_rank} - 1\}$.

    Phase 4 (Combine and weighted merging):

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{shared\_expert\_num\_per\_rank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert sum term is zero. The final output $Y$ is of the BF16 data type. The complete data type flow of A4W4-FP is: `BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`. All MX scaling factors are of the type `FLOAT8_E8M0`, and the quantization granularity is 32 consecutive elements.
    </details>
    <!-- end id23 -->

## Function Prototype

Call the get_symm_buffer_for_mega_moe API first to encapsulate input parameters and create the SymmBuffer structure, and then call the mega_moe API to perform the computation.

```python
get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, *, max_recv_token_num=0, dispatch_quant_mode=0, dispatch_quant_out_dtype=None, combine_quant_mode=0, comm_alg="", topk_weights_type=0) -> SymmBuffer
```

```python
mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, activation_params=None, weight1_type=None, weight2_type=None, shared_l1_weights=None, shared_l2_weights=None, shared_l1_weights_sf=None, shared_l2_weights_sf=None, shared_l1_bias=None, shared_l2_bias=None) -> (Tensor, Tensor)
```

### Elastic Scaling APIs

```python
sym_buffer.query_mask_buffer(mask_status) -> None
sym_buffer.update_mask_buffer(rank, masked) -> None
sym_buffer.clean_mask_buffer() -> None
sym_buffer.get_local_buffer_tensor(dtype, size=None, offset=0) -> Tensor
sym_buffer.update_group(group) -> None
```

By default, `SymmBuffer` does not create `mask_buffer`. Therefore, in all product generations, this parameter is not passed to the operator by default. The all-0 int32 mask with shape `[ep_world_size]` is created on the NPU only upon the first call to `query_mask_buffer`, `update_mask_buffer`, or `clean_mask_buffer`. 0 indicates a normal rank, and 1 indicates a disabled rank. Disabled ranks are skipped in subsequent computation and inter-device communication.

- `query_mask_buffer` receives `mask_status` pre-created by the caller. Its dtype must be `torch.int32`, and its shape and NPU must be the same as those of the internal `mask_buffer`. The API writes the current mask to `mask_status` through a D2D copy and does not return a tensor.
- `update_mask_buffer` updates only the local mask in the current process. The caller must ensure mask consistency across ranks. When a rank is restored (that is, the `masked` input is `False`), you must call `get_local_buffer_tensor` to clear the existing communication buffer flags of the operator. The sample code is as follows:

  ```python
  symm_buffer.update_mask_buffer(0, False)  # Restore communication for rank 0.
  symm_buffer.update_mask_buffer(1, False)  # Restore communication for rank 1.
  local_buffer = symm_buffer.get_local_buffer_tensor(torch.uint8)
  local_buffer.zero_()
  ```

- `clean_mask_buffer` sets all elements of the local mask to zero on the current NPU stream. Each rank must call this API separately, and the caller must still ensure consistency across ranks. If the mask has not been created, the API creates an all-0 mask.
- `get_local_buffer_tensor` wraps the local CCL buffer of the current rank into an NPU tensor through zero copy. `offset` is in units of `dtype` elements. If `size` is `None`, a 1D view from `offset` to the end of the buffer is returned; otherwise, a tensor of the specified shape is returned. The returned tensor does not own the underlying memory, and its lifecycle must not exceed that of `SymmBuffer`. After `update_group` is called, old views become invalid and must be re-obtained. Before writing to the original communication buffer, the caller must ensure that the related operators have completed execution and that the write range is correct.
- `update_group` fully rebuilds the communication link using the new group. The caller must ensure that operators on the old communication link complete execution and that the new group is consistent with the existing mask shape. To clear the disabled status, explicitly call `clean_mask_buffer`.

<!-- npu="A3" id26 -->
- <term>Atlas A3 products</term>: `mask_buffer` and the five `SymmBuffer` elastic scaling APIs above are supported.
<!-- end id26 -->
<!-- npu="950,910b" id27 -->
- <term>Ascend 950PR&950DT products</term>, <term>Atlas A2 products</term>: `mask_buffer` and the five `SymmBuffer` elastic scaling APIs above are not supported and must not be called.
<!-- end id27 -->

## Mandatory

### get_symm_buffer_for_mega_moe

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>str</td>
        <td>Required</td>
        <td>EP communicator name (expert parallel communicator).</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>Required</td>
        <td>Total number of experts in the MoE model.</td>
    </tr>
    <tr>
        <td>num_max_tokens_per_rank</td>
        <td>int</td>
        <td>Required</td>
        <td>Maximum number of tokens per device that may occur on each Rank in the communicator. In each call, x.shape[0] must not exceed this value. All Ranks must use the same upper bound. For whether the actual token count may differ across Ranks, see Constraints.</td>
    </tr>
    <tr>
        <td>num_topk</td>
        <td>int</td>
        <td>Required</td>
        <td>Number of experts sent per token.</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>Required</td>
        <td>Size of each token.</td>
    </tr>
    <tr>
        <td>intermediate_hidden</td>
        <td>int</td>
        <td>Required</td>
        <td>Intermediate feature dimension after activation. Linear1 generates both the gate and up branches, so the full output width of Linear1 is 2 × intermediate_hidden. Currently not used and not involved in communication buffer size calculation.</td>
    </tr>
    <tr>
        <td>max_recv_token_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Maximum number of tokens each rank can receive. The default value 0 indicates automatic calculation. Default value: 0.</td>
    </tr>
    <tr>
        <td>dispatch_quant_mode</td>
        <td>int</td>
        <td>Optional</td>
        <td>Quantization mode for dispatch communication. 0 indicates non-quantization (A16W16 use case), 2 indicates int8 quantization (A8W8-INT, A8W4-INT use cases), and 4 indicates MXFP quantization (A8W8-FP, A8W4-FP, A4W4-FP use cases). For values supported by each product, see Constraints. Default value: 0.</td>
    </tr>
    <tr>
        <td>dispatch_quant_out_dtype</td>
        <td>torch.dtype</td>
        <td>Optional</td>
        <td>Data type of the output after dispatch quantization. Supported values: torch.int8, torch.float8_e5m2, torch.float8_e4m3fn, and torch.float4_e2m1. For values supported by each product, see Constraints. Default value: None.</td>
    </tr>
    <tr>
        <td>combine_quant_mode</td>
        <td>int</td>
        <td>Optional</td>
        <td>Quantization mode for the combine communication. 0 indicates no quantization, 3 indicates the MXFP float8_e5m2 type, and 4 indicates the MXFP float8_e4m3 type. For values supported by each product, see Constraints. Default value: 0.</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>Optional</td>
        <td>This parameter is not supported currently. Use the default value. Default value: "".</td>
    </tr>
    <tr>
        <td>topk_weights_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>topkWeights pre-forward switch. 0: disabled; 1: enabled (sends topkWeights together with token data to the target rank in advance during the dispatch phase, reducing communication traffic in the combine phase). The default value is 0.</td>
    </tr>
</tbody>
</table>

<!-- npu="950" id28 -->
- <term>Ascend 950PR&950DT products</term>: the actual number of tokens may differ across Ranks; `num_max_tokens_per_rank` only sets the common upper bound.
<!-- end id28 -->

### mega_moe

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Descriptions with superscript <sup>1</sup> in the parameter table are not supported. Use default values for the unsupported parameters.
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: Descriptions with superscript <sup>2</sup> in the parameter table are not supported. Use default values for the unsupported parameters.
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
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Hidden states of tokens input to the MoE layer.</td>
        <td>bfloat16</td>
        <td>ND</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>topk_ids</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Expert index matrix, indicating the num_topk experts selected by each token. Element values range from <code>[0, num_experts)</code>, and the num_topk experts selected for the same token must be unique.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Gating weight coefficients corresponding to the num_topk experts selected by the expert gating network of the MoE model for the current input token.</td>
        <td>bfloat16, float32</td>
        <td>ND</td>
        <td>(num_tokens, num_topk)</td>
    </tr>
    <tr>
        <td rowspan="6">l1_weights</td>
        <td rowspan="6">list[Tensor]</td>
        <td rowspan="6">Required</td>
        <td rowspan="6">Weight matrix of the first linear layer of the MoE expert network (including the gating and up-projection). It maps the input to the intermediate dimension and feeds the output to the activation function. The number of MoE experts per device is <code>local_moe_expert_num</code>.</td>
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
        <td>(local_moe_expert_num, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>float8_e4m3fn<sup>1</sup></td>
        <td>ND/FRACTAL_NZ</td>
        <td>(local_moe_expert_num, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>float4_E2M1(uint8)<sup>1</sup></td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(local_moe_expert_num, 2 × intermediate_hidden, hidden / 2)</td>
    </tr>
    <tr>
        <td rowspan="6">l2_weights</td>
        <td rowspan="6">list[Tensor]</td>
        <td rowspan="6">Required</td>
        <td rowspan="6">Weight matrix of the second linear layer of the MoE expert network, projecting the activated intermediate features back to the hidden dimension. The data type is the same as that of l1_weights. The number of MoE experts per device is <code>local_moe_expert_num</code>.</td>
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
        <td>(local_moe_expert_num, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>float8_e4m3fn<sup>1</sup></td>
        <td>ND/FRACTAL_NZ</td>
        <td>(local_moe_expert_num, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>float4_E2M1(uint8)<sup>1</sup></td>
        <td>FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(local_moe_expert_num, hidden, intermediate_hidden / 2)</td>
    </tr>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>Yes</td>
        <td>A structure created by the <a href="#get_symm_buffer_for_mega_moe">get_symm_buffer_for_mega_moe</a> API</td>
        <td>SymmBuffer</td>
        <td>SymmBuffer</td>
        <td>SymmBuffer</td>
    </tr>
    <tr>
        <td rowspan="2">l1_weights_sf</td>
        <td rowspan="2">list[Tensor]</td>
        <td rowspan="2">Optional</td>
        <td rowspan="2">Quantization scale factor of the weight matrix in the first linear layer of the MoE expert network. Required in the MXFP quantization use case.</td>
        <td>uint64<sup>2</sup></td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>float8_e8m0<sup>1</sup></td>
        <td>ND</td>
        <td>(local_moe_expert_num, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)</td>
    </tr>
    <tr>
        <td rowspan="2">l2_weights_sf</td>
        <td rowspan="2">list[Tensor]</td>
        <td rowspan="2">Optional</td>
        <td rowspan="2">Quantization scale factor of the weight matrix of the second linear layer of the MoE expert network. Required for MXFP quantization use cases.</td>
        <td>uint64<sup>2</sup></td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>float8_e8m0<sup>1</sup></td>
        <td>ND</td>
        <td>(local_moe_expert_num, hidden, CeilDiv(intermediate_hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>l1_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Bias of the first linear layer of the MoE expert network. This parameter is required only in the A8W4-INT quantization use case for precision compensation.</td>
        <td>float32</td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>l2_bias<sup>2</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Bias of the second linear layer in the MoE expert network. Required only in the A8W4-INT quantization use case for precision compensation.</td>
        <td>float32</td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
    <tr>
        <td>x_active_mask<sup>2</sup></td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Indicates whether the token participates in communication.</td>
        <td>int8</td>
        <td>ND</td>
        <td>(num_tokens, )</td>
    </tr>
    <tr>
        <td>activation</td>
        <td>str</td>
        <td>Optional</td>
        <td>Activation function type. The default value is "swiglu". Optional values are "swiglu", "swiglustep", "swigluoai", and "situglu".</td>
        <td>str</td>
        <td>N/A</td>
        <td>N/A</td>
    </tr>
    <tr>
        <td>activation_clamp</td>
        <td>float</td>
        <td>Optional</td>
        <td>Clamp value for "swiglu", "swiglustep", and "swigluoai". Defaults to the maximum float value if not configured. "situglu" ignores this value. The value must be ≥ 0 and cannot be NaN.</td>
        <td>float</td>
        <td>N/A</td>
        <td>N/A</td>
    </tr>
    <tr>
        <td>activation_params</td>
        <td>dict[str, float]</td>
        <td>Optional</td>
        <td>Dictionary of activation function parameters, with a default value of None. No configuration is required for "swiglu" and "swiglustep". "swigluoai" supports "alpha" and "beta", with default values of 1.702 and 1.0 respectively. "situglu" supports "beta" and "linear_beta". The default value of "beta" is 1.0. If "linear_beta" is not configured, the up branch remains unchanged. On each product, the "beta" value actually used and the configured "linear_beta" value must both be finite values greater than 0.</td>
        <td>float</td>
        <td>N/A</td>
        <td>N/A</td>
    </tr>
    <!-- npu="950" id22 -->
    <tr>
        <td>weight1_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Logical data type of weight1. Optional in the A8W8-FP use case; by default, it is inferred from the Tensor data type of l1_weights. In the A8W4-FP and A4W4-FP use cases, it must be set to the type enumeration corresponding to float4_e2m1fn_x2. The value must be the same as that of weight2_type.</td>
        <td>int</td>
        <td>N/A</td>
        <td>N/A</td>
    </tr>
    <tr>
        <td>weight2_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Logical data type of weight2. Optional in the A8W8-FP use case, with the default value deduced from the Tensor data type of l2_weights. In the A8W4-FP and A4W4-FP use cases, it must be set to the type enumeration corresponding to float4_e2m1fn_x2. The value must be the same as that of weight1_type.</td>
        <td>int</td>
        <td>N/A</td>
        <td>N/A</td>
    </tr>
    <!-- end id22 -->
    <tr>
        <td rowspan="3">shared_l1_weights<sup>1</sup></td>
        <td rowspan="3">list[Tensor]</td>
        <td rowspan="3">Optional</td>
        <td rowspan="3">Weight matrix of the first linear layer of the shared expert network (including gating and up projection). It maps the input to the intermediate dimension, and the output is fed to the activation function. Its logical data type must be the same as that of l1_weights, the first linear layer weight of MoE experts.</td>
        <td>FLOAT8_E5M2</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>ND/FRACTAL_NZ</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden)</td>
    </tr>
    <tr>
        <td>FLOAT4_E2M1(uint8)</td>
        <td>FRACTAL_NZ/FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, hidden / 2)</td>
    </tr>
    <tr>
        <td rowspan="3">shared_l2_weights<sup>1</sup></td>
        <td rowspan="3">list[Tensor]</td>
        <td rowspan="3">Optional</td>
        <td rowspan="3">Weight matrix of the second linear layer of the shared expert network. It projects the activated intermediate features back to the hidden dimension. Its data type must be the same as the logical data type of l2_weights, the weight of the second linear layer of MoE experts.</td>
        <td>FLOAT8_E5M2</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>ND/FRACTAL_NZ</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden)</td>
    </tr>
    <tr>
        <td>FLOAT4_E2M1(uint8)</td>
        <td>FORMAT_FRACTAL_NZ_C0_32</td>
        <td>(shared_expert_num_per_rank, hidden, intermediate_hidden / 2)</td>
    </tr>
    <tr>
        <td>shared_l1_weights_sf<sup>1</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Quantization scale factor of the weight matrix of the first linear layer in the shared expert network.</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>shared_l2_weights_sf<sup>1</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Quantization scale factor of the weight matrix of the second linear layer in the shared expert network.</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>(shared_expert_num_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2)</td>
    </tr>
    <tr>
        <td>shared_l1_bias<sup>1,2</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Bias of the first linear layer of the shared expert network, currently not supported.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(2 × intermediate_hidden, )</td>
    </tr>
    <tr>
        <td>shared_l2_bias<sup>1, 2</sup></td>
        <td>list[Tensor]</td>
        <td>Optional</td>
        <td>Bias of the second linear layer of the shared expert network, currently not supported.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(hidden, )</td>
    </tr>
</tbody>
<tfoot>
<tr>
    <td colspan="7">In the table, <code>CeilDiv(<var>x</var>, <var>y</var>) = ⌈<var>x</var> / <var>y</var>⌉ = ⌊(<var>x</var> + <var>y</var> - 1) / <var>y</var>⌋</code></td>
</tr>
<tr>
    <td colspan="7">In the table, T1(T2) indicates that a logical data type T1 is passed in on the PyTorch side with data type T2 as its physical carrier, and the shape in the table is the actual input shape of the T2 carrier. For example, int4(int32) means that each int32 packs 8 int4 elements; float4_E2M1(uint8) means that each uint8 packs 2 FLOAT4_E2M1 elements, and the logical data type of the weights must be specified as <code>float4_e2m1fn_x2</code> through <code>weight1_type</code> and <code>weight2_type</code>.</td>
</tr>
</tfoot>
</table>

## Return Values

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
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>Required</td>
        <td>Encapsulates input parameters and generates <code>context</code>, <code>ep_world_size</code>, and <code>ccl_buffer_size</code>.</td>
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
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Token data received by the local device, corresponding to Y in the formula. The data type is consistent with input <code>x</code>. Must be a 2D tensor with the ND data format. Non-contiguous tensors are supported.</td>
        <td>bfloat16</td>
        <td>(num_tokens, hidden)</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Number of tokens actually received by each expert of the local device. Must be a 1D tensor with the ND data format. Non-contiguous tensors are supported.</td>
        <td>int32</td>
        <td>(local_moe_expert_num,)</td>
    </tr>
</tbody>
</table>

## Constraints

- The constraints on the list[Tensor] length, whether transposition is required, and whether non-contiguous tensors are supported for each tensor parameter are as follows:

    <table style="undefined;table-layout: fixed; width:1000px"><colgroup>
    <col style="width: 160px">
    <col style="width: 320px">
    <col style="width: 160px">
    <col style="width: 220px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>list[Tensor] Length</th>
        <th>Whether Transposed</th>
        <th>Whether Non-contiguous Tensor Is Supported</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>N/A</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>topk_ids</td>
            <td>N/A</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>topk_weights</td>
            <td>N/A</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>l1_weights</td>
            <td>local_moe_expert_num in the per-expert layout and 1 in the stacked layout. For the supported layouts and the exact values, see the constraints below this table.</td>
            <td>No (bfloat16/int8/int4 use cases)/Yes (float8_e5m2/float8_e4m3fn/float4_E2M1 use cases)</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>l2_weights</td>
            <td>local_moe_expert_num in the per-expert layout and 1 in the stacked layout. For the supported layouts and the exact values, see the constraints below this table.</td>
            <td>No (bfloat16/int8/int4 use cases)/Yes (float8_e5m2/float8_e4m3fn/float4_E2M1 use cases)</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>l1_weights_sf</td>
            <td>Same as the TensorList length and layout of the corresponding weight; local_moe_expert_num in the per-expert layout and 1 in the stacked layout. For the supported layouts and the exact values, see the constraints below this table.</td>
            <td>No</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>l2_weights_sf</td>
            <td>Same as the TensorList length and layout of the corresponding weight; local_moe_expert_num in the per-expert layout and 1 in the stacked layout. For the supported layouts and the exact values, see the constraints below this table.</td>
            <td>No</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>l1_bias</td>
            <td>num_experts_per_rank</td>
            <td>No</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>l2_bias</td>
            <td>num_experts_per_rank</td>
            <td>No</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>N/A</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
    </tbody>
    </table>

  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT products</term>: the MXFP use case supports two TensorList layouts. For `l1_weights`, `l2_weights`, `l1_weights_sf`, and `l2_weights_sf`, in the per-expert layout, all four TensorLists have a length of `local_moe_expert_num`; tensors in the weight lists are 2D, and tensors in the weight scale factor lists are 3D. In the stacked layout, all four TensorLists have a length of 1; the weight tensors are 3D, the weight scale factor tensors are 4D, and dim0 of each tensor is `local_moe_expert_num`. The four inputs must use the same layout. When the shared expert is enabled, the corresponding four shared expert inputs must also use the same layout as the MoE experts.
  <!-- end id21 -->
  <!-- npu="A3,910b" id29 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `l1_weights`, `l2_weights`, `l1_weights_sf`, and `l2_weights_sf` support only the per-expert layout, and all four TensorLists have a length of `num_experts_per_rank` (= `num_experts` / `ep_world_size`, the same value as `local_moe_expert_num`).
  <!-- end id29 -->

- **Parameter Consistency Constraints**:
  - All input parameters of the mega_moe API and their corresponding tensor dimensions must be consistent with the parameters of the same names in get_symm_buffer_for_mega_moe (for example, `num_experts`, `hidden`, `intermediate_hidden`, etc.).
  - The values of parameters such as `num_experts`, `max_recv_token_num`, `dispatch_quant_mode`, `dispatch_quant_out_dtype`, and `num_max_tokens_per_rank` used in the operator call must be consistent across all devices and across different layers in the network.

- **Communicator constraints**:
    - The value of the `ep_world_size` parameter must be consistent across all devices.
    <!-- npu="A3,910b" id30 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: the communicator buffer size must be consistent. `ccl_buffer_size` is the **total size** (in Bytes) of the CCL communication buffer allocated on HBM. It consists of two equal-sized spaces, **windowIn** and **windowOut**. During verification, the size of a single space, `ccl_buffer_size / 2`, must satisfy:

        $$ccl\_buffer\_size\ /\ 2 \ge \mathrm{offsetTokenPerExpert} + \mathrm{offsetTensor} + \mathrm{offsetFlag} + 10\,\mathrm{MB}$$
    <!-- end id30 -->
    <!-- npu="910b" id11 -->
     **<term>Atlas A2 products</term>:**

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
     **<term>Atlas A3 products</term>:**

    ```text
    offsetTokenPerExpert = ep_world_size × CeilAlign(ep_world_size × maxExpertPerRank + 1, 128) × 4B

    // winIn (winIn only, no winOut)
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
     **<term>Ascend 950PR&950DT products</term>:**

    The communication buffer is automatically calculated and allocated by `get_symm_buffer_for_mega_moe` based on parameters such as the communicator configuration and `num_max_tokens_per_rank`. Users do not need to calculate or set `ccl_buffer_size`. A larger `num_max_tokens_per_rank` value requires more internally allocated communication memory. Set the value appropriately based on the expected maximum number of tokens per device.
    <!-- end id13 -->
  - The driver versions of all nodes in a communicator must be the same.

- **Networking Constraints**:

  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>: A multi-node communicator requires a switch-based network. Dual-device direct connection is not supported.
  <!-- end id14 -->
  <!-- npu="A3" id15 -->
  - <term>Atlas A3 products</term>: A multi-node communicator must be within one SuperPoD. Dual-device direct connection and cross-SuperPoD networking are not supported.
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: With `topo_type` = 0 (MTE topology), the UB Memory protocol is used and each rank accesses the peer communication window directly; with `topo_type` = 1 (URMA cross-SuperPoD topology), URMA (UB_CTP) is used for all peers.
  <!-- end id16 -->

- **Parameter Constraints**:

  <!-- npu="A3,910b" id17 -->
  - **<term>Atlas A2 products</term>, <term>Atlas A3 products</term>:**

    - `num_tokens` must be consistent across devices. Value range: `1 ≤ num_tokens ≤ num_max_tokens_per_rank`.
    - `ep_world_size`: values `2`, `4`, `8`, `16`, `32`, `48`, `64`, `96`, `128`.
    - `num_experts`: value range `world_size ≤ num_experts ≤ 1024`, and `num_experts % ep_world_size == 0`.
    - `num_experts_per_rank`: value range `1 ≤ num_experts_per_rank ≤ 128`, and `num_experts_per_rank = num_experts / world_size`.
    - `num_max_tokens_per_rank`: value range `1 ≤ num_max_tokens_per_rank < +∞`.
    - `max_recv_token_num` must be greater than 0. A value of 0 indicates automatic computation, using the formula `num_tokens × ep_world_size × min(num_topk, local_moe_expert_num)`.
    - `num_topk`: value range `1 ≤ num_topk ≤ 16`.
    - `hidden`: value range `1024 ≤ hidden ≤ 8192`, and `hidden % 512 == 0`.
    - `intermediate_hidden`: value range `512 ≤ intermediate_hidden ≤ 3072`, and `intermediate_hidden % 512 == 0`.
    - `dispatch_quant_mode`: The value range is `0` (no quantization) and `2` (pertoken quantization).
    - `dispatch_quant_out_dtype`: The value is `torch.int8`.
    - Three computation use cases are supported (A16W16, A8W8-INT, A8W4-INT). For each use case, the required status and data types of optional input parameters (scale factors, biases, etc.) are strictly matched. When calling the API, provide all corresponding parameters based on the selected use case. Do not mix or omit parameters. See the following table for the pairing relationships.
        <table>
        <thead>
            <tr>
            <th>Use Case</th>
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
                <td colspan="11">"–" indicates that this parameter is <strong>not required</strong> in this use case. Pass <code>None</code> or keep the default.</td>
            </tr>
            <tr>
                <td colspan="11">A data type written directly (such as <code>uint64</code>) indicates that this parameter is <strong>required</strong> in this use case and must be of this data type.</td>
            </tr>
        </tfoot>
        </table>

  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - **<term>Ascend 950PR&950DT products</term>:**
    - `activation` supports "swiglu", "swiglustep", "swigluoai", and "situglu". `activation_clamp` configures the clamp values of the first three activations; `activation_params` configures `alpha` and `beta` of "swigluoai" as well as `beta` and `linear_beta` of "situglu".
    - num_tokens (x.dim0) is in the range [1, +∞) and must not exceed `num_max_tokens_per_rank` configured when creating `sym_buffer` for each call. The actual num_tokens can vary across ranks, and the same `sym_buffer` can also be used for multiple calls with different num_tokens values.
    - `num_max_tokens_per_rank` must be greater than or equal to 1, and its value must be consistent across all ranks. It is recommended to set it to the maximum number of tokens on a single device that may occur on all ranks during `sym_buffer` reuse. If the original upper bound is exceeded, recreate `sym_buffer` with a larger upper bound.
    - hidden (x.dim1) is in the range [1024, 8192]. In A8W8-FP use cases, hidden must be a multiple of 32; in A8W4-FP use cases, hidden must be a multiple of 64; in A4W4-FP use cases, only l1_weights in FRACTAL_NZ format is supported, and hidden must be a multiple of 64.
    - num_topk (topk_ids.dim1) is in the range [1, 32].
    - num_experts_per_rank is in the range [1, 1024].
    - intermediate_hidden indicates the intermediate feature dimension after activation. It is in the range [256, 4096] and must be aligned to 128. The full output width of Linear1 is 2 × intermediate_hidden.
    - ep_world_size is in the range [2, 1024].
    - num_experts is in the range [ep_world_size, 2048], and num_experts % ep_world_size == 0.
    - max_recv_token_num is in the range [0, num_max_tokens_per_rank × ep_world_size × min(num_topk, local_moe_expert_num)]. It is recommended to keep the default value 0 so that the API automatically calculates the receive capacity.
    - dispatch_quant_out_dtype supports only torch.float8_e5m2, torch.float8_e4m3fn, or torch.float4_e2m1.
    - The current version supports only the MXFP quantization mode (dispatch_quant_mode = 4). The dispatch phase uses MX per-group quantization (group size = 32), and the quantization scale factor is of the FLOAT8_E8M0 type.
    - In the current version, the x_active_mask and scales parameters must be None. Non-None inputs are not supported.
    - combine_quant_mode supports 0 (non-quantized), 3 (MX mode with float8_e5m2 type), and 4 (MX mode with float8_e4m3 type).
    - comm_alg is a reserved parameter and must be the empty string "".
    - The data type of y is the same as that of x.
    - The output dimension of Linear1 in l1_weights must be 2 × intermediate_hidden, and the input dimension of l2_weights must be intermediate_hidden.
    - l1_weights_sf and l2_weights_sf cannot be null pointers.
    - local_moe_expert_num = num_experts / ep_world_size. When the shared expert is enabled, shared_expert_num_per_rank = shared_l1_weights.dim0 and num_experts_per_rank = shared_expert_num_per_rank + local_moe_expert_num. When the shared expert is disabled, shared_expert_num_per_rank = 0.
    - shared_expert_num_per_rank range: [0, 4].
    - topo_type is automatically derived from the communicator context. 0 indicates the MTE topology, and 1 indicates the URMA cross-SuperPoD topology.
    - Under the URMA topology, `activation` does not support "swiglustep" or "swigluoai".
    - topk_weights_type is 0 or 1. 0 disables topkWeights forwarding, and 1 enables it.
    - The communication buffer is automatically calculated and allocated by `get_symm_buffer_for_mega_moe`. Users do not need to calculate or set `ccl_buffer_size`.
    - l1_weights and l2_weights must have the same data type, and only FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT4_E2M1 are supported.
    - The data type of topk_weights supports only BF16 or FP32.

    - **MXFP quantization use case constraints**:
        - The logical shape of l1_weights is (local_moe_expert_num, 2 × intermediate_hidden, hidden), and the logical shape of l2_weights is (local_moe_expert_num, hidden, intermediate_hidden). FLOAT4_E2M1 weights are packed as uint8 on the PyTorch side. The corresponding physical shapes are (local_moe_expert_num, 2 × intermediate_hidden, hidden / 2) and (local_moe_expert_num, hidden, intermediate_hidden / 2).
        - The shape of l1_weights_sf is (local_moe_expert_num, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2), where CeilDiv(hidden, 64) = ⌈hidden / 64⌉ = ⌊(hidden + 63) / 64⌋.
        - The shape of l2_weights_sf is (local_moe_expert_num, hidden, CeilDiv(intermediate_hidden, 64), 2), where CeilDiv(intermediate_hidden, 64) = ⌈intermediate_hidden / 64⌉ = ⌊(intermediate_hidden + 63) / 64⌋.
        - The logical shape of shared_l1_weights is (shared_expert_num_per_rank, 2 × intermediate_hidden, hidden), and the logical shape of shared_l2_weights is (shared_expert_num_per_rank, hidden, intermediate_hidden). FLOAT4_E2M1 weights are passed in as uint8-packed data on the PyTorch side, and the corresponding physical shapes are (shared_expert_num_per_rank, 2 × intermediate_hidden, hidden / 2) and (shared_expert_num_per_rank, hidden, intermediate_hidden / 2).
        - The shape of shared_l1_weights_sf is (shared_expert_num_per_rank, 2 × intermediate_hidden, CeilDiv(hidden, 64), 2), and the shape of shared_l2_weights_sf is (shared_expert_num_per_rank, hidden, CeilDiv(intermediate_hidden, 64), 2).
        - The dim3 of l1_weights_sf and the dim3 of l2_weights_sf must be 2.
        - In the A8W4-FP use case, l1_weights of type FLOAT4_E2M1 must use the FORMAT_FRACTAL_NZ_C0_32 format; in the A4W4-FP use case, l1_weights of type FLOAT4_E2M1 must use the FRACTAL_NZ format. l2_weights in both use cases must use the FORMAT_FRACTAL_NZ_C0_32 format. When the shared expert is enabled, shared_l1_weights and shared_l2_weights must use the same format as l1_weights and l2_weights respectively.
        - In the A8W8-FP use case, l1_weights and l2_weights must both be FLOAT8_E5M2 or both be FLOAT8_E4M3FN, and `weight1_type` and `weight2_type` can be omitted and deduced from the data types of the weight tensors. In the A8W4-FP and A4W4-FP use cases, weights of both layers are FLOAT4_E2M1. `weight1_type` and `weight2_type` must be explicitly set to the type enum corresponding to `float4_e2m1fn_x2`, and the two values must be identical.
        - x_active_mask and scales must be None.
    - Three computation use cases are supported: A8W8-FP, A8W4-FP, and A4W4-FP. For different use cases, the optional input parameters (such as scale factors and biases) are subject to strict matching requirements on necessity and data types. During the call, all corresponding input parameters must be provided in full according to the selected use case. Do not mix use cases or omit parameters. See the following table for the matching relationships.
        <table>
        <thead>
            <tr>
            <th>Use case</th>
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
                <td colspan="11">"–" indicates that this parameter is <strong>not required</strong> in this use case. Pass <code>None</code> or keep the default value.</td>
            </tr>
            <tr>
                <td colspan="11">Entering a data type (such as <code>uint64</code>) indicates that this parameter is <strong>required</strong> in this use case and must use this data type.</td>
            </tr>
        </tfoot>
        </table>

  <!-- end id18 -->

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call:

  The following example chains the two APIs in call order: initialize the communicator first, construct sym_buffer using get_symm_buffer_for_mega_moe, and then call mega_moe to run the operator.

  <!-- npu="950" id19 -->
  - **<term>Ascend 950PR&950DT products</term>**:

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
    scene = "A8W8"  # Options: "A8W8", "A8W4", "A4W4"

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def ceil(a, b):
        return (a + b - 1) // b

    def pack_fp4(weight):
        # This example uses a random uint8 to simulate one byte that packs two FP4 E2M1 values; pass quantized weights in actual use.
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
        # Create the HCCL communication link and initialize the EP domain.
        print(f'[INFO] device_{rank} created HCCL communication link')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")

        print(f"device {rank} initialized EP domain")
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
        print(f'[INFO] device_{rank} constructed input data for the megamoe operator')
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
        # Step 1: Construct distribute_buffer (SymmBuffer structure).
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
        # Step 2: Run mega_moe, passing in the sym_buffer constructed in the previous step.
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

        # Put outputs of all types into the same list. category_outputs stores the lists of outputs of each type.
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
        # Construct inputs.
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
  - **<term>Atlas A2 products</term>, <term>Atlas A3 products</term>**:

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
    server_index = 0  # ID of the current machine in the cluster
    master_addr = '127.0.0.1'
    master_port = 50001
    scene = 'A16W16'

    world_size = num_servers * num_ranks_per_server
    num_experts_per_rank = num_experts // world_size

    torch_npu.npu.config.allow_internal_format = True


    def init_hccl_comm(rank):
        print(f'device_{rank} start init_process_group')
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        # When in use, set the value of hccl_buffer_size according to the communicator buffer size constraint specified in the operator API documentation.
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

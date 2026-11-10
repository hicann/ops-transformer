# MegaMoE

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √    |
| <term>Atlas A3 products</term>       |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                               |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- Operator function: The MegaMoE operator fuses the complete computation flow of the expert FFN of the MoE layer and the preceding and following data communication (that is, Dispatch + Linear1 + Activation + Linear2 + Combine) into a single operator, achieving overlapping of communication and computation.

- Computation formulas:
  - Inputs:
    - $\mathbf{X} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$: activation matrix, corresponding to the input parameter `x`. $\text{totalNumTokens}$ is the global total number of tokens, and $\text{hidden}$ is the hidden dimension.
    - $\mathbf{E} \in \mathbb{Z}^{\text{totalNumTokens} \times \text{topK}}$: matrix of expert indices selected by the tokens, corresponding to the input parameter `topkIds`. $\text{topK}$ is the number of experts selected by each token.
    - $\mathbf{G} \in \mathbb{R}^{\text{totalNumTokens} \times \text{topK}}$: matrix of gating weights of the experts selected by the tokens, corresponding to the input parameter `topkWeights`.
    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{hidden} \times (2 \text{intermediateHidden})}$: Linear1 weights of the routed MoE experts, corresponding to the MoE expert part of the input parameter `weight1`.
    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{intermediateHidden} \times \text{hidden}}$: Linear2 weights of the routed MoE experts, corresponding to the MoE expert part of the input parameter `weight2`.
    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{hidden} \times (2 \text{intermediateHidden})}$: Linear1 weights of the shared experts, corresponding to the input parameter `sharedWeight1`.
    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{intermediateHidden} \times \text{hidden}}$: Linear2 weights of the shared experts, corresponding to the input parameter `sharedWeight2`.
  - Outputs:

    - $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$: final output matrix, corresponding to the output parameter `y`.
  - Conventions:
    - $⋅$ denotes matrix multiplication, and $⊙$ denotes element-wise multiplication.
    - $\left \lfloor z\right \rceil$ denotes rounding $z$ to the nearest integer, and $\left \lfloor z\right \rfloor$ denotes rounding $z$ down.
    - $|z|$ denotes taking the absolute value, and $\max(z)$ denotes taking the maximum value.
    - The set of all tokens is $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{totalNumTokens} - 1\} \}$.
    - The token representation (that is, the hidden state vector) of $\text{token}_i$ is $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$, and $\mathbf{x}_i = \mathbf{X}[i,:]$.
    - The expert indices of $\text{token}_i$ are $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{topK} - 1\},\quad e_{i,k} \in \{0,\dots,\text{moeExpertNum} - 1\}$.
    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$. Here, the superscript $\text{sym}$ of $\mathbb{Z}_8^{\text{sym}}$ denotes a symmetric quantization value range: its value range is rounded symmetrically about $-127$ and $127$, which differs from the $[-128, 127]$ value range of standard INT8, hence the $\text{sym}$ superscript is used for distinction.
    - Tensor slicing uses the Python-style `start:stop:step` notation; for example, $[0::2, :]$ selects the even rows, and $[1::2, :]$ selects the odd rows.
    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ denotes the binary reinterpretation operation, which reinterprets the underlying binary data of the tensor $\mathbf{Z}$ as the target type $T$.
  - <span id="activation-formulas">Activation function formulas:</span>
    - Let the gate branch of the split Linear1 output be $\mathbf{G}$, the up branch be $\mathbf{U}$, and the activation output be $\mathbf{A}$. In the following text, the specific activation computation jointly determined by `activation` and `activation_params` is collectively referred to as Activation. The Sigmoid function and the SiLU function are defined as:
      $$
      \sigma(z)=\frac{1}{1+e^{-z}}, \qquad
      \operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}.
      $$
      Let the clamp value be $c$; when clamping is not enabled (not configured or configured as $0$), mathematically $c=+\infty$. Define
      $\mathbf{G}_c=\min(\mathbf{G},c)$, $\mathbf{U}_c=\operatorname{clip}(\mathbf{U},-c,c)$.
    - `swiglu`:
      $$
      \mathbf{A}=\operatorname{SwiGLU}(\mathbf{G},\mathbf{U})
      =\operatorname{Swish}_1(\mathbf{G}_c)\odot\mathbf{U}_c
      =\operatorname{SiLU}(\mathbf{G}_c)\odot\mathbf{U}_c.
      $$
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
      \mathbf{U}, & \text{linear\_beta not configured},\\
      \beta_{\mathrm{linear}}\tanh\!\left(\dfrac{\mathbf{U}}{\beta_{\mathrm{linear}}}\right),
      & \text{linear\_beta configured}.
      \end{cases}
      $$
    - `gelu`: element-wise activation, without gate/up splitting, applied directly to the complete output $\mathbf{Z}$ of Linear1, and unaffected by the clamp value $c$. The tanh approximation is used:
      $$
      \mathbf{A}=\operatorname{GELU}(\mathbf{Z})
      =\frac{1}{2}\mathbf{Z}\left(1+\tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(\mathbf{Z}+0.044715\,\mathbf{Z}^{3}\right)\right)\right)
      =\mathbf{Z}\odot\sigma\!\left(\sqrt{\tfrac{8}{\pi}}\left(\mathbf{Z}+0.044715\,\mathbf{Z}^{3}\right)\right).
      $$
      where $\mathbf{Z}$ is the complete output of Linear1. The Linear1 output width of `weight1` is always 2 × `intermediateHidden`, regardless of `activation`: gated activation splits it into halves for gate and up, and the activation output width is `intermediateHidden`; `gelu` does not split, and the entire segment participates in the activation, so the activation output width is 2 × `intermediateHidden`. Therefore, the expressions used elsewhere in this document, such as "$W_1$ is split into two halves along the column direction", "activation output dimension $N/2$", and using `intermediateHidden` to denote the activation output width, the input dimension of `weight2`, and the last dimension of `weightScales2`, are all premised on gated activation; when `activation` = "gelu", the corresponding positions are 2 × `intermediateHidden`, while the `weight1`-side expressions in the form of 2 × `intermediateHidden` apply to both activations.
- Computation process

    The data types of the activation matrix A and the weight matrix W during **Linear computation** supported by each product are as follows:

    | Scenario    |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN, FLOAT8_E5M2 |FLOAT8_E4M3FN, FLOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the A8W8-FP scenario in the preceding table is not supported.
    - <term>Ascend 950PR&950DT products</term>: the A16W16, A8W8-INT, and A8W4-INT scenarios in the preceding table are not supported.
    - The dispatch stage of <term>Ascend 950PR&950DT products</term> has two paths: when `dispatchQuantMode` = 4, the operator performs MX group-wise quantization on BF16 `x` internally as described in each quantization scenario below; when `dispatchQuantMode` = 0, it is pre-quantized pass-through, where `x` is already quantized data and the scale factors are given by the `scales` input parameter. The operator skips the quantization step of the first stage described below and directly uses `x` and `scales` in the subsequent matrix multiplications; the other stages are the same as described below.
    - <term>Ascend 950PR&950DT products</term>: MoE experts and shared experts can adopt different weight types and quantization types. The quantization output type of the shared experts is specified by `sharedExpertQuantOutDtype`; when not explicitly configured, it is the same as `dispatchQuantOutDtype`.

    <details>
    <summary> A16W16 non-quantization scenario</summary>

    - **EP Dispatch**

        In the Dispatch stage, each $\text{token}_i$ sends its token representation $\mathbf{x}_i$ to the experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$. That is, for each $k$, expert $e_{i,k}$ receives a copy of $\mathbf{x}_i$.

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the set of token indices dispatched to expert $e$. The size $N_e = |I_e|$ of the set $I_e$ is the total number of tokens that expert $e$ needs to process. Then $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ is the matrix formed by **row-stacking in any fixed order** the token representations $\mathbf{x}_i$ of all $i$ satisfying $i \in I_e$; this matrix is the complete set of token representations received by expert $e$ after Dispatch.

        For each $\text{token}_i$ and its $k$-th selected expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$. This mapping records the position of $\mathbf{x}_i$ in the input matrix $\mathbf{X}_{e_{i,k}}$ of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts a gated activation structure specified by `activation` to improve expressiveness. The entire computation process consists of the following three sub-steps.

        **1. Linear1 projection**

        The Linear1 projection is the first linear transformation of the expert network, and it simultaneously produces the pre-activation values required for the **gate part** and the **up part**. The computation formula is
        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}\bigr)
        $$
        **2. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, and compute the intermediate activation representation $\mathbf{A}_e$ according to `activation`; for the computation methods of each optional activation type, see [Activation function formulas](#activation-formulas).

        **3. Linear2 projection**

        As the second linear transformation, the Linear2 projection projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, so that the expert output is compatible with subsequent operations such as residual connections.
        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        After the preceding computation, each output row of expert $e$ corresponds to one token in its batch. For $\text{token}_i$, its output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.
    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs a weighted sum according to the expert weights originally assigned to each token, and finally generates a fused output for each token. Using the position indices $\operatorname{row}(i,k)$ recorded earlier, retrieve the rows belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiply them by the gating weights, and sum the results:

        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in input order as the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.
    </details>

    <details>
    <summary> A8W8-INT quantization scenario</summary>

    - **Inputs**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: per-channel scale factors of the Linear1 weight matrix, corresponding to the input parameter `weightScales1`.
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: per-channel scale factors of the Linear2 weight matrix, corresponding to the input parameter `weightScales2`.

    - **EP Dispatch**

        Before the Dispatch communication, the original BF16 activation matrix $\mathbf{X}$ is first quantized to INT8. For each $\text{token}_i$, compute its per-token scale factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        Then quantize to obtain the INT8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        In the Dispatch communication stage, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and scale factor $s^{X}_i$ to the experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$ ($e_{i,k} = \mathbf{E}[i,k]$).

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the set of token indices dispatched to expert $e$. The size $N_e = |I_e|$ of the set $I_e$ is the total number of tokens that expert $e$ needs to process. Then $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is the matrix formed by **row-stacking in any fixed order** the $\mathbf{q}_i$ of all $i$ satisfying $i \in I_e$; this matrix is the complete set of token representations received by expert $e$ after Dispatch. Similarly, the scale factor vector received by expert $e$ is denoted $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are the $s^{X}_i$ of all $i$ satisfying $i \in I_e$ stacked in the same row order as $\mathbf{Q}_e$.

        For each $\text{token}_i$ and its $k$-th selected expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts a gated activation structure specified by `activation` to improve expressiveness. In the A8W8 scenario, both linear layers perform matrix multiplication with INT8 inputs and INT8 weights, obtain INT32 intermediate results, and then dequantize. The process is divided into three sub-steps.

        **1. Linear1 projection (INT8 matrix multiplication + dequantization)**

        The Linear1 projection is the first linear transformation of the expert network, and it simultaneously produces the pre-activation values required for the **gate part** and the **up part**. During computation, INT8 matrix multiplication is performed to obtain the INT32 computation result:
        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$
        Then dequantize to obtain the pre-activation value $\mathbf{H}_e$:
        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **2. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, and compute the intermediate activation representation $\mathbf{A}_e$ according to `activation`; for the computation methods of each optional activation type, see [Activation function formulas](#activation-formulas).

        **3. Linear2 projection (quantization + INT8 matrix multiplication + dequantization)**

        As the second linear transformation, the Linear2 projection projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, so that the expert output is compatible with subsequent operations such as residual connections.
        In the A8W8 scenario, the activation values need to be quantized to INT8. Therefore, first compute the scale factor for each row of $\mathbf{A}_e$ (each token):
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        This yields the activation scale factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ of expert $e$ for the Linear2 computation. Then quantize:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$
        Then perform INT8 matrix multiplication and dequantize:
        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        After the preceding computation, each output row of expert $e$ corresponds to one token in its batch. For $\text{token}_i$, its output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs a weighted sum according to the expert weights originally assigned to each token, and finally generates a fused output for each token. Using the position indices $\operatorname{row}(i,k)$ recorded earlier, retrieve the rows belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiply them by the gating weights, and sum the results:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in input order as the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.

    </details>

    <details>
    <summary> A8W4-INT quantization scenario</summary>

    - **Inputs**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: per-channel scale factors of the Linear1 weight matrix, corresponding to the input parameter `weightScales1`.
        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: per-channel scale factors of the Linear2 weight matrix, corresponding to the input parameter `weightScales2`.
        - $\mathbf{B}_1 \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$: bias matrix of Linear1, generated offline by the INT4 quantization process, corresponding to the input parameter `bias1`.
        - $\mathbf{B}_2 \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$: bias matrix of Linear2, generated offline by the INT4 quantization process, corresponding to the input parameter `bias2`.

    - **EP Dispatch**

        Before the Dispatch communication, the original BF16 activation matrix $\mathbf{X}$ is first quantized to INT8. For each $\text{token}_i$, compute its per-token scale factor:
        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        Then quantize to obtain the INT8 representation:
        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        In the Dispatch communication stage, each $\text{token}_i$ sends its quantized vector $\mathbf{q}_i$ and scale factor $s^{X}_i$ to the experts $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$

        Let $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ denote the set of token indices dispatched to expert $e$. The size $N_e = |I_e|$ of the set $I_e$ is the total number of tokens that expert $e$ needs to process. Then $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ is the matrix formed by **row-stacking in any fixed order** the $\mathbf{q}_i$ of all $i$ satisfying $i \in I_e$; this matrix is the complete set of token representations received by expert $e$ after Dispatch. Similarly, the scale factor vector received by expert $e$ is denoted $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$, whose elements are the $s^{X}_i$ of all $i$ satisfying $i \in I_e$ stacked in the same row order as $\mathbf{Q}_e$.

        For each $\text{token}_i$ and its $k$-th selected expert $e_{i,k}$, there exists a unique row index $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$ such that $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$. This mapping records the position of $\mathbf{q}_i$ in the input matrix of expert $e_{i,k}$.

    - **Expert Compute**

        In the MoE layer, each expert is essentially an independent feedforward network (FFN) that adopts a gated activation structure specified by `activation` to improve expressiveness. In the A8W4-INT scenario, both linear layers use the MSD (Mixed-precision Split-activation Decomposition) scheme for matrix multiplication. By splitting an INT8 value into two signed INT4 values (the high 4 bits and the low 4 bits), the INT8×INT4 matrix multiplication can be decomposed into two INT4×INT4 matrix multiplications, thereby leveraging the hardware acceleration of INT4 matrix multiplication. For the mathematical principle and implementation logic of this scheme, see [GroupedMatmul W4A8 Quantization and MSD Scheme](https://atomgit.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81).

        **1. Generate the accuracy compensation bias matrix (generated offline outside the operator and passed in as an operator input)**

        The core of the MSD scheme is to split the quantized INT8 activation values bit-wise into two INT4 components. Because the low INT4 component is defined as $(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$, its value range is mapped to $[-8, 7]$, which is equivalent to subtracting 8 from the original unsigned low 4 bits (0 to 15). When the high and low INT4 components are multiplied with the weight matrix separately and then merged, this offset introduces a constant term that must be compensated in the final result.

        Let the original INT8 activation matrix be $\mathbf{X}^{\text{int8}}$, the split high and low INT4 activation matrices be $\mathbf{X}_1^{\text{int4}}$ and $\mathbf{X}_2^{\text{int4}}$, and the weight matrix be $\mathbf{W}$. The recovery relation is:

        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        where $\mathbf{1}_{\text{mat}}$ is an all-ones matrix with the same shape as $\mathbf{X}_2^{\text{int4}}$, used to add 8 element-wise. Let $\mathbf{1}$ be an all-ones column vector whose shape matches the input feature dimension. Then the matrix multiplication expands to:

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        Here, $\mathbf{1}^\top \cdot \mathbf{W}$ denotes the sum over each column of the weight matrix $\mathbf{W}$, and the result is a row vector (with the same shape as the output dimension). In the actual implementation, this row vector is broadcast along the batch dimension and added to the outputs of all tokens. As can be seen, if only the first two terms are computed, a positive term is missing. To recover the result precisely, the compensation bias must be precomputed:

        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        Both $\mathbf{B}_1$ and $\mathbf{B}_2$ follow this method: multiply the corresponding all-ones column vectors (with dimensions $\text{hidden}$ and $\text{intermediateHidden}$ respectively) by their respective weight matrices to obtain shape-matched bias row vectors. These biases are computed in the offline stage and then passed to the operator, and they directly participate in the addition when the high and low results are merged, thereby eliminating the error introduced by the offset.

        **2. Linear1 projection (INT4 × INT4 matrix multiplication + dequantization)**

        The Linear1 projection is the first linear transformation of the expert network, and it simultaneously produces the pre-activation values required for the **gate part** and the **up part**.

        **(1) Reinterpret activations as INT4**

        Reinterpret the INT8 activation tensor $\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ at the binary level into a view of two interleaved INT4 parts:

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        where the high and low components are given directly by the even rows and the odd rows of $\mathbf{Q}_e^{\text{int4}}$:

        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        Numerically, they can be obtained from the original INT8 through the following computation:

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        The recovery relation is $\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the type view, $\mathbf{Q}_e^{\text{int4}}$ and $\mathbf{Q}_e^{\text{int8}}$ share the underlying physical memory, without any data rearrangement or copying.

        **(2) INT4 × INT4 matrix multiplication and weight dequantization**

        Multiply the reinterpreted INT4 activation view $\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ by the weight matrix $\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediateHidden}}$, and apply the weight scale factor $\mathbf{s}^{W1}_e$ for dequantization to obtain the result:

        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediateHidden}}
        $$

        Record the result as the following matrix views by even rows and odd rows respectively; they correspond to the computation results of $\mathbf{Q}_e^{\text{high}}$ and $\mathbf{Q}_e^{\text{low}}$:

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **(3) Accuracy compensation and activation dequantization**

        Use the bias $\mathbf{B}_1[e]$ and the activation scale factor $\mathbf{s}^{X}_e$ (row vector) to perform accuracy compensation and activation dequantization respectively. The final pre-activation value is:

        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        where $\odot \mathbf{s}^{X}_e$ denotes multiplying each row of the matrix by the corresponding scalar in $\mathbf{s}^{X}_e$.

        **3. Activation**

        Split $\mathbf{H}_e$ along the column dimension into the gate branch $\mathbf{G}_e$ and the up branch $\mathbf{U}_e$, and compute the intermediate activation representation $\mathbf{A}_e$ according to `activation`; for the computation methods of each optional activation type, see [Activation function formulas](#activation-formulas).

        **4. Linear2 projection (quantization + INT4 × INT4 matrix multiplication + dequantization)**

        As the second linear transformation, the Linear2 projection projects the intermediate activation representation $\mathbf{A}_e$ from the high-dimensional space back to the original hidden dimension $\text{hidden}$, so that the expert output is compatible with subsequent operations such as residual connections.
        In the A8W4 scenario, first quantize the activation values to INT8, then reinterpret them at the binary level into a concatenated view of two INT4 parts to complete the computation with a single matrix multiplication. The process is as follows.

        **(1) Activation quantization**

        Compute the scale factor for each row of $\mathbf{A}_e$ and quantize to INT8:
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        This yields the activation scale factor $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$ of expert $e$ for the Linear2 computation, and compute the quantization result:
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$

        **(2) Reinterpret activations as INT4**

        Reinterpret the INT8 activation tensor $\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}$ at the binary level into a view of two interleaved INT4 parts:

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}
        $$

        where the high and low components are given directly by the even rows and the odd rows of $\mathbf{A}_e^{\text{int4}}$:

        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}
        $$

        Numerically, they can be obtained from the original INT8 through the following computation:

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        The recovery relation is $\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$. Since $\mathrm{bitcast}$ only changes the type view, $\mathbf{A}_e^{\text{int4}}$ and $\mathbf{A}_e^{\text{int8}}$ share the underlying physical memory, without any data rearrangement or copying.

        **(3) INT4 × INT4 matrix multiplication and weight dequantization**

        Multiply the reinterpreted INT4 activation view $\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}$ by the weight matrix $\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediateHidden} \times \text{hidden}}$, and apply the weight scale factor $\mathbf{s}^{W2}_e$ for dequantization to obtain the result:

        $$
        \mathbf{D}_e = \bigl( \mathbf{A}_e^{\text{int4}} \cdot \mathbf{W}_2[e] \bigr) \odot \mathbf{s}^{W2}_e
        \;\in\; \mathbb{R}^{2N_e \times \text{hidden}}
        $$

        Record the result as the following matrix views by even rows and odd rows respectively; they correspond to the computation results of $\mathbf{A}_e^{\text{high}}$ and $\mathbf{A}_e^{\text{low}}$:

        $$
        \mathbf{D}_e^{\text{high}} = \mathbf{D}_e[0::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}, \qquad
        \mathbf{D}_e^{\text{low}}  = \mathbf{D}_e[1::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        **(4) Accuracy compensation and activation dequantization**

        Use the bias $\mathbf{B}_2[e]$ and the activation scale factor $\mathbf{s}^{A_e}_e$ (row vector) to perform accuracy compensation and activation dequantization respectively. The final output is:

        $$
        \mathbf{Y}_e = \Bigl( 16 \cdot \mathbf{D}_e^{\text{high}} + \mathbf{D}_e^{\text{low}} + \mathbf{B}_2[e] \Bigr) \odot \mathbf{s}^{A_e}_e
        \quad\in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        where $\odot \mathbf{s}^{A_e}_e$ denotes multiplying each row of the matrix by the corresponding scalar in $\mathbf{s}^{A_e}_e$.

        After the preceding computation, each output row of expert $e$ corresponds to one token in its batch. For $\text{token}_i$, its output row in expert $e_{i,k}$ is $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$.

    - **Token Combine**

        Combine collects the output vectors computed by all experts, performs a weighted sum according to the expert weights originally assigned to each token, and finally generates a fused output for each token. Using the position indices $\operatorname{row}(i,k)$ recorded earlier, retrieve the rows belonging to $\text{token}_i$ from the output matrix of expert $e_{i,k}$, multiply them by the gating weights, and sum the results:
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        where $w_k = \mathbf{G}[i,k]$ is the gating weight of $\text{token}_i$ for expert $e_{i,k}$.

        The outputs of all tokens are stacked in input order as the final output $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$.

    </details>

    <details>
    <summary> A8W8-FP quantization scenario</summary>

    In the first stage, the input tokens are grouped and collected by expert and then quantized with MXFP8, generating the quantized input and scale factors of each expert:

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    Description: Tokens are sorted and collected by expert according to `topkIds`; $\mathcal{T}_e$ is the set of token indices assigned to expert $e$, and $E_{\text{local}}$ is the number of routed MoE experts on this rank. $X[\mathcal{T}_e]$ is the input submatrix corresponding to expert $e$, whose number of rows is determined by the number of tokens actually received by that expert in this call. $\mathrm{Q}_{\text{MX}}$ denotes MX group-wise quantization (group size = 32): for each group of 32 elements, the shared exponent is extracted and the group is quantized to the FP8 target type (FLOAT8_E5M2 or FLOAT8_E4M3FN), while FLOAT8_E8M0 scale factors are output. The quantized data serves as the input of GMM1.

    In the second stage, GMM1 matrix multiplication (with $W_1$ split into two halves along the column direction and computed separately), Activation, and MX quantization are performed for each expert:

    $$
    G_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(G)}, S_{1,e}^{(G)}), \quad U_e = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(U)}, S_{1,e}^{(U)})
    $$

    $$
    A_e = \operatorname{Activation}(G_e,U_e)
    $$

    $$
    \hat{A}_e,\ S_{A,e} = \mathrm{Q}_{\text{MX}}(A_e)
    $$

    Description: The first $N/2$ columns $W_{1,e}^{(G)}$ and the last $N/2$ columns $W_{1,e}^{(U)}$ of $W_1$ are multiplied with the MX-dequantized input respectively, obtaining the gate branch $G_e$ and the up branch $U_e$. For the computation methods of the specific activation functions, see [Activation function formulas](#activation-formulas); the output dimension is $N/2$. Then MX quantization is applied to the activation output; the quantization type is the same as that in the Dispatch stage, that is, both are `FLOAT8_E5M2` or both are `FLOAT8_E4M3FN`, obtaining the quantized input $\hat{A}_e$ of GMM2.

    In the third stage, GMM2 matrix multiplication is performed for each expert, and the results are distributed to the target ranks:

    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{A}_e, S_{A,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    Description: The quantized activation output is multiplied with the second group of weights $W_2$ after MX dequantization, mapping the $N/2$-dimensional intermediate representation back to the $H$-dimensional hidden space and obtaining the output $O_e$ of each expert. After the computation, the results are written to the remote end through RDMA peermem at the expert offset addresses of the target ranks, achieving cross-rank aggregation.

    When shared experts are enabled (`sharedExpertNumPerRank` > 0), each shared expert locally performs, on its own card and for all tokens on that card, the same GMM1 + Activation + GMM2 computation as the routed experts, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, and `sharedWeightScales2`, without participating in Dispatch communication. The output of each shared expert is denoted $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    In the fourth stage, a weighted sum is computed over all tokens according to the routing weights, the shared expert outputs are added, and the result is restored to an output with the same shape as the input:

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i]
    $$

    Description: For each token $i$, retrieve the corresponding rows from the aggregated expert results according to the sorted routing indices $\pi(i,k)$, weight and accumulate them element-wise with the weights in `topkWeights`, and then directly add the outputs of the shared experts for the current token, obtaining the final output $Y$. When shared experts are not enabled, the shared expert summation term is zero.

    Here, $X$ denotes the parameter `x`, $W$ denotes the parameter `topkWeights`, $W_1$ denotes the parameter `weight1`, $W_2$ denotes the parameter `weight2`, $Y$ denotes the parameter `y`, $E_{\text{local}}$ denotes `localMoeExpertNum = moeExpertNum / epWorldSize` (the number of routed MoE experts per rank), and $K$ denotes the second dimension of `topkIds`.

    Description of local variables:
    - $\mathcal{T}_e$: set of token indices routed to expert $e$, determined by sorting `topkIds`.
    - $m_e=|\mathcal{T}_e|$: number of tokens actually received by expert $e$ in this call; $m_e$ can differ across experts.
    - $\hat{X}_e,\ S_{X,e}$: quantized input of expert $e$ and its MX scale factors, an intermediate result of the first stage.
    - $W_{1,e}^{(G)}$ and $W_{1,e}^{(U)}$: submatrices of the first $N/2$ columns and the last $N/2$ columns of $W_1$ corresponding to expert $e$, derived by splitting `weight1` into the gate branch and the up branch.
    - $S_{1,e}^{(G)}$ and $S_{1,e}^{(U)}$: MX scale factors corresponding to $W_{1,e}^{(G)}$ and $W_{1,e}^{(U)}$, sliced from `weightScales1` by dimension.
    - $S_{2,e}$: MX scale factor corresponding to $W_{2,e}$, from the parameter `weightScales2`.
    - $G_e,\ U_e$: gate branch and up branch outputs of GMM1, intermediate results.
    - $A_e$: activation output with dimension $m_e \times N/2$, an intermediate result.
    - $\hat{A}_e,\ S_{A,e}$: quantized activation output and its MX scale factors, intermediate results.
    - $O_e$: expert-level output of GMM2 with dimension $m_e \times H$, an intermediate result.
    - $\pi(i, k)$: row index of the $k$-th top-k expert of token $i$ after flattening and sorting, determined by the routing order.
    - $\mathrm{Q}_{\text{MX}}(\cdot)$: MX group-wise quantization operation with block size = 32, outputting FP8 data and E8M0 scale factors.
    - $\mathrm{DQ}_{\text{MX}}(\cdot)$: MX group-wise dequantization operation, implicitly performed inside the matmul.
    </details>

    <details>
    <summary> A8W4-FP quantization scenario</summary>

    Stage 1 (token selection, quantization, and Dispatch):

    For the input tokens $X \in \mathbb{R}^{B\times H}$ of this rank, obtain the token index set $T_e$ corresponding to each expert $e$ according to `topkIds`, and quantize the selected BF16 tokens into MXFP8 E4M3 in groups of 32 elements:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    where the data type of $\hat{X}_e$ is `FLOAT8_E4M3FN` and the data type of $S_{X,e}$ is `FLOAT8_E8M0`. The quantized tokens and their scale factors are then sent to the ranks where the experts reside.

    The first-layer and second-layer weights $W_{1,e}$ and $W_{2,e}$ of the routed MoE experts are both MXFP4 E2M1 data, with E8M0 scale factors, and $e$ ranges over $[0,\text{localMoeExpertNum})$. The shared expert weights are provided separately by `sharedWeight1` and `sharedWeight2`, whose first dimension is `sharedExpertNumPerRank`. The A8W4 kernel processes FP4 weights in the matmul Prologue, and the logical data flow fed into the matmul is FP8 activation multiplied by FP4 weight.

    Stage 2 (GMM1, Activation, and re-quantization):

    For the tokens received by expert $e$, the first-layer grouped matrix multiplication and the activation function are computed as:

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

    For the computation methods of the specific activation functions, see [Activation function formulas](#activation-formulas). The output is further quantized into MXFP8 E4M3 in groups of 32 elements, consistent with the quantization type in the Dispatch stage, both being `FLOAT8_E4M3FN`, for use by the second-layer matrix multiplication:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    Stage 3 (GMM2):

    The second-layer matrix multiplication is still A8W4, that is, FP8 E4M3 activation multiplied by FP4 E2M1 weight:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When shared experts are enabled (`sharedExpertNumPerRank` > 0), each shared expert locally performs, on its own card and for all tokens on that card, the same GMM1 + Activation + GMM2 computation as the routed experts, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, and `sharedWeightScales2`, without participating in Dispatch communication. The output of each shared expert is denoted $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    Stage 4 (Combine and weighted merging):

    The outputs of the experts are sent back to the original rank, weighted merging is performed according to `topkWeights`, and the shared expert outputs are added:

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert summation term is zero. The data type of the final output $Y$ is BF16. The main data type flow of A8W4-FP is `BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`.
    </details>

    <details>
    <summary> A4W4-FP quantization scenario</summary>

    Stage 1 (token selection, quantization, and Dispatch):

    For the token set $T_e$ corresponding to each expert $e$, quantize the selected BF16 tokens into MXFP4 E2M1 in groups of 32 elements:

    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    where the data type of $\hat{X}_e$ is `FLOAT4_E2M1` and the data type of $S_{X,e}$ is `FLOAT8_E8M0`. The first-layer and second-layer weights $W_{1,e}$ and $W_{2,e}$ of the routed MoE experts are both MXFP4 E2M1, and $e$ ranges over $[0,\text{localMoeExpertNum})$; the first dimension of the shared expert weights is `sharedExpertNumPerRank`. The weight scale factors are E8M0.

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

    For the computation methods of the specific activation functions, see [Activation function formulas](#activation-formulas). In the A4W4-FP scenario, the activation result after the first-layer matrix multiplication is quantized into MXFP8 E4M3 in groups of 32 consecutive elements, with E8M0 scale factors, and serves as the input of the second-layer matrix multiplication. This design avoids the large accuracy loss that would result from using A4W4 for both layers of matrix multiplication:

    $$
    \hat{A}_e,\;S_{A,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(A_e\right).
    $$

    Stage 3 (A8W4 GMM2):

    Because the quantized activation output is FP8 E4M3 while the second-layer weights remain FP4 E2M1, the second-layer matrix multiplication is actually A8W4 rather than A4W4:

    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{A}_e,S_{A,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    When shared experts are enabled (`sharedExpertNumPerRank` > 0), each shared expert locally performs, on its own card and for all tokens on that card, the same GMM1 + Activation + GMM2 computation as the routed experts, using `sharedWeight1`, `sharedWeight2`, `sharedWeightScales1`, and `sharedWeightScales2`, without participating in Dispatch communication. The output of each shared expert is denoted $O^{\mathrm{shared}}_s$, $s \in \{0, \dots, \text{sharedExpertNumPerRank} - 1\}$.

    Stage 4 (Combine and weighted merging):

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)] + \sum_{s=0}^{\text{sharedExpertNumPerRank}-1} O^{\mathrm{shared}}_s[i],
    $$

    When shared experts are not enabled, the shared expert summation term is zero. The data type of the final output $Y$ is BF16. The complete data type flow of A4W4-FP is `BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`. All MX scale factors are of the type `FLOAT8_E8M0`, and the quantization granularity is 32 consecutive elements.
    </details>

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>Input</td>
   <td>Communicator information data of the local card.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>Input</td>
   <td>Token hidden states input to the MoE layer. The shape is (BS, H), where H is the logical hidden dimension and is consistent with the last dimension of weight1. Except in the pre-quantized pass-through scenario, the data type of x must be BF16, and quantization (if any) is performed inside the operator. In the pre-quantized pass-through scenario (dispatchQuantMode is 0 and dispatchQuantOutDtype is FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT4_E2M1), the operator does not perform quantization internally; in this case, the data type of x must be consistent with the logical type specified by dispatchQuantOutDtype, and scales must be passed in at the same time. For the data types supported by each product, see the Constraints section.</td>
   <td>BF16, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkIds</td>
   <td>Input</td>
   <td>Expert index matrix, indicating the topK experts selected by each token. The value range of the elements is [0, moeExpertNum), and the topK experts selected by the same token cannot be duplicated.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkWeights</td>
   <td>Input</td>
   <td>Gating weight coefficients corresponding to the topK experts selected by the expert gating network of the MoE model for the current input tokens.</td>
   <td>FP32, BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td>Input</td>
   <td>Weight matrix of the first linear layer of the MoE expert network (including gating and up-projection), used to map the input to the intermediate dimension, with the output fed to the activation function. The number of routed MoE experts is <code>localMoeExpertNum</code>.</td>
   <td>BF16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>Input</td>
   <td>Weight matrix of the second linear layer of the MoE expert network, which projects the activated intermediate features back to the hidden dimension. The data type is the same as that of weight1. The number of routed MoE experts is <code>localMoeExpertNum</code>.</td>
   <td>BF16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weightScales1</td>
   <td>Optional input</td>
   <td>Quantization scale factors of the weight matrix of the first linear layer of the MoE expert network. Required in MXFP quantization scenarios.</td>
   <td>FLOAT8_E8M0, UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScales2</td>
   <td>Optional input</td>
   <td>Quantization scale factors of the weight matrix of the second linear layer of the MoE expert network. Required in MXFP quantization scenarios.</td>
   <td>FLOAT8_E8M0, UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias1</td>
   <td>Optional input</td>
   <td>Bias of the first linear layer of the MoE expert network. This parameter is required only in the A8W4-INT quantization scenario and is used for accuracy compensation.</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias2</td>
   <td>Optional input</td>
   <td>Bias of the second linear layer of the MoE expert network. This parameter is required only in the A8W4-INT quantization scenario and is used for accuracy compensation.</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td>Optional input</td>
   <td>Indicates whether a token participates in communication.</td>
   <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>Optional input</td>
   <td>Dequantization scale factors of x, giving the scale values corresponding to each quantization group of x. Required in the pre-quantized pass-through scenario (dispatchQuantMode is 0 and dispatchQuantOutDtype is FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT4_E2M1), with shape (BS, CeilDiv(H, 32)); the data type must be FLOAT8_E8M0. In other scenarios, a null pointer must be passed in. For the support status of each product, see the Constraints section.</td>
   <td>FLOAT8_E8M0, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeight1</td>
   <td>Optional input</td>
   <td>Weight matrix of the first linear layer of the shared expert network (including gating and up-projection), used to map the input to the intermediate dimension, with the output fed to the activation function. The data type can differ from weight1 (the weight of the first linear layer of the MoE experts). MoE experts and shared experts can use different weight types and quantization types (for example, W8 for MoE experts with W4 for shared experts, or W4 for MoE experts with W8 for shared experts). The data types of sharedWeight1 and sharedWeight2 must be the same.</td>
   <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeight2</td>
   <td>Optional input</td>
   <td>Weight matrix of the second linear layer of the shared expert network, which projects the activated intermediate features back to the hidden dimension. The data type can differ from weight2 (the weight of the second linear layer of the MoE experts); the value requirements are the same as those of sharedWeight1.</td>
   <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
   <td>ND, FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeightScales1</td>
   <td>Optional input</td>
   <td>Quantization scale factors of the weight matrix of the first linear layer of the shared expert network.</td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeightScales2</td>
   <td>Optional input</td>
   <td>Quantization scale factors of the weight matrix of the second linear layer of the shared expert network.</td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias1</td>
   <td>Optional input</td>
   <td>Bias of the first linear layer of the shared expert network. Not supported currently.</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias2</td>
   <td>Optional input</td>
   <td>Bias of the second linear layer of the shared expert network. Not supported currently.</td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>Attribute</td>
   <td>Total number of experts in the MoE model.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Attribute</td>
   <td>Size of the expert parallel communicator.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>cclBufferSize</td>
   <td>Attribute</td>
   <td>CCL communication buffer size.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>maxRecvTokenNum</td>
   <td>Optional attribute</td>
   <td>Maximum number of tokens that each rank can receive. The default value is 0. When this value is 0, memory is reserved based on the maximum value bs*epWorldSize*min(topK,localMoeExpertNum). When this value is not 0, memory is reserved according to the input value. In this scenario, the user must ensure that the filled-in value is greater than or equal to the maximum number of tokens that each rank can receive.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantMode</td>
   <td>Optional attribute</td>
   <td>Quantization mode during dispatch communication. 0 indicates that the operator does not perform dispatch quantization internally, 2 indicates INT8 pertoken quantization, and 4 indicates that the operator performs MX group-wise quantization internally. The default value is 0. For the values supported by each product, the corresponding computation scenarios, and the specific meaning of value 0, see the Constraints section.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantOutDtype</td>
   <td>Optional attribute</td>
   <td>Data type of the output after dispatch quantization. Supported values are 1 (INT8), 23 (FLOAT8_E5M2), 24 (FLOAT8_E4M3FN), and 296 (FLOAT4_E2M1). The default value is DT_UNDEFINED. When dispatchQuantMode is 4, this attribute indicates the output type of quantization inside the operator. In the pre-quantized pass-through scenario (dispatchQuantMode is 0 and this attribute is 23, 24, or 296), it indicates the logical type of the pre-quantized x, and in this case it must be consistent with the actual data type of x. For the value ranges actually supported by each product, see the Constraints section.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>sharedExpertQuantOutDtype</td>
   <td>Optional attribute</td>
   <td>Quantization data type of the shared expert inputs. The default value is DT_UNDEFINED, which indicates consistency with dispatchQuantOutDtype. When explicitly configured, supported values are 23 (FLOAT8_E5M2), 24 (FLOAT8_E4M3FN), and 296 (FLOAT4_E2M1), for scenarios where shared experts and MoE experts use different quantization configurations. In the pre-quantized pass-through scenario (dispatchQuantMode is 0), only DT_UNDEFINED or the same value as dispatchQuantOutDtype can be used.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combineQuantMode</td>
   <td>Optional attribute</td>
   <td>Quantization mode during combine communication. 0 indicates non-quantization, 3 indicates the MXFP float8_e5m2 type, and 4 indicates the MXFP float8_e4m3 type. The default value is 0.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>Optional attribute</td>
   <td>Reserved parameter. Keep the default value. The default value is "".</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>numMaxTokensPerRank</td>
   <td>Optional attribute</td>
   <td>Number of tokens on each card. When numTokens differs across ranks, this is the maximum numTokens size. The default value is 0.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>activation</td>
   <td>Optional attribute</td>
   <td>Activation function type. The default value is "swiglu". Optional values are "swiglu", "swiglustep", "swigluoai", "situglu", and "gelu". Among them, "gelu" is supported only on <term>Ascend 950PR&950DT products</term>. For the values supported by each product, see the Constraints section.</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>activation_params</td>
   <td>Optional attribute</td>
   <td>Activation function parameter list. The default value is []. The parameter order and count are determined by activation: "swiglu" and "swiglustep" support [] or [clamp]; "swigluoai" supports [] or [clamp, alpha, beta]; "situglu" supports [], [beta], or [beta, linear_beta]; "gelu" does not support configuring a clamp value. A list of length 1 whose element is the maximum float value must be passed in (used only as a placeholder and not involved in the clamp computation), and an empty list is not supported. When an empty list is used, clamp defaults to the maximum float value, alpha defaults to 1.702, and the beta of swigluoai defaults to 1.0. clamp must be ≥ 0 and cannot be NaN; alpha and the beta of swigluoai must be finite values; the beta actually used by situglu and the explicitly configured linear_beta must both be finite values greater than 0. For differences in support for the "situglu" parameter list across products, see the Constraints section.</td>
   <td>LIST_FLOAT</td>
   <td></td>
  </tr>
  <tr>
   <td>activationOutDtype</td>
   <td>Optional attribute</td>
   <td>Data type of the activation function output. The default value is DT_UNDEFINED.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight1</td>
   <td>Optional attribute</td>
   <td>Whether weight1 is transposed. The default value is false.</td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight2</td>
   <td>Optional attribute</td>
   <td>Whether weight2 is transposed. The default value is false.</td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>weight1Interleave</td>
   <td>Optional attribute</td>
   <td>Interleaving parameter of weight1. Reserved parameter. The default value is 0.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>topoType</td>
   <td>Optional attribute</td>
   <td>Communication topology type, automatically derived from the communicator context. 0 indicates the MTE topology, 1 indicates the URMA cross-supernode topology, and 2 indicates the hybrid topology with intra-domain URMA and inter-domain RoCE. The default value is 0.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>rankNumPerServer</td>
   <td>Optional attribute</td>
   <td>Number of ranks in each communication domain, evenly and consecutively divided by the rank sequence numbers within the EP communication domain, not inferred by physical server. The default value is 2. This attribute does not take effect under some communication topologies on some products. See the Constraints section for details.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>topkWeightsType</td>
   <td>Optional attribute</td>
   <td>Switch for sending topkWeights in advance. 0 indicates disabled, and 1 indicates enabled (topkWeights is sent to the target rank in advance together with the token data in the dispatch stage, reducing the communication volume in the combine stage). The default value is 0.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combineCommMode</td>
   <td>Optional attribute</td>
   <td>Switch for communication deduplication in the combine stage. 0 indicates disabled, and 1 indicates enabled (for the multiple experts hit by the same token on the local card, the combine results are first merged into one row on the local card and then sent back to the source card, reducing the communication volume in the combine stage from the number of rows in the group to 1). The value supports 0, 1, or 2: 1 indicates intra-card deduplication under the MTE topology, and 2 indicates intra-server reduction and inter-domain secondary return under the Layered topology (topoType = 1 or 2). The two are routed by topology and cannot be interchanged. After <term>Atlas A3 products</term> enables this function, the operator still performs capability validation and benefit gating internally (infeasible configurations report an error or fall back automatically, and low-benefit shapes fall back silently; see the Constraints section); for <term>Ascend 950PR&950DT products</term>, whether to enable it is entirely decided by the caller based on the shape, and the operator does not make a benefit decision internally. Enabling it changes the communication record layout, and all ranks must be configured with the same value. The default value is 0. For the support scope of this switch across products and communication topologies, see the Constraints section.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>y</td>
   <td>Output</td>
   <td>Computation output result. The shape is (BS, H), consistent with the shape of x. For the data type, see the Constraints section.</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNums</td>
   <td>Output</td>
   <td>Number of tokens actually received by each expert on the local card.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## Constraints

- **Reserved and non-public parameter description**:
  - Some parameters and data types in the parameter table are not yet publicly available and are reserved or used for internal implementation. The introduction to the interface parameters and their constraints are described in detail in the interface document [MegaMoE Operator Interface Document](docs/torchapi_mega_moe_en.md).

- **Parameter consistency constraints**:
  - The values of parameters such as `moeExpertNum`, `maxRecvTokenNum`, `dispatchQuantMode`, `dispatchQuantOutDtype`, `numMaxTokensPerRank`, `topkWeightsType`, and `combineCommMode` used during operator invocation must be consistent across all cards and must also be consistent across different layers in the network. Among them, `topkWeightsType`, `combineCommMode`, and `dispatchQuantOutDtype` determine the layout of communication records and the communication buffer size; inconsistent values across cards cause read/write misalignment.
  - When <term>Atlas A3 products</term> enables communication deduplication in the combine stage (`combineCommMode` = 1), the `BS` of all cards must be exactly the same; inconsistent `BS` (even if all values are not less than 1024) causes the operator to hang or produce misaligned results due to inconsistent communication chunk rounds or tail-chunk layout strides across cards. Deployments where the `BS` values differ across cards must align them at the upper layer before invoking the operator.

- **Communicator constraints**:
  - The values of the `epWorldSize` and `cclBufferSize` parameters must be consistent across all cards.
  - The driver versions of all nodes in the communicator must be the same.
  - <term>Ascend 950PR&950DT products</term>: the Torch interface automatically computes and requests `cclBufferSize` through `get_symm_buffer_for_mega_moe`, covering both the communication window and the intermediate computation data area of the operator, so users do not need to compute or set it themselves; the framework workspace retains only the system space and no longer carries service data. `numMaxTokensPerRank`, `intermediateHidden`, `dispatchQuantOutDtype`, and `combineCommMode` all affect the required capacity. When an insufficient value is set by directly calling the aclnn API, the operator reports an error directly in the tiling stage and provides the required minimum number of bytes, without silent truncation.

- **Networking constraints**:
  - <term>Atlas A2 products</term>: the multi-machine communicator requires switch-based networking; direct connection between two machines is not supported.
  - <term>Atlas A3 products</term>: the multi-machine communicator must be within one supernode; direct connection between two machines and cross-supernode networking are not supported.
  - <term>Ascend 950PR&950DT products</term>: when `topoType` = 0 or 1, only the UB Memory communication protocol is used; when `topoType` = 2 (hybrid topology of URMA and RoCE), the UB Memory communication protocol is used within UB domains, and RoCE is used between UB domains.

- **Parameter constraints**:
  - **<term>Atlas A2 products</term> and <term>Atlas A3 products</term>**:
    - Scenario matching matrix:

      | Scenario | x | weight1 | weight2 | weightScales1 | weightScales2 | bias1 | bias2 | y | dispatchQuantMode | dispatchQuantOutDtype |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | A16W16 | BF16 | BF16 | BF16 | – | – | – | – | BF16 | 0 | – |
      | A8W8-INT | BF16 | INT8 | INT8 | UINT64 | UINT64 | – | – | BF16 | 2 | 1 (INT8) |
      | A8W4-INT | BF16 | INT4(INT32) | INT4(INT32) | UINT64 | UINT64 | FP32 | FP32 | BF16 | 2 | 1 (INT8) |

    - `dispatchQuantMode` supports only 0 and 2. 0 corresponds to the A16W16 scenario (the operator does not perform dispatch quantization internally), and 2 corresponds to the A8W8-INT and A8W4-INT scenarios (INT8 pertoken quantization). The values of `dispatchQuantMode` and `dispatchQuantOutDtype` are uniquely determined by the data type of `weight1`; when the values do not match, the operator reports an error in the tiling stage: when `weight1` is INT4 or INT8, the values are 2 and 1 (INT8) respectively; when `weight1` is BF16 or FLOAT16, the values are 0 and the same data type as `weight1` respectively. When `dispatchQuantOutDtype` is passed as DT_UNDEFINED, this check is not performed. For the corresponding values in each scenario, see the preceding table.
    - `scales` is a reserved input and currently must be passed as a null pointer.
    - When `activation` is "situglu", `activation_params` supports [], [beta], or [beta, linear_beta]; when an empty list is used, `beta` defaults to 1.0 and `linear_beta` is not enabled.
    - **<term>Atlas A2 products</term>**: communication deduplication in the combine stage is not supported; `combineCommMode` must keep the default value 0.
    - **<term>Atlas A3 products</term>**: communication deduplication in the combine stage is supported only in the `A8W4-INT` scenario; the value of `combineCommMode` is 0 or 1, and the other scenarios must keep the default value 0 (when the value is 1, the operator reports an error in the tiling stage).
      - When `combineCommMode` = 1, actual enabling must satisfy both the capability constraints and the benefit gating: when `maxRecvTokenNum` does not satisfy the single-round receiving constraint (manually configured too small), the operator automatically falls back (without enabling deduplication) and prints a log describing the reason; when the communication buffer does not satisfy the minimum window requirement of deduplication, the operator reports an error with a suggested value; when the benefit gating is not met (low-benefit shapes such as small batches), the operator silently falls back and prints a decision log without reporting an error.
      - Capability constraints: `maxRecvTokenNum` ≥ min(`BS`, 1024) × `epWorldSize` × min(`topK`, `localMoeExpertNum`) (the value automatically calculated by the interface when the default value 0 is kept exactly satisfies this constraint; when it is not satisfied, automatic fallback is performed); the communication buffer must satisfy the minimum window requirement of the deduplication layout. When the buffer is applied for through `get_symm_buffer_for_mega_moe` (`combineCommMode` = 1), the capacity is automatically calculated according to this requirement; when a manually configured buffer does not satisfy it, the operator reports an error with a suggested value.
      - When deduplication is enabled, the `BS` of all cards must be exactly the same; inconsistent `BS` (even if all values are not less than 1024) causes the operator to hang or produce misaligned results due to inconsistent communication chunk rounds or tail-chunk layout strides across cards.
      - When `combineCommMode` = 1, the outputs of multiple experts hit by the same token on the same card are first accumulated with FP32 weighting on that card and merged into one row before being sent: compared with the disabled state, in addition to the cross-expert accumulation order, the weighting position and the number of intermediate roundings also change (disabled state: each expert output is quantized to BF16 first and then multiplied by the topk weight; enabled state: FP32 weighted accumulation throughout and a single rounding at the end). Therefore, the result may differ even when `num_topk` = 1; it is a normal difference that `y` is not bitwise identical to that of `combineCommMode` = 0, and the enabled state is usually closer to the reference value. Use `combineCommMode` = 0 when bitwise consistency with the disabled state is required.
      - Applicable scenarios: shapes with a large `topK`, a high proportion of the same token hitting multiple experts on the same card, and a performance bottleneck in communication benefit significantly after deduplication is enabled; low-benefit shapes such as small batches are automatically reverted by the benefit gating (deduplication is not enabled), so the caller does not need to evaluate each shape; for edge shapes near the gating boundary, it is recommended to run actual measurement before deciding.
    - The data type of `y` is the same as that of `x`.

  - **<term>Ascend 950PR&950DT products</term>**:
    - `activation` supports "swiglu", "swiglustep", "swigluoai", "situglu", and "gelu". For "situglu", `activation_params` must be explicitly specified, supporting only [beta] or [beta, linear_beta]; an empty list is not supported. For "gelu", `activation_params` must also be explicitly specified, supporting only a placeholder list of length 1 whose element is the maximum float value; an empty list is not supported, and configuring a clamp value is not supported either. For the parameter matching relationships of the other activations, see the parameter description.
    - `BS` is `x`.dim0 of the current invocation on this rank, with the supported range [1, +∞), and must not exceed `numMaxTokensPerRank` set when creating `sym_buffer`. The actual `BS` values of different ranks can differ, and the same `sym_buffer` can be reused for multiple invocations with different `BS` values.
    - `numMaxTokensPerRank` has a value range of [1, UINT32_MAX] and must be configured consistently across all ranks. It is recommended to set it to the maximum per-card `BS` that may occur on all ranks during `sym_buffer` reuse. A larger value means more communication memory requested internally.
    - `H` (`x`.dim1) range [1024, 8192], and all quantization scenarios and communication topologies require `H` to be a multiple of 64 (consistent criteria for the MTE topology with `topoType` = 0 and the Layered topology with `topoType` = 1 or 2). The A4W4-FP scenario supports only `weight1` in FRACTAL_NZ format.
    - `topK` (`topkIds`.dim1) supports [1, 32].
    - `localMoeExpertNum` (= `moeExpertNum` / `epWorldSize`) range [1, 1024]; the number of shared experts `sharedExpertNumPerRank` is separately constrained to [0, 4] and is not counted in this upper limit.
    - `intermediateHidden` is half of the Linear1 output width of `weight1`, with range [256, 4096] and 128 alignment (that is, the output width is 2 × `intermediateHidden`, with range [512, 8192] and 256 alignment); the value range is independent of `activation`. Gated activations ("swiglu", "swiglustep", "swigluoai", "situglu") split this width into halves for gate and up, and the activation output width is `intermediateHidden`; "gelu" is an element-wise activation without splitting, and the activation output width is 2 × `intermediateHidden`.
    - `epWorldSize` range [2, 1024].
    - `moeExpertNum` range [`epWorldSize`, 2048], and `moeExpertNum` % `epWorldSize` == 0.
    - `maxRecvTokenNum` range [0, `numMaxTokensPerRank` × `epWorldSize` × min(`topK`, `localMoeExpertNum`)]. It is recommended to keep the default value 0 so that the interface automatically computes the receive capacity. The actually effective number of received rows (the automatically computed result when the default value 0 is used) must not exceed UINT32_MAX; when it is exceeded, the operator reports an error in the tiling stage, without silent truncation.
    - `dispatchQuantOutDtype` supports only 23 (FLOAT8_E5M2), 24 (FLOAT8_E4M3FN), or 296 (FLOAT4_E2M1).
    - `dispatchQuantMode` is 0 or 4. 4 indicates that the operator performs MX group-wise quantization on BF16 `x` internally (group size = 32, with the quantization scale factor type FLOAT8_E8M0), corresponding to the A8W8-FP, A8W4-FP, and A4W4-FP scenarios. 0 indicates pre-quantized pass-through, that is, `x` is already quantized data and the operator does not quantize it again internally; the scale factors are given by the `scales` input parameter.
    - Constraints of pre-quantized pass-through (`dispatchQuantMode` = 0): the data type of `x` must be consistent with the logical type specified by `dispatchQuantOutDtype`; `scales` is required; only `topoType` = 0 (MTE topology) is supported; `sharedExpertQuantOutDtype` can only be DT_UNDEFINED or the same as `dispatchQuantOutDtype`.
    - `combineQuantMode` is 0, 3, or 4. 0 indicates non-quantization, 3 indicates the MXFP float8_e5m2 type, and 4 indicates the MXFP float8_e4m3 type.
    - `commAlg` is a reserved parameter; keep the default empty string "".
    - The data type of `y` is fixed to BF16.
    - Under gated activation, the Linear1 output dimension of `weight1` equals 2 × `intermediateHidden` and the input dimension of `weight2` equals `intermediateHidden`; under "gelu", the Linear1 output dimension of `weight1` equals 2 × `intermediateHidden` and the input dimension of `weight2` equals 2 × `intermediateHidden`. If not satisfied, the operator reports an error in the tiling stage.
    - `localMoeExpertNum` = `moeExpertNum` / `epWorldSize`; `sharedExpertNumPerRank` = `sharedWeight1`.dim0 (0 when shared experts are not enabled); `expertPerRank` = `sharedExpertNumPerRank` + `localMoeExpertNum`.
    - `sharedExpertNumPerRank` range [0, 4].
    - `topoType` is automatically derived from the communicator context. 0 indicates the MTE topology, 1 indicates the URMA cross-supernode topology, and 2 indicates the hybrid topology with intra-domain URMA and inter-domain RoCE. 1 and 2 are collectively called the Layered topology, and the constraints for the URMA topology in this section apply to both.
    - The hybrid topology of URMA and RoCE (`topoType` = 2) requires all UB domains to have equal width and the group ranks within a domain to be consecutive. `rankNumPerServer` indicates the number of ranks in each UB domain; the UB Memory channel is used within a domain and RoCE is used between domains, which is determined by querying the topology information from the communicator context, not inferred by the number of physical hosts.
    - Under the Layered topology (`topoType` = 1 or 2), the value range of `rankNumPerServer` is [1, `epWorldSize`] and it must be divisible by `epWorldSize`; if not satisfied, the operator reports an error in the tiling stage.
    - Under the MTE topology (`topoType` = 0), `rankNumPerServer` does not take effect, and the operator processes it based on `epWorldSize`.
    - Under the Layered topology (`topoType` = 1 or 2), `activation` supports only "swiglu", "situglu", and "gelu", and does not support "swiglustep" and "swigluoai".
    - `topkWeightsType` is 0 or 1. 0 indicates that sending topkWeights in advance is disabled, and 1 indicates that it is enabled.
    - `combineCommMode` is 0, 1, or 2. 0 indicates that communication deduplication in the combine stage is disabled, 1 indicates that intra-card deduplication under the MTE topology is enabled, and 2 indicates that intra-server reduction and inter-domain secondary return under the Layered topology are enabled. The two enabling modes are routed by topology: the MTE topology accepts only 0 or 1, and the Layered topology accepts only 0 or 2; misconfiguration always causes an error in the tiling stage. Whether to enable it is entirely decided by the caller; the operator does not make a benefit decision internally. All ranks must be configured with the same value; otherwise, inconsistent communication record layouts across cards cause read/write misalignment. Applicable scenarios: shapes with a large `topK`, a high proportion of the same token hitting multiple experts on the same card, and a performance bottleneck in communication benefit significantly after enabling. Shapes with many experts and a bottleneck in computation may gain no benefit or even slow down; it is recommended to measure with actual shapes before deciding.
    - Capability constraints when `combineCommMode` = 1 (if not satisfied, the operator directly reports an error in the tiling stage and prompts to change back to 0, without silent fallback): only `topoType` = 0 (MTE topology) is supported; `combineQuantMode` must be 0 (no quantization in combine); pre-quantized `x` is not supported when `topkWeightsType` = 0; `numMaxTokensPerRank` must satisfy `numMaxTokensPerRank` ≤ 128 × min(512, ⌊4608 / `topK`⌋) (for example, the upper limit is 18432 when `topK` = 32, and 65536 when `topK` ≤ 9).
    - When `combineCommMode` = 1, the outputs of multiple experts for the same token on the same card are accumulated in FP32 on that card and then converted to BF16 once before being sent. The accumulation order and the number of intermediate rounding operations differ from the disabled state, so `y` is not bitwise identical to the result of `combineCommMode` = 0; the difference is a normal error caused by the different BF16 accumulation order. Use `combineCommMode` = 0 when bitwise consistency with the disabled state is required.
    - Capability constraints when `combineCommMode` = 2 (if not satisfied, the operator directly reports an error in the tiling stage, without silent fallback): only the Layered topology (`topoType` = 1 or 2) is supported; `combineQuantMode` must be 0 (no quantization in combine). In this mode, the contributions of each rank within the same domain are first sent to the relay rank of the domain for reduction, and then the aggregated results are returned by the relay with the same number across domains. The accumulation path differs from `combineCommMode` = 0, so `y` is not bitwise identical to that in the disabled state.
    - `weightScales1` and `weightScales2` are required inputs, and their data type must be FLOAT8_E8M0.
    - The data types of `weight1` and `weight2` must be the same, and only FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT4_E2M1 are supported. Here, "the same" means consistency within the same expert group; MoE experts and shared experts can adopt different weight types and quantization types.
    - When the quantization configurations of MoE experts and shared experts differ, only `topoType` = 0 (MTE topology) is supported.
    - The data type of `topkWeights` supports only BF16 or FP32.
    - In the current version, `xActiveMask` does not support non-null input; a null pointer must be passed in.
    - `scales` is required only in the pre-quantized pass-through scenario (`dispatchQuantMode` = 0 and `dispatchQuantOutDtype` is FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT4_E2M1), with shape (`BS`, CeilDiv(`H`, 32)); the data type must be FLOAT8_E8M0. In other scenarios, a null pointer must be passed in.

  - **MXFP quantization scenario constraints**:
      - `weight1` shape is (`localMoeExpertNum`, 2 × `intermediateHidden`, `H`), and `weight2` shape is (`localMoeExpertNum`, `H`, `intermediateHidden`).
      - `weightScales1` shape is (`localMoeExpertNum`, 2 × `intermediateHidden`, CeilDiv(`H`, 64), 2).
      - `weightScales2` shape is (`localMoeExpertNum`, `H`, CeilDiv(`intermediateHidden`, 64), 2).
      - `sharedWeight1` shape is (`sharedExpertNumPerRank`, 2 × `intermediateHidden`, `H`), and `sharedWeight2` shape is (`sharedExpertNumPerRank`, `H`, `intermediateHidden`).
      - `sharedWeightScales1` shape is (`sharedExpertNumPerRank`, 2 × `intermediateHidden`, CeilDiv(`H`, 64), 2), and `sharedWeightScales2` shape is (`sharedExpertNumPerRank`, `H`, CeilDiv(`intermediateHidden`, 64), 2).
      - dim3 of `weightScales1` and dim3 of `weightScales2` must equal 2.
      - In the A8W4-FP scenario, `weight1` of the FLOAT4_E2M1 type must use the FORMAT_FRACTAL_NZ_C0_32 format; in the A4W4-FP scenario, `weight1` of the FLOAT4_E2M1 type must use the FRACTAL_NZ format. In both scenarios, `weight2` must use the FORMAT_FRACTAL_NZ_C0_32 format. When shared experts are enabled, the formats of `sharedWeight1` and `sharedWeight2` are determined by the quantization type of the shared experts themselves (W8 uses ND or FRACTAL_NZ, and W4 uses FRACTAL_NZ or FORMAT_FRACTAL_NZ_C0_32), which can differ from the formats of `weight1` and `weight2`.
      - In the A8W8-FP scenario, `weight1` and `weight2` must be both FLOAT8_E5M2 or both FLOAT8_E4M3FN. In the A8W4-FP and A4W4-FP scenarios, the weights of both layers are FLOAT4_E2M1.
      - The weights and their scales support two TensorList layouts. For `weight1`, `weight2`, `weightScales1`, and `weightScales2`, in the per-expert layout, the lengths of the four TensorLists are all `localMoeExpertNum`; the tensors in the weight lists are two-dimensional, and the tensors in the weight scale lists are three-dimensional. In the stacked layout, the lengths of the four TensorLists are all 1; the weight tensors are three-dimensional, the weight scale tensors are four-dimensional, and dim0 of each tensor is `localMoeExpertNum`. The four inputs must adopt the same layout; when shared experts are enabled, the corresponding four shared expert inputs must also adopt the same layout as the MoE experts.

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch interface invocation | - | Invoke the mega_moe operator through the [mega_moe](docs/torchapi_mega_moe_en.md) PyTorch interface. |

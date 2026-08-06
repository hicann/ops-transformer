# MhcPre

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。

- **计算公式**：

其中，xFlat表示将x的最后两维n和D视作长度为nD的向量，gammaFlat表示将gamma视作长度为nD的向量，@表示矩阵乘法，$\odot$表示逐元素乘法。

$$
\begin{aligned}
invRms &= \left(mean(xFlat^{2}) + normEps\right)^{-\frac{1}{2}}\\
xGamma &= \begin{cases}
    xFlat \odot gammaFlat, & gamma \ne null \\
    xFlat, & gamma = null
\end{cases}\\
hMix &= xGamma @ phi^{T}\\
w &= hMix \odot invRms\\
[pPre, pPost, pRes] &= \begin{cases}
    split(w, [n, n, n^{2}]), & alpha.shape=[3] \\
    [split(w, [n, n]), 0], & alpha.shape=[2]
\end{cases}\\
hPre &= \sigma(pPre \odot alpha0 + bias0) + hcEps\\
hPost &= 2\sigma(pPost \odot alpha1 + bias1)\\
hRes &= \begin{cases}
    pRes \odot alpha2 + bias2, & alpha.shape=[3] \\
    0, & alpha.shape=[2]
\end{cases}\\
hIn_{d} &= \sum_{i=0}^{n-1} hPre_{i} x_{i,d}
\end{aligned}
$$

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>待计算数据，表示网络中mHC层的输入数据，对应公式中的x。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>mHC的参数矩阵，对应公式中的phi。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>mHC的缩放参数，对应公式中的alpha。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>mHC的bias参数，对应公式中的bias。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gamma</td>
        <td>可选输入</td>
        <td>表示进行RmsNorm计算的缩放因子，对应公式中的gamma。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out_flag</td>
        <td>可选输入</td>
        <td>表示是否输出公式中的invRms、hMix和hPre，默认为0表示不输出，为1表示全部输出。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>norm_eps</td>
        <td>可选输入</td>
        <td>RmsNorm的防除零参数，对应公式中的normEps。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>可选输入</td>
        <td>h_pre的sigmoid后的eps参数，对应公式中的hcEps。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>op_impl_mode</td>
        <td>可选输入</td>
        <td>指定MhcPre算子的计算模式，0表示Cube使用FP32模式计算，1表示Cube使用HF32模式计算，默认值为0。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h_in</td>
        <td>输出</td>
        <td>输出的h_in作为Attention/MLP层的输入，对应公式中的hIn。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>输出</td>
        <td>输出的mHC的h_post变换矩阵，对应公式中的hPost。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_res</td>
        <td>输出</td>
        <td>输出的mHC的h_res变换矩阵（未做sinkhorn变换），对应公式中的hRes。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inv_rms</td>
        <td>可选输出</td>
        <td>RmsNorm计算得到的1/r，对应公式中的invRms。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_mix</td>
        <td>可选输出</td>
        <td>xGamma与phi矩阵乘的结果，对应公式中的hMix。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_pre</td>
        <td>可选输出</td>
        <td>做完sigmoid计算之后的h_pre矩阵，对应公式中的hPre。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- n目前支持4、6、8。
- D支持1~16384范围以内，需满足D为16对齐。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_pre](examples/test_aclnn_mhc_pre.cpp) | 通过[aclnnMhcPre](docs/aclnnMhcPre.md)接口方式调用MhcPre算子。 |
| aclnn调用 | [test_aclnn_mhc_pre_v2](examples/test_aclnn_mhc_pre_v2.cpp) | 通过[aclnnMhcPreV2](docs/aclnnMhcPreV2.md)接口方式调用MhcPre算子，并通过opImplMode选择Cube的FP32或HF32计算模式。 |

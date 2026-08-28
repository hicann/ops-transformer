# RotaryPositionEmbedding3D

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

**算子功能**：推理场景，对视频特征序列按帧/高/宽三轴独立编码位置信息，实现 Wan2.2 风格的 3D 空间 Rotary Position Embedding。

与标准 1D RoPE 的核心区别：
- **不等长频带划分**：时间轴 d0 = D/2，高度/宽度轴各 D/4（2:1:1 比例）
- **三维位置解码**：l -> (t,i,j) 将一维序列位置解码为视频网格坐标
- **三轴独立旋转**：T/H/W 三个频带分别用各自坐标乘以独立频率表，再应用 half-mode 旋转

**计算公式**：

频带划分：DT = D/2, DH = D/4, DW = D/4

频率衰减率：rT = b^(-2/DT), rH = b^(-2/DH), rW = b^(-2/DW), b = 10000

Theta 表（等比序列）：thetaT[k] = rT^k, thetaH[k] = rH^k, thetaW[k] = rW^k

位置解码（S -> T x H x W）：
t = l / (H * W), i = (l mod H*W) / W, j = l mod W

Half-mode 旋转：
yl[k] = xl[k] * cos(a[k]) - xr[k] * sin(a[k])
yr[k] = xl[k] * sin(a[k]) + xr[k] * cos(a[k])

其中角度 a 按频带分别构造：
- T-band: a[k] = t * thetaT[k]
- H-band: a[k] = i * thetaH[k]
- W-band: a[k] = j * thetaW[k]

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|-------------|------|---------|---------|
| x | 输入 | 输入 tensor，shape 为 [B, S, D] | FLOAT16, FLOAT, BFLOAT16 | ND |
| cos | 输入 | 预计算 cos/sin 表，shape 为 [B, S, D]，左半 cos、右半 sin | FLOAT16, FLOAT, BFLOAT16 | ND |
| y | 输出 | 旋转结果 tensor，shape 为 [B, S, D] | FLOAT16, FLOAT, BFLOAT16 | ND |
| t_output_dim | 可选属性 | 视频时间轴帧数 T。不传或为 0 时自动分解。 | INT64 | - |
| h_output_dim | 可选属性 | 视频高度 token 数 H。不传或为 0 时自动分解。 | INT64 | - |
| w_output_dim | 可选属性 | 视频宽度 token 数 W。不传或为 0 时自动分解。 | INT64 | - |

## 约束说明

- 输入 x 和 cos 的 shape 必须相同。
- D 必须为偶数，且 D >= 8。
- 序列长度 S 建议能被因子分解为 T x H x W 以充分利用 3D 位置编码；若无法分解，退化为 1D（T=1, H=1, W=S）。
- 默认通过 FactorVideoDims 自动将 S 分解为视频维度（T, H, W）；对视频模型（如 Wan2.2）建议显式传入 t_output_dim / h_output_dim / w_output_dim（需满足 T x H x W == S），保证位置解码正确。
- 该接口仅支持 ND 格式，不支持其他数据格式。
- 该接口暂不支持入图模式。

## 调用说明
| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_rotary_position_embedding3d](examples/test_aclnn_rotary_position_embedding3d.cpp) | 通过 aclnnRotaryPositionEmbedding3d 接口方式调用 RotaryPositionEmbedding3D 算子。host 端预计算 cos/sin 表传入第二输入，kernel 内完成按帧/高/宽三轴频带独立旋转。 |

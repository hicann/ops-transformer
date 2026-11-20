#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""QkvRmsNormRopeCache 多通路 golden(TestSpec 范式)。

支持的通路:
| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅  | op_kernel/arch35/ 有 arch35 regbase 实现 |
| geir   | ✅  | op_graph/qkv_rms_norm_rope_cache_proto.h 已注册 REG_OP(QkvRmsNormRopeCache) |
| aclnn  | ✅  | op_host/op_api/aclnn_qkv_rms_norm_rope_cache.{h,cpp} |
| e2e    | ✅  | libtorch_npu.so 命中 aclnnQkvRmsNormRopeCache 与 npu_qkv_rms_norm_rope_cache |

计算语义按【算子契约】(README + op_graph proto + aclnn doc)写,不照抄内核的步骤划分与
内部布局:内核按 (token, head) 行循环、把 cache 写成 [BlockNum, N*D1, BlockSize, D0] 的
PA_NZ 分片;本 golden 只按"输入张量语义 + 输出张量语义"逐步拼接 —— split -> rmsnorm ->
rope(half-and-half) -> quant -> scatter,布局从 cache 的 shape 反推。
"""

import numpy as np
import torch

__spec__ = {
    "qkv_rms_norm_rope_cache": "QkvRmsNormRopeCacheKernelSpec",
    "aclnnQkvRmsNormRopeCache": "QkvRmsNormRopeCacheAclnnSpec",
    "torch.ops.npu.npu_qkv_rms_norm_rope_cache": "QkvRmsNormRopeCacheTorchSpec",
}

_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    # 量化输出走 quant 判据(绝对误差 <= 1 LSB),不参与三方
    "int8": {"standard": "quant"},
}


def _as_int_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().strip("[]()")
        return [int(float(x)) for x in v.split(",") if x.strip() != ""]
    return [int(x) for x in v]


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _cache_shape(block_num, head_num, head_dim, block_size, elem_bytes):
    d0 = 32 // elem_bytes
    return (int(block_num), int(head_num * head_dim // d0), int(block_size), int(d0))


def _rms_norm(x, gamma, eps):
    """x: [..., D] 沿最后一维归一化。"""
    var = torch.mean(x * x, dim=-1, keepdim=True)
    return x / torch.sqrt(var + eps) * gamma


def _rope_half_and_half(x, cos, sin):
    """half-and-half:out = x*cos + cat(-x2, x1)*sin,沿最后一维。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def _cast_like(t, ref):
    """标量原样返回;张量对齐到 ref.dtype(fp32→fp64 是精确转换,不丢信息)。"""
    return t.to(ref.dtype) if hasattr(t, "to") else t


def _quant(x, scale, offset):
    """量化:x / scale + offset 后 round + clamp 到 int8 值域(x 为 Ref 精度 fp64)。

    scale/offset 显式对齐到 x.dtype:torch 的混合 dtype 提升本也会给出 fp64,
    这里写死是为了防"某一路忘了抬"被静默降回 fp32,而不是依赖隐式规则。
    """
    if scale is not None:
        x = x / _cast_like(scale, x)
    if offset is not None:
        x = x + _cast_like(offset, x)
    if scale is not None:
        x = torch.round(x).clamp(-128, 127)
    return x


def _scatter_pa_nz(data_rows, cache, index, head_dim, num_head):
    """data_rows: [T, num_head*head_dim];cache: [BlockNum, num_head*head_dim/D0, BlockSize, D0]。

    cache 的第 2 维是「head × D 内的 D0 分片」,即 [BlockNum][head*D1][BlockSize][D0],
    与 A2 契约的 [BlockNum, N*D/D0, BlockSize, D0] 同一件事(只是把第 2 维拆成 head 与分片两层)。
    布局从 cache 的 shape 反推,不照抄内核的地址算术。
    """
    block_num, hd, block_size, d0 = cache.shape
    d1 = head_dim // d0
    idx = index.reshape(-1).to(torch.int64)
    valid = idx >= 0
    page_id = torch.where(valid, idx // block_size, torch.zeros_like(idx))
    in_page = torch.where(valid, idx % block_size, torch.zeros_like(idx))
    slot = (page_id * block_size + in_page)[
        valid
    ]  # 命中 [BlockNum*BlockSize] 展平后的行号
    # 把 [BN][head*D1][BS][D0] 换成 [BN][BS][head*D1][D0] 再展平成 [BN*BS] 行,才谈得上"按槽位整行覆盖"
    o = cache.permute(0, 2, 1, 3).contiguous().reshape(block_num * block_size, hd * d0)
    src = data_rows.reshape(-1, hd * d0).to(o.dtype)
    o[slot] = src[valid]
    return o.reshape(block_num, block_size, hd, d0).permute(0, 2, 1, 3).contiguous()


_COMPUTE_KWARGS = (
    "qkv_size",
    "head_nums",
    "epsilon",
    "cache_mode",
    "is_output_qkv",
    "output_dtypes",
)


# aclnn 头文件 / torch 接口的属性名与 def.cpp 常常不同名(GNSQ 实例同款坑)。
# 三套命名都收:kernel=def.cpp 蛇形、aclnn=头文件驼峰、torch=接口签名。
_ALIASES = {
    "qkv_size": ("qkv_size", "qkvSize", "qkv_size_list"),
    "head_nums": ("head_nums", "headNums"),
    "epsilon": ("epsilon",),
    "cache_mode": ("cache_mode", "cacheModeOptional", "cacheMode"),
    "is_output_qkv": ("is_output_qkv", "isOutputQkv"),
    # 无 output_dtypes:KernelSpec.golden 的输出 dtype 由原始入参推导
    # (见 _derive_out_dtypes),不再读取 TTK 下发的这项;aclnn / e2e 各自直读 kwargs。
}


def _pick(kwargs):
    """TTK 会附带 full_soc_version / testcase_name 等框架侧 kwargs,只取算子契约用到的。"""
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n in kwargs:
                out[key] = kwargs[n]
                break
    return out


def _compute(
    qkv,
    q_gamma,
    k_gamma,
    cos,
    sin,
    index,
    q_out,
    k_cache,
    v_cache,
    k_scale=None,
    v_scale=None,
    k_offset=None,
    v_offset=None,
    qkv_size=None,
    head_nums=None,
    epsilon=1e-6,
    cache_mode="PA_NZ",
    is_output_qkv=False,
    output_dtypes=None,
):
    """按算子契约计算 6 个输出,顺序 = def.cpp 的输出序。

    golden 必须在【最高可用精度】上算,只做精度向上兜底(CPU half 残缺),不向下砍档。
    """
    if qkv.dim() == 2:
        qkv = qkv.unsqueeze(0) if qkv.numel() == 0 else qkv
    # 统一抬到 fp64:参考值必须是"真值"。
    #   1) TTK 的 Promote(ttk/utilities/dtypes.py:75-85 `DTYPE_PROMOTE_MAP`)对 fp16/bf16
    #      只抬到 fp32,对 **fp32 输入则规定抬到 fp64** —— 本 golden 原先把目标停在 fp32,
    #      等于比框架自己的约定还低一档;fp32 输入的用例根本没被抬过。
    #   2) 抬升是**精确转换**(fp16/bf16 经 Promote 已是 fp32,而 fp32/fp16 的每个值在
    #      fp64 里都可精确表示),不改变任何一个输入值,只抬高中间计算精度。
    #   3) 算子契约没有规定 fp32 中间量。fp32 中间量在 RoPE 的灾难性相消点上会把误差
    #      放大到 1e-1 量级(实测 L0_133:归一化输出偏 1 个 fp32-ULP → 输出跳 128 个
    #      fp16-ULP = 7.1%),此时用 golden 当分母判"实现是否正确"= 判舍入运气。
    #      fp64 下同样的 1 个末位噪声只有 ~2e-16 相对量级,远小于 1 个输出 ULP。
    # 整型 qkv(契约上不出现)不参与提升,保持原 dtype。
    qkv = qkv.to(torch.float64) if qkv.is_floating_point() else qkv
    # gamma/cos/sin 一律跟随 qkv 提升后的 dtype(现为 fp64):它们只承载输入值
    # (fp16/bf16→fp32→fp64 或 fp32→fp64 都是精确转换),绝不能在这里落回 fp32,
    # 否则后面的乘加会重新引入 fp32 舍入 —— 本算子 RoPE 相消点对这点极敏感。
    q_gamma = q_gamma.to(qkv.dtype)
    k_gamma = k_gamma.to(qkv.dtype)
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    total, n_all_dim = qkv.shape

    b_size, s_size, n_all, d_dim = [int(v) for v in _as_int_list(qkv_size)]
    n_q, n_k, n_v = [int(v) for v in _as_int_list(head_nums)]
    eps = float(epsilon)
    assert total == b_size * s_size, "qkv dim0 must equal B*S"
    assert n_all_dim == n_all * d_dim, "qkv dim1 must equal N*D"
    assert n_all == n_q + n_k + n_v and n_k == n_v, (
        "head_nums must satisfy N = Nq+Nk+Nv, Nk = Nv"
    )

    x = qkv.reshape(total, n_all, d_dim)
    q = x[:, :n_q, :]
    k = x[:, n_q : n_q + n_k, :]
    v = x[:, n_q + n_k :, :]

    # cos/sin 是 [B*S, D],逐 token 广播到该 token 的所有 head。
    # .to(x.dtype) 是显式对齐(此处 x 已随 qkv 抬到 fp64)—— 不写的话
    # torch 的类型提升也会得到 fp64,但显式对齐能挡住后人把某一路提前降档。
    cos_r = cos.reshape(total, 1, d_dim).to(x.dtype)
    sin_r = sin.reshape(total, 1, d_dim).to(x.dtype)

    q_norm = _rope_half_and_half(_rms_norm(q, q_gamma, eps), cos_r, sin_r)
    k_norm = _rope_half_and_half(_rms_norm(k, k_gamma, eps), cos_r, sin_r)

    q_rows = q_norm.reshape(total, n_q * d_dim)
    k_rows = k_norm.reshape(total, n_k * d_dim)
    v_rows = v.reshape(total, n_v * d_dim)

    k_final = _scatter_pa_nz(
        _quant(k_norm, k_scale, k_offset).reshape(total, n_k * d_dim),
        k_cache,
        index,
        d_dim,
        n_k,
    )
    v_final = _scatter_pa_nz(
        _quant(v, v_scale, v_offset).reshape(total, n_v * d_dim),
        v_cache,
        index,
        d_dim,
        n_v,
    )

    # 全程保持提升后的 fp64(除 cache 输出由 _scatter_pa_nz 按 cache 自身 dtype 落地),
    # 向下 cast 到算子的输出 dtype 由各通路外壳负责 —— 那一步是**唯一**的降档点,
    # 且 fp64→fp16/bf16/int8 是一次性直接舍入,不存在 fp64→fp32→fp16 的二次舍入。
    return [q_rows, k_final, v_final, q_rows.clone(), k_rows, v_rows]


def _mask_unrequested_qkv(outs, is_output_qkv):
    """is_output_qkv=false 时,outputs 3-5 按 proto 契约不产出 → 置 None。

    依据 `op_graph/qkv_rms_norm_rope_cache_proto.h:60-66`(三路 before_quant 仅
    "if is_output_qkv is true ... is required")、`infershape.cpp:88-91`(false 时
    直接 return、不推导这三路 shape)、内核三处 `if (tilingData_->isOutputQkv)`
    才写这三路。TTK 的 None 语义 = 该输出不比较(golden None → SUPPRESSED+PASS,
    见 ttk/core_modules/common/comparison/comparison.py:109-110),不是 FAIL。

    ⚠️ 只在**返回 6 路**的 kernel/aclnn 外壳上做;TorchSpec 只返回 3 路且取的
    是 outs[0]/[4]/[5](逐 token 投影值,恒产出),不能在这里被置 None。
    """
    if _as_bool(is_output_qkv):
        return outs  # true:6 路全比较,算子真实能力不降级
    outs = list(outs)
    for i in (3, 4, 5):
        if i < len(outs):
            outs[i] = None
    return outs


def _np_of(t, name):
    """torch.Tensor -> numpy.ndarray,按算子输出 dtype。None(不比较的槽位)原样返回。

    bf16 必须走 ml_dtypes:torch 的 bf16 张量在本环境 `.numpy()` 不支持,
    而 `np.ndarray.astype('bfloat16')` 会静默回落到 'V2'(void 2 字节)。
    """
    if t is None:
        return None
    dt = _torch_dtype(str(name))
    if dt is torch.bfloat16:
        import ml_dtypes

        # 载体用 float64 而不是 float32:torch 的 bf16 张量在本环境 .numpy() 不支持,
        # 必须经浮点载体转;若载体是 fp32,就变成 fp64→fp32→bf16 的**二次舍入**
        # (真值恰好落在 bf16 中点上时,与直接舍入差 1 个 bf16-ULP)。fp16/bf16/int8
        # 张量经 fp64 载体是无损往返,不受影响。
        return t.to(torch.float64).numpy().astype(ml_dtypes.bfloat16)
    return t.to(dt).numpy()


def _cast_outs(outs, kwargs, derived=None):
    """aclnn / e2e 通路:按 CSV 的 output_dtypes 把结果 cast 回算子输出 dtype。

    必须经 torch 转 —— np.ndarray.astype('bfloat16') 会静默回落到 'V2'(void 2 字节),
    让 bf16 输出在 TTK 侧的元素数差 2 倍,表现为莫名其妙的 shape mismatch。
    None(is_output_qkv=false 时不产出的槽位)原样透传,不参与 cast。

    `derived`:TTK 不传 `output_dtypes` 时(实测 aclnn 通路 kwargs 只有
    device/scalar_dtypes/short_soc_version/tensor_dtypes/tensor_formats/testcase_name/use_numpy)
    由调用方按输入推导的输出 dtype 兜底;不兜底的话 `_compute` 的 fp64 结果会原样返回,
    与算子按声明 dtype(如 bf16)的输出在 isclose 默认阈值下大面积判 fail。
    None 槽位仍然原样透传。
    """
    od = kwargs.get("output_dtypes") or derived or []
    od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
    return [
        o.to(_torch_dtype(od[i])) if (o is not None and i < len(od)) else o
        for i, o in enumerate(outs)
    ]


def _torch_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int8": torch.int8,
        "int64": torch.int64,
        "float64": torch.float64,
    }.get(str(name), torch.float32)


def _dtype_name(t, default="float32"):
    return str(t.dtype).replace("torch.", "") if hasattr(t, "dtype") else default


def _derive_out_dtypes(qkv, k_cache=None, v_cache=None):
    """缺省 output_dtypes 时按输入推导 6 路输出的 dtype(def.cpp 的输出序)。

    e2e 通路 TTK 不传 output_dtypes;不推导的话竞品/参考实现会停在 fp32,
    与算子实际输出 dtype 不对等(实测 Golden Dtypes 首项为 float32)。
    输出 0/3/4/5 与输入 qkv 同 dtype;1/2 与各自的 cache 输入同 dtype。
    """
    q = _dtype_name(qkv)
    return [q, _dtype_name(k_cache, q), _dtype_name(v_cache, q), q, q, q]


def _fill_optional(tensors, dtypes):
    """把 TTK 传进来的 None / 空张量规整成 None,便于按契约分支。"""
    out = []
    for t, d in zip(tensors, dtypes):
        if t is None or (hasattr(t, "numel") and t.numel() == 0):
            out.append(None)
        else:
            out.append(t)
    return out


class _Compose:
    """竞品标杆:用 torch 原生算子拼等价语义,在远端 A100 上执行。

    与 _compute 保持相互独立:这里用 permute/cat 的显式展开而非同一组表达式,
    但**算法与 dtype 一致**(rms 用 mean+sqrt、half-and-half 旋转、round+clamp 到 int8),
    输出一律 cast 回 NPU 输出 dtype —— 竞品留在 fp32 会让 cross_check 的比值凭空爆表。
    """

    def __init__(self, **kwargs):
        self.qkv_size = _as_int_list(kwargs.get("qkv_size"))
        self.head_nums = _as_int_list(kwargs.get("head_nums"))
        self.epsilon = float(kwargs.get("epsilon", 1e-6))
        self.is_output_qkv = _as_bool(kwargs.get("is_output_qkv", False))
        self.output_dtypes = kwargs.get("output_dtypes")
        # 逐实例缓存编译结果:TTK 每个用例新建一个实例(属名/输出 dtype 不同),
        # 放模块级会让不同属性的用例共用一个已编译图 -> 结果错。
        self._compiled = None

    def _impl(
        self,
        qkv,
        q_gamma,
        k_gamma,
        cos,
        sin,
        index,
        q_out,
        k_cache,
        v_cache,
        k_scale=None,
        v_scale=None,
        k_offset=None,
        v_offset=None,
    ):
        """独立于 _compute 的实现:同一算法(逐条对齐:mean+sqrt、half-and-half 旋转、
        round+clamp 量化、PA_NZ 散射),但表达式与数据组织不同 —— 用 split 切分、用切片赋值
        拼旋转项、用 index_put_ 一次性散射,便于两侧交叉验证(照抄实现会让两边同错)。"""
        f32 = torch.float32
        # 输出 dtype 必须在提升到 fp32 之前推导,否则会一律推导成 float32
        od = self.output_dtypes or _derive_out_dtypes(qkv, k_cache, v_cache)
        qkv = qkv.to(f32)
        q_gamma, k_gamma, cos, sin = (
            q_gamma.to(f32),
            k_gamma.to(f32),
            cos.to(f32),
            sin.to(f32),
        )
        total, n_all_dim = qkv.shape
        b_size, s_size, n_all, d_dim = [int(v) for v in self.qkv_size]
        n_q, n_k, n_v = [int(v) for v in self.head_nums]
        eps = float(self.epsilon)

        rows = qkv.reshape(total, n_all, d_dim)
        q, k, v = rows.split([n_q, n_k, n_v], dim=1)

        cos_r = cos.reshape(total, 1, d_dim)
        sin_r = sin.reshape(total, 1, d_dim)

        def _norm(t, g):
            # 必须与算子实现同算法:先 sqrt 再相除(不是乘 1/sqrt —— 那会多一次舍入,
            # 让竞品凭空更准,三方比值失真)
            return t / torch.sqrt(torch.mean(t * t, dim=-1, keepdim=True) + eps) * g

        def _rotate(t):
            half = t.shape[-1] // 2
            rot = torch.empty_like(t)
            rot[..., :half] = -t[..., half:]
            rot[..., half:] = t[..., :half]
            return t * cos_r + rot * sin_r

        def _quant(t, sc, off):
            y = t
            if sc is not None:
                y = y / sc
            if off is not None:
                y = y + off
            if sc is not None:
                y = torch.round(y).clamp(-128, 127)
            return y

        def _scatter_pa_nz(data_rows, cache):
            block_num, hd, block_size, d0 = cache.shape
            d1 = d_dim // d0
            n_head = data_rows.shape[-1] // d_dim
            idx = index.reshape(-1).to(torch.int64)
            good = idx >= 0
            page = (idx // block_size)[good]
            off = (idx % block_size)[good]
            cache_bs = (
                cache.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(block_num * block_size, hd * d0)
            )
            src = data_rows.reshape(-1, n_head * d1 * d0).to(cache_bs.dtype)[good]
            flat_row = page * block_size + off
            cols = torch.arange(cache_bs.shape[1], device=cache_bs.device).expand(
                flat_row.numel(), -1
            )
            cache_bs.index_put_((flat_row.unsqueeze(1), cols), src, accumulate=False)
            return (
                cache_bs.reshape(block_num, block_size, n_head, d1, d0)
                .permute(0, 2, 3, 1, 4)
                .reshape(block_num, hd, block_size, d0)
            )

        q_res = _rotate(_norm(q, q_gamma)).reshape(total, n_q * d_dim)
        k_pre = _rotate(_norm(k, k_gamma))
        v_pre = v
        k_res = _scatter_pa_nz(
            _quant(k_pre, k_scale, k_offset).reshape(total, n_k * d_dim), k_cache
        )
        v_res = _scatter_pa_nz(
            _quant(v_pre, v_scale, v_offset).reshape(total, n_v * d_dim), v_cache
        )

        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        outs = [
            q_res,
            k_res,
            v_res,
            q_res.clone(),
            k_pre.reshape(total, n_k * d_dim),
            v_pre.reshape(total, n_v * d_dim),
        ]
        # 三方输出槽位口径必须与 golden 一致:is_output_qkv=false 时 3-5 按契约
        # 不产出 → None(竞品也不比这三路)。保持长度 6——ttk 的 third_party 长度
        # 校验是 len(third_parties) < len(outputs) 即 COMPARE_FAILURE
        # (comparison.py:131-133),顶层 None 槽位是合法表达
        # (executor.py:729-732 / dispatcher.py:219-222)。
        outs = _mask_unrequested_qkv(outs, self.is_output_qkv)
        return [
            o.to(_torch_dtype(od[i])) if (o is not None and i < len(od)) else o
            for i, o in enumerate(outs)
        ]

    def __call__(self, *tensors, **kwargs):
        """性能/三方腿按竞品最优形态执行:torch.compile(dynamic=True) 融合。

        竞品是**小算子拼接**,不融合的话对方十几个 kernel 的启动开销会全算进去,
        G/N 系统性虚高 —— 看着达标、实际没对标(规范 verification.md §7.1)。
        dynamic=True 是必须的:本算子的用例 shape 跨度极大,默认静态形状会按
        每个新 shape 重编译,编译开销虽不计入 device_us(TTK 的 profiler schedule
        wait=1/warmup=1 已跳过前两轮),却会把单例墙钟拖到不可接受。

        编译失败或运行期异常都逐实例回退 eager —— 三方腿宁可慢,也不能静默缺腿。
        实测 A100 + torch 2.13:编译结果与 eager 逐位一致(max diff 0.0)。
        """
        args = list(tensors[:13])
        args += [None] * (13 - len(args))
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(*args)
        except Exception:
            self._compiled = self._impl
            return self._impl(*args)


def _to_npu_array(a):
    """NPU 侧(customize_inputs)用:原样保留数组与 dtype,只把"该输入缺省"统一成 None。

    与 _to_tensor 的关键区别:【不做任何 dtype 提升】。customize_inputs 的返回值会替换
    TTK 的运行时输入,提升上去就下不来了(内核会拿到"声明 bf16、实为 fp32"的缓冲)。
    """
    if a is None or not isinstance(a, np.ndarray):
        return a
    if a.dtype == object or a.size == 0:
        return None
    return a


# 用例名里声明「index 负向意图」的标记。生成器(gen_full_cases.make_case / make_cases.make_case)
# 在 index_neg=True 时追加到用例名尾部;运行期 CSV 只有用例名能带这个意图进来
# (index_neg 不是 CSV 的列,attributes 会一并传给算子、不能塞测试专有字段)。
NEG_MARK = "_negidx"


def _fill_index(index, total, block_num, block_size, seed, neg=False, as_numpy=False):
    """index 契约:[B*S],取值不可重复,值域 [-1, BlockNum*BlockSize);-1 表示跳过。

    取值本身由 numpy 生成;as_numpy 决定返回 numpy 还是 torch(NPU 侧要 numpy)。

    ⚠️ total > capacity 时(**capacity 装不下全部 token**)也必须无重复:
      · 槽位不够时,超出 capacity 的部分才置 -1(跳过);
      · 但**前 capacity 个仍须是 capacity 的一个排列**。

    neg=True(用例名带 NEG_MARK,即用例声明了 index 负向意图):
      再**故意**留一批 token 不写(-1),把"跳过 cache 写入"这条分支真正走到 ——
      否则本算子的测试里永远只写不跳(槽位恒够),该分支一次都测不到。
      同族 A2/A3 的 ATK golden(input_funcs)候选池含 -1,槽位充足时也会抽到,
      这里对齐该语义,但保持"取值不可重复"的契约(ATK 那版 random.sample 是可能重复的)。
    """
    rng = np.random.RandomState(seed)
    capacity = block_num * block_size
    # 必须写 cache 的 token 数:槽位够就全写,不够则超出部分只能跳过
    writable = min(total, capacity)
    if neg and writable > 1:
        # 至少留 1 个可写,否则整片跳过、反而测不到正常写入
        writable = int(rng.randint(1, writable))
    # 前 writable 个取 [0, capacity) 的一个排列(天然无重复),其余置 -1 跳过
    idx = np.full(total, -1, dtype=np.int64)
    idx[:writable] = rng.permutation(capacity)[:writable]
    return idx if as_numpy else torch.from_numpy(idx)


def _seed_of(kwargs):
    import zlib

    name = str(kwargs.get("testcase_name", "qkv_rms_norm_rope_cache"))
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


_TORCH_SAFE_DTYPES = frozenset(
    {
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.dtype(np.float16),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
        np.dtype(np.int64),
        np.dtype(np.int32),
        np.dtype(np.int16),
        np.dtype(np.int8),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.bool_),
    }
)


def get(seq, idx):
    """越界取 None(可选张量在不同通路的个数不同)。"""
    return seq[idx] if seq is not None and idx < len(seq) else None


def _numel(t):
    """元素个数;None(numpy 与 torch 通用的"缺省槽位")返回 0。"""
    if t is None:
        return 0
    if hasattr(t, "numel"):
        return int(t.numel())
    return int(t.size)


def _to_tensor(a):
    """None / object ndarray(TTK 用它表达"该可选输入缺省")保持 None,其余转 torch。"""
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a
    if not isinstance(a, np.ndarray):
        return a
    if a.dtype == object or a.size == 0:
        return None
    if a.dtype not in _TORCH_SAFE_DTYPES:
        # ml_dtypes 的 bfloat16 / float8 等,torch.from_numpy 不认;
        # bf16 -> fp32 是精确的向上提升,不影响精度判据
        a = a.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a))


def _prepare(tensors, *, for_npu=False, **kwargs):
    """把 index 重写成合法取值。返回新的 list;None 位置保持 None。

    只重建 index,其余张量【原对象原样返回】——它们本来就不需要改,
    多绕一趟 torch 只会平白引入 dtype 漂移(见 _to_npu_array)。

    for_npu=False(golden 计算):经 _to_tensor 转 torch,bf16 会抬成 fp32
        —— 无害,算完按输出 dtype cast 回去即可。
    for_npu=True(customize_inputs,NPU 侧):全程 numpy,不碰任何 dtype
        —— 其返回值会替换 TTK 的运行时输入,dtype 必须与声明一致。
    """
    t = (
        [_to_npu_array(a) for a in tensors]
        if for_npu
        else [_to_tensor(a) for a in tensors]
    )
    index = t[5]
    k_cache = t[7]
    # 空/缺省输入守卫:TTK 用 object ndarray 或空张量表达"该槽位缺省",_to_npu_array /
    # _to_tensor 会把它们规整成 None。此时既没有 index 可写、也读不出 cache 的
    # BlockNum/BlockSize,直接原样返回 —— 让算子自己拒绝成为可观察结果,
    # 而不是在 golden 前置阶段抛 AttributeError/IndexError。
    if _numel(index) == 0 or _numel(k_cache) == 0 or len(k_cache.shape) < 3:
        return t
    total = int(index.size if for_npu else index.numel())
    block_num = int(k_cache.shape[0])
    block_size = int(k_cache.shape[2])
    neg = NEG_MARK in str(kwargs.get("testcase_name", ""))
    t[5] = _fill_index(
        index, total, block_num, block_size, _seed_of(kwargs), neg=neg, as_numpy=for_npu
    )
    return t


class QkvRmsNormRopeCacheKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray,返 numpy.ndarray。"""

    def customize_inputs(*inputs, **kwargs):
        # 走 numpy 侧:_prepare 只重建 index,其余输入原对象返回,dtype 不会漂移。
        return _prepare(inputs, for_npu=True, **kwargs)

    def golden(*inputs, **kwargs):
        # ⚠️ 输出 dtype 的推导必须在 `_prepare` **之前**,并且只认【原始入参】:
        # `_prepare` 里的 `_to_tensor` 会把 ml_dtypes 的 bf16 抬成 fp32,`_compute`
        # 再把一切抬到 fp64 —— 在 prepared 上推导会一律得到 float32/float64,
        # 与算子声明的 bf16 输出不对等。这里取位置参数的 0(qkv)/7(k_cache)/8(v_cache);
        # 可选输入可能是 None、入参也可能被截断,故用 `get` 做 None/越界安全取值,
        # `_derive_out_dtypes` 对 None 会回落到 qkv 的 dtype。
        # 与 aclnn 通路同一套语义:**从原始入参推导 -> 计算 -> 结果 cast 回**,
        # 不再使用 TTK 下发的 `output_dtypes`(golden 收的是 numpy 入参,
        # 输出 dtype 应由入参决定)。
        derived = _derive_out_dtypes(get(inputs, 0), get(inputs, 7), get(inputs, 8))
        tensors = _prepare(inputs, **kwargs)
        # k_cache/v_cache 缺省或零元素时推导不出 cache 布局(_scatter_pa_nz 会解引用 None),
        # 直接返回全 None 占位:让算子的拒绝成为可观察结果,而不是 golden 抛 AttributeError。
        if _numel(get(tensors, 7)) == 0 or _numel(get(tensors, 8)) == 0:
            return [None] * 6
        picked = _pick(kwargs)
        outs = _mask_unrequested_qkv(
            _compute(*tensors, **picked), picked.get("is_output_qkv", False)
        )
        # None = is_output_qkv=false 时 3-5 按契约不产出(SUPPRESSED),不 cast(不能变成数值);
        # 其余槽位统一 cast 回上面推导出的 dtype,bf16 由 `_np_of` 走 ml_dtypes。
        return [
            None if o is None else _np_of(o, derived[i]) for i, o in enumerate(outs)
        ]

    third_party = {"torch": _Compose}
    tolerance = _TOL


class QkvRmsNormRopeCacheAclnnSpec:
    """aclnn 通路。golden 收已 H2D 的 torch.Tensor,直接返 Tensor。

    参数名取自 op_host/op_api/aclnn_qkv_rms_norm_rope_cache.h。
    """

    # aclnn 形参序(与 aclnn_qkv_rms_norm_rope_cache.h 一一对应,共 20 个,全部按位置下发):
    #   13 张量 -> qkvSize -> headNums -> epsilon -> cacheModeOptional -> 3 个 proto 输出
    def _split(
        qkv,
        qGamma,
        kGamma,
        cos,
        sin,
        index,
        qOut,
        kCache,
        vCache,
        kScale,
        vScale,
        kOffset,
        vOffset,
        qkvSize,
        headNums,
        epsilon,
        cacheModeOptional,
        qOutBeforeQuant,
        kOutBeforeQuant,
        vOutBeforeQuant,
        **kwargs,
    ):
        tensors = [
            qkv,
            qGamma,
            kGamma,
            cos,
            sin,
            index,
            qOut,
            kCache,
            vCache,
            kScale,
            vScale,
            kOffset,
            vOffset,
        ]
        picked = {
            "qkv_size": qkvSize,
            "head_nums": headNums,
            "epsilon": 1e-6 if epsilon is None else epsilon,
            # aclnn 层由 proto 指针是否存在推导 is_output_qkv(与 op_api L2 的判据一致)
            "is_output_qkv": qOutBeforeQuant is not None,
            "output_dtypes": kwargs.get("output_dtypes"),
        }
        return tensors, picked, kwargs

    def customize_inputs(
        qkv,
        qGamma,
        kGamma,
        cos,
        sin,
        index,
        qOut,
        kCache,
        vCache,
        kScale,
        vScale,
        kOffset,
        vOffset,
        qkvSize,
        headNums,
        epsilon,
        cacheModeOptional,
        qOutBeforeQuant,
        kOutBeforeQuant,
        vOutBeforeQuant,
        **kwargs,
    ):
        """aclnn 通路:原地改写 index 后返回 None。

        TTK 契约(ttk/core_modules/aclnn/input_generate.py 的 _apply_custom_input_result):
        钩子要么返回"纯输出槽之外的输入张量序列",要么返回 None 表示原地写法。
        本算子有 6 个纯输出槽(qOut/kCache/vCache + 三路 before-quant),逐个回填既
        依赖用例的 inplace/pure-output 标注,也容易与槽位错配 —— 返回 None 直接走
        原地写法,index 的改写用 copy_ 落到原张量上。
        """
        tensors = [
            qkv,
            qGamma,
            kGamma,
            cos,
            sin,
            index,
            qOut,
            kCache,
            vCache,
            kScale,
            vScale,
            kOffset,
            vOffset,
        ]
        new = _prepare(tensors, **kwargs)
        if hasattr(index, "copy_") and new[5] is not None:
            index.copy_(new[5])
        return None

    def golden(
        qkv,
        qGamma,
        kGamma,
        cos,
        sin,
        index,
        qOut,
        kCache,
        vCache,
        kScale,
        vScale,
        kOffset,
        vOffset,
        qkvSize,
        headNums,
        epsilon,
        cacheModeOptional,
        qOutBeforeQuant,
        kOutBeforeQuant,
        vOutBeforeQuant,
        **kwargs,
    ):
        tensors, picked, kwargs = QkvRmsNormRopeCacheAclnnSpec._split(
            qkv,
            qGamma,
            kGamma,
            cos,
            sin,
            index,
            qOut,
            kCache,
            vCache,
            kScale,
            vScale,
            kOffset,
            vOffset,
            qkvSize,
            headNums,
            epsilon,
            cacheModeOptional,
            qOutBeforeQuant,
            kOutBeforeQuant,
            vOutBeforeQuant,
            **kwargs,
        )
        # ⚠️ 输出 dtype 推导必须发生在**任何 dtype 抬升之前**:用 golden 的入参
        # qkv/kCache/vCache,不要用 prepared —— _prepare 经 _to_tensor 把 bf16 抬成
        # fp32、_compute 再抬到 fp64,在 prepared 上推会得到 float64/float32。
        derived = None
        if not kwargs.get("output_dtypes"):
            derived = _derive_out_dtypes(qkv, kCache, vCache)
        prepared = _prepare(tensors, **kwargs)
        # 与 KernelSpec.golden 同一守卫:cache 缺省/零元素时 _scatter_pa_nz 无从推导,
        # 返回全 None 占位(aclnn 通路 6 路输出),让算子的拒绝成为可观察结果。
        if _numel(prepared[7]) == 0 or _numel(prepared[8]) == 0:
            return [None] * 6
        # is_output_qkv=false(aclnn 侧由 qOutBeforeQuant 指针是否存在推导)时,
        # 3-5 按契约不产出 → None,ttk 侧表现为该输出不比较。
        outs = _mask_unrequested_qkv(
            _compute(*prepared, **picked), picked.get("is_output_qkv", False)
        )
        return _cast_outs(outs, kwargs, derived)

    third_party = {"torch": _Compose}
    tolerance = _TOL


class QkvRmsNormRopeCacheTorchSpec:
    """e2e(torch_npu.npu_qkv_rms_norm_rope_cache)。

    ⚠️ 形参序与 aclnn **不同**,必须按 torch schema 写(libtorch_npu 实测):

        (qkv, q_gamma, k_gamma, cos, sin, index,
         q_out, k_cache, v_cache,          # 原地输出,既是输入也是返回值
         int[4] qkv_size, int[3] head_nums, # ← 位置参数,在 * 之前
         *, k_scale=None, v_scale=None, k_offset=None, v_offset=None,   # 可选张量
         float epsilon=1e-6, str cache_mode="PA_NZ", bool is_output_qkv=False)
        -> (Tensor, Tensor, Tensor)        # ← 只返回 3 个,proto 不返回

    所以属性里 `qkv_size`/`head_nums` **走位置**、不走 kwargs —— 早先按
    `golden(*inputs, **kwargs)` 写,`_pick(kwargs)` 取不到它们,`_compute` 里
    `[int(v) for v in _as_int_list(None)]` 直接 TypeError。
    """

    def golden(
        qkv,
        q_gamma,
        k_gamma,
        cos,
        sin,
        index,
        q_out,
        k_cache,
        v_cache,
        qkv_size,
        head_nums,
        k_scale=None,
        v_scale=None,
        k_offset=None,
        v_offset=None,
        epsilon=1e-6,
        cache_mode="PA_NZ",
        is_output_qkv=False,
        **kwargs,
    ):
        tensors = [
            qkv,
            q_gamma,
            k_gamma,
            cos,
            sin,
            index,
            q_out,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            k_offset,
            v_offset,
        ]
        # 与 KernelSpec.golden 同一守卫:cache 缺省/零元素时 _scatter_pa_nz 无从推导。
        # 注意 e2e 只返回 3 路(torch schema),占位也按 3 路给。
        prepared = _prepare(tensors, **kwargs)
        if _numel(prepared[7]) == 0 or _numel(prepared[8]) == 0:
            return [None] * 3
        # e2e 通路 TTK **不传 output_dtypes**(只有 kernel/aclnn 传),不补的话
        # _compute 因向上兜底把 q_out 留在 fp32,而算子实际输出是 fp16/bf16 ->
        # 比对时 dtype 不对等(实测 Golden Dtypes 首项为 float32)。
        # 输出 dtype 由输入推:q_out 同 qkv;k_cache/v_cache 同各自的输入 cache。
        out_dtypes = kwargs.get("output_dtypes")
        if not out_dtypes:
            # ⚠️ 三个输出都用 **qkv 的 dtype**,不是各自输入 cache 的 dtype。
            # torch 绑定返回的是量化**前**的逐 token k/v(见下方返回索引说明),
            # 其 dtype 与 qkv 一致;早先按输入 cache 取 dtype,会在 v 量化(输入 int8)
            # 的用例上把 golden 定成 int8、去比 NPU 的 float16 —— 实测匹配率骤降到 25%。
            # 与 _Compose 共用同一条推导:_derive_out_dtypes 的第 0/4/5 项即 qkv 的 dtype。
            derived = _derive_out_dtypes(qkv)
            out_dtypes = [derived[i] for i in (0, 4, 5)]
        outs = _compute(
            *prepared,
            qkv_size=qkv_size,
            head_nums=head_nums,
            epsilon=1e-6 if epsilon is None else epsilon,
            cache_mode=cache_mode or "PA_NZ",
            is_output_qkv=bool(is_output_qkv),
            output_dtypes=out_dtypes,
        )
        # ⚠️ 取 outs[0]/[4]/[5],**不是** outs[:3]。
        # `_compute` 返回 [q_rows, k_final, v_final, q_rows.clone(), k_rows, v_rows]:
        #   · k_final / v_final = PA_NZ 缓存(4D [bn, nk*D/D0, bs, D0])
        #   · k_rows  / v_rows  = 逐 token 的投影值(2D [T, nk*D])
        # torch_npu 绑定返回的是**后者**:实测即便显式传 cache_mode='PA_NZ'、且传入 PA_NZ 4D
        # 的原地缓冲,返回仍是 2D([T, Nk*D] / [T, Nv*D])。对照实测(b2 s8 q4 k1 v1 D128):
        #   NPU [(16,512),(16,128),(16,128)]
        #   golden q_rows(16,512) / k_rows(16,128) / v_rows(16,128)   ← 逐项吻合
        # 早先取 outs[:3] 会拿 PA_NZ 4D 去比 2D,直接报尺寸不符(133120 vs 180224),全例 FAIL。
        return [
            outs[i].to(_torch_dtype(out_dtypes[j])) for j, i in enumerate((0, 4, 5))
        ]  # torch schema 只返回 3 个(q_out / k / v)

    def customize_inputs(*inputs, **kwargs):
        """e2e 通路:与 aclnn 同理,index 需原地改写(返回值被 TTK 丢弃)。"""
        index = inputs[5] if len(inputs) > 5 else None
        new = _prepare(list(inputs), **kwargs)
        if index is not None and hasattr(index, "copy_"):
            index.copy_(new[5])
        return list(inputs)

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】tf / onnx / caffe 通路:算子目录内无 framework 插件源、无 onnx 插件源,
# 全仓 find -name '*qkv_rms_norm*' 仅命中算子目录自身。

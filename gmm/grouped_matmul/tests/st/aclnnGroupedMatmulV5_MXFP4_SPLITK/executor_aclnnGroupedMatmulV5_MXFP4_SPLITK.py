#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
grouped_matmul MXFP4 + group_type=2（k 轴分组）TN 场景 ST executor（A5 / Ascend950DT）。

与库上 ST（aclnnGroupedMatmulV5_A8W8O8 / V4_SplitK）同构：双 executor（CPU golden +
device kernel），per-case 配置（k_groups / seed）下沉 json 的 attrs 字段，executor 经
input_data.kwargs 读取（k_groups 为 G=组数 个同名 attrs，ATK 聚合为 list；V4_SplitK
KTiling 同模式）。为何自建而非通用 golden：ATK 生成不出正确 FP4 字节布局、无 stride
列表达 sA 转置视图、k 轴分组 golden 在本仓未验证——故 init_by_input_data rebuild 正确
FP4 layout，并自带权威 golden。权威精度以本文件独立 runner 为准（ATK 对 bfloat16
output 的 double-benchmark verdict 为 `-`，不可信）。
运行：source /usr/local/Ascend/cann-9.1.0/set_env.sh; python3 executor_aclnnGroupedMatmulV5_MXFP4_SPLITK.py
"""

import ctypes
import json
import os
import sys

import numpy as np
import torch
import torch_npu

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

# ATK 内置 DataTypeFactory 不认 float4_e2m1 / float8_e8m0（也不在 EXCEPT_TYPES 兜底），
# `atk task` 数据生成阶段会 KeyError。下列 datatype plugin 注册这两个名字让框架走流程；
# 占位张量会被 executor 的 init_by_input_data 用 build_case 重建的正确 FP4 layout 覆盖。
# 仅 `atk task -p <本文件>` 时生效；直接 `python3 <本文件>` 不走 ATK 数据生成，零影响。
from atk.case_generator.generator.data_types import DATATYPE_REGISTRY
from atk.case_generator.generator.data_types.data_base import BaseDataType

# ============================================================================
# 用例矩阵来源：json 的 attrs 字段 kGroups(G 个同名) + seed；本文件不内置 CASE_TABLE。
#   case_name -> (G, m, n, k_groups, seed)，与 json 一一对应（详设 §8.2 T1–T8 + §2.3 customer）。
#   不纳入 K=1 + [1,1] 打包歧义用例（详设 T8–T10）：PTA aclTensor 转换限制，非本功能精度项。
# ============================================================================
# E2M1 幅值表：code 低 3 位 -> 幅值；最高位为符号位（1 表示负）
FP4_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# E8M0 编码范围：129..135 对应 2^2..2^8；255 为 NaN，必须避开
E8M0_CODE_LO = 129
E8M0_CODE_HI = 135


def fp4_decode(codes):
    """E2M1 4bit 编码 -> float64 精确值（与概设 golden 的 fp4() 一致）。"""
    signs = np.where((codes & 0x8) != 0, -1.0, 1.0)
    mags = np.array(FP4_E2M1_MAG, dtype=np.float64)[codes & 0x7]
    return signs * mags


def e8m0_decode(bytes_u8):
    """E8M0 编码 -> float64：value = 2^(code-127)，code 255 为 NaN（构造时不产生）。"""
    assert not np.any(bytes_u8 == 255), "E8M0 code 255 is NaN, must not be generated"
    return np.exp2((bytes_u8.astype(np.int64) - 127).astype(np.float64))


def pack_fp4(codes):
    """[rows, cols] 4bit 编码 -> [rows, cols/2] 字节：低 nibble 为偶数列（列 0 在低 4 位）。"""
    rows, cols = codes.shape
    assert cols % 2 == 0, f"打包轴必须为偶数，实际 cols={cols}"
    low = codes[:, 0::2].astype(np.uint8)
    high = codes[:, 1::2].astype(np.uint8) << 4
    return (low | high).astype(np.uint8)


def make_fp4_tensor(packed, transpose, device):
    """用紧密 packed 字节构造 FP4 张量（按字节形状分配，见文件头）。

    torch 的 float4_e2m1fn_x2 的 shape 按「字节」计，而 ACL FLOAT4_E2M1 的 shape 是
    「逻辑元素数」；按字节形状分配后 torch_npu 转 aclTensor 时把打包轴翻倍成逻辑形状。
    转置视图必须在 **NPU 张量上**做 transpose（纯 view 操作），否则 .npu() 会把
    CPU 侧的非连续视图重排成连续，丢失转置语义。
    """
    buf = torch.empty(packed.shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    buf.view(torch.uint8).copy_(torch.from_numpy(packed))
    buf_npu = buf.npu()
    return buf_npu.t() if transpose else buf_npu


def make_sa_tensor(sa_phys_u8, device):
    """sA 物理 [S,m,2] -> 对外转置视图 [m,S,2]，stride (2, m*2, 1)。"""
    t = torch.from_numpy(sa_phys_u8.copy()).view(torch.float8_e8m0fnu)
    return t.npu().transpose(0, 1)


def group_scale_start(p_g, g):
    """组 g 在 S 轴上的起始行号：s_g = floor(p_g/64) + g（详设 4.4）。"""
    return p_g // 64 + g


def golden_mxfp4_splitk(a_codes, b_codes, sa_codes, sb_codes, k_groups, m, n):
    """逐组精确 golden（float64），返回 [G,m,n]。空组 k_g=0 输出全零 [m,n]。"""
    a_val = fp4_decode(a_codes)
    b_val = fp4_decode(b_codes)
    sa_val = e8m0_decode(sa_codes)
    sb_val = e8m0_decode(sb_codes)

    G = len(k_groups)
    out = np.zeros((G, m, n), dtype=np.float64)
    p_g = 0
    for g, k_g in enumerate(k_groups):
        if k_g == 0:
            continue
        s_g = group_scale_start(p_g, g)
        a_g = a_val[p_g : p_g + k_g, :]
        b_g = b_val[p_g : p_g + k_g, :]
        acc = np.zeros((m, n), dtype=np.float64)
        for b in range((k_g + 31) // 32):
            lo = 32 * b
            hi = min(32 * (b + 1), k_g)
            row = s_g + b // 2
            sub = b % 2
            dot = a_g[lo:hi, :].T @ b_g[lo:hi, :]
            acc += dot * sa_val[row, :, sub][:, None] * sb_val[row, :, sub][None, :]
        out[g] = acc
        p_g += k_g
    return out


def build_case(name, m, n, k_groups, seed):
    """构造一组 NPU 输入 + 权威 golden（参数显式传入，供 ATK executor 与独立 runner 复用）。

    G 派生自 len(k_groups)。k_groups / seed 在 ATK 通路来自 input_data.kwargs，在独立
    runner通路来自 json；m/n 来自 x/weight 占位张量 shape（= json 逻辑 shape）。
    """
    G = len(k_groups)
    k = int(sum(k_groups))
    assert k > 0, "总 k 必须为正（详设 5.3.2：kSize==0 会触发 tiling 除零）"
    assert m % 2 == 0 and n % 2 == 0, "TN 下 m、n 是 FP4 打包轴，必须为偶数"
    S = k // 64 + G

    rng = np.random.RandomState(seed)
    a_codes = rng.randint(0, 16, size=(k, m)).astype(np.uint8)
    b_codes = rng.randint(0, 16, size=(k, n)).astype(np.uint8)
    sa_codes = rng.randint(E8M0_CODE_LO, E8M0_CODE_HI + 1, size=(S, m, 2)).astype(
        np.uint8
    )
    sb_codes = rng.randint(E8M0_CODE_LO, E8M0_CODE_HI + 1, size=(S, n, 2)).astype(
        np.uint8
    )
    # 尺度异组：每组用不同 E8M0 编码（覆盖详设 §8.2 T8 / 4.4 组间尺度偏移）
    for g in range(G):
        p_g = int(sum(k_groups[:g]))
        s_g = group_scale_start(p_g, g)
        sa_codes[s_g] = (E8M0_CODE_LO + g) % (E8M0_CODE_HI + 1)
        sb_codes[s_g] = (E8M0_CODE_LO + 2 * g) % (E8M0_CODE_HI + 1)

    x = make_fp4_tensor(
        pack_fp4(a_codes), transpose=True, device="npu"
    )  # 逻辑 view [m,k]，物理 [k,m]
    weight = make_fp4_tensor(
        pack_fp4(b_codes), transpose=False, device="npu"
    )  # 逻辑 [k,n]，物理 [k,n]
    sa = make_sa_tensor(sa_codes, device="npu")  # view [m,S,2]，物理 [S,m,2]
    sb = torch.from_numpy(sb_codes.copy()).view(torch.float8_e8m0fnu).npu()  # [S,n,2]
    group_list = torch.tensor(np.cumsum(k_groups), dtype=torch.int64).npu()  # 前缀和

    golden = golden_mxfp4_splitk(a_codes, b_codes, sa_codes, sb_codes, k_groups, m, n)
    golden_np = torch.from_numpy(golden).to(torch.bfloat16).to(torch.float32).numpy()

    print(
        f"  [{name}] x={tuple(x.shape)} w={tuple(weight.shape)} sa={tuple(sa.shape)} "
        f"sb={tuple(sb.shape)} gl={group_list.cpu().tolist()}"
    )
    return {
        "x": x,
        "weight": weight,
        "sa": sa,
        "sb": sb,
        "group_list": group_list,
        "golden_np": golden_np,
        "m": m,
        "n": n,
        "k_groups": k_groups,
        "S": S,
        "G": G,
        "k": k,
    }


def _invoke(x, weight, sa, sb, group_list):
    """公共调用体：split_item=3 / group_type=2 / group_list_type=0 / 双 E2M1+双 E8M0+BF16。"""
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        bias=None,
        scale=[sb],
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=[sa],
        group_list=group_list,
        split_item=3,
        group_type=2,
        group_list_type=0,
        output_dtype=torch.bfloat16,
        x_dtype=torch_npu.float4_e2m1fn_x2,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]


# ============================================================================
# ATK datatype 插件（仅 `atk task` 数据生成用；直接执行本文件不经过这里）
# ============================================================================
@DATATYPE_REGISTRY.register("float4_e2m1")
@DATATYPE_REGISTRY.register("float4_e2m1fn_x2")
class DataTypeFloat4E2M1(BaseDataType):
    """FP4(E2M1) 占位：真实布局由 executor 的 init_by_input_data 用 build_case 重建。"""

    def gen_data(self):
        return torch.zeros(list(self.config.shape), dtype=torch.uint8)


@DATATYPE_REGISTRY.register("float8_e8m0")
@DATATYPE_REGISTRY.register("float8_e8m0fnu")
class DataTypeFloat8E8M0(BaseDataType):
    """E8M0 占位：真实布局由 executor 的 init_by_input_data 用 build_case 重建。"""

    def gen_data(self):
        return torch.zeros(list(self.config.shape), dtype=torch.uint8)


# ============================================================================
# 用例参数读取：ATK 通路经 input_data.kwargs；独立 runner 经 json
# ============================================================================
def _read_case_params(kwargs):
    """从 input_data.kwargs 取 (m, n, k_groups, seed)。

    m/n/seed/kGroups 均经 json attrs 传入（attrs/int 不被框架的 FP4 字节表示改变）。
    注意：不可用 kwargs["x"]/["weight"] 占位张量的 shape 取 m/n——pyaclnn backend 会把
    FP4(E2M1) 张量按「字节表示」落数据，占位张量 shape 是 (m/2, k)/(k, n/2)，按它取会
    # 小 4 倍（实测 T3：占位 x=(128,192) 而逻辑 x=(256,192)）。m/n 必须从 attrs 取。
    """

    def _last(v):
        return v[-1] if isinstance(v, (list, tuple)) else v

    m = _last(kwargs["m"])
    n = _last(kwargs["n"])
    seed = _last(kwargs["seed"])
    k_groups = list(kwargs["kGroups"])
    return m, n, k_groups, seed


def _load_case_params(json_path=None):
    """独立 runner 用：读 json 构造 name -> (m, n, k_groups, seed)。"""
    json_path = json_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "atk_aclnnGroupedMatmulV5_MXFP4_SPLITK.json",
    )
    data = json.load(open(json_path))
    table = {}
    for c in data:
        name = c["name"]
        kgs = seed = m = n = None
        for inp in c["inputs"]:
            if isinstance(inp, list):  # group（kGroups/seed/m/n 均按 group 存）
                gname = inp[0]["name"]
                if gname == "kGroups":
                    kgs = [x["range_values"] for x in inp]
                elif gname == "seed":
                    seed = inp[0]["range_values"]
                elif gname == "m":
                    m = inp[0]["range_values"]
                elif gname == "n":
                    n = inp[0]["range_values"]
        table[name] = (m, n, kgs, seed)
    return table


# ============================================================================
# ATK executor（register 类）：npu kernel + cpu golden，同源输入
# ============================================================================
@register("function_grouped_matmul_mxfp4_splitk")
class GroupedMatmulMxFp4SplitKApi(BaseApi):
    """MXFP4 + group_type=2（k 轴分组）TN 精度 executor。

    - npu : torch_npu.npu_grouped_matmul PTA 直调（单算子即刻执行）
    - cpu : golden_mxfp4_splitk（FP4 精确解码 + 逐 32-k 双侧尺度 + S=floor(k/64)+G
            + s_g=floor(p_g/64)+g + 空组全零），float64 高精度真值
    输入在 init_by_input_data 重建为正确 FP4 layout（字节打包 + sA 转置视图 +
    group_list 前缀和），npu/cpu 共用同一份输入。
    """

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.case_name = self.task_result.case_config.name
        self._built = None

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        # case 名驱动：按 json 的 kGroups/seed 重建正确 FP4 layout（忽略框架按 shape/dtype
        # 随机生成的字节布局——那不对，见文件头）。
        m, n, k_groups, seed = _read_case_params(input_data.kwargs)
        self._built = build_case(self.case_name, m, n, k_groups, seed)
        # 同步回 input_data.kwargs，使框架侧元数据与重建输入一致（名称对齐）
        input_data.kwargs["x"] = self._built["x"]
        input_data.kwargs["weight"] = self._built["weight"]
        input_data.kwargs["scale"] = self._built["sb"]
        input_data.kwargs["perTokenScale"] = self._built["sa"]
        input_data.kwargs["groupList"] = self._built["group_list"]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        b = self._built
        if self.device == "npu":
            out = _invoke(b["x"], b["weight"], b["sa"], b["sb"], b["group_list"])
            torch.npu.synchronize()
            return [out.contiguous()]
        # cpu golden：float64 真值 -> bf16 -> float32（与权威小脚本同口径，供框架比对）
        golden = torch.from_numpy(b["golden_np"]).to(torch.float32)
        return [golden.contiguous()]


# ============================================================================
# ATK device（pyaclnn）executor：PTA 直调真 kernel
# ============================================================================
# 为什么 device 侧自建 executor（而非用框架 aclnn_function 通用路径）：
#   ATK（26.3.6 与 26.9.24 皆然）的 acl C-tensor 通路不认 FP4(E2M1)/FP8(E8M0)——
#   TORCH_TO_ACLTYPE/`TorchDtype_to_AclDtype` 无这两个 dtype 映射，`AclDataType`
#   枚举虽在 26.9.24 补了 ACL_FLOAT4_E2M1/ACL_FLOAT8_E8M0，但 `create_acl_tensor`
#   仍按 torch 的「字节 shape」直传、且 `aclDataTypeSize`/shape 不做 FP4 的
#   字节→逻辑翻倍。通用路径会在 `convert_input_data -> create_acl_tensor` 对
#   `torch.float4_e2m1fn_x2` 抛 `KeyError`（见 atk.log 实测）。
#   故 device kernel 走需求界定的 PTA 直调 `torch_npu.npu_grouped_matmul`
#   （与权威独立 runner、CPU golden 同源输入），FP4 打包→逻辑 shape 由 PTA 正确处理。
#
# 与库上 ST（aclnnGroupedMatmulV5_A8W8O8）同构的双 executor：
#   api_type      = function_grouped_matmul_mxfp4_splitk  -> 上方 GroupedMatmulMxFp4SplitKApi（CPU golden）
#   aclnn_api_type = aclnn_grouped_matmul_mxfp4_splitk     -> 下方本类（device kernel）
# before_call 兼容性（已逐行核对 pyaclnn_backend）：
#   - `get_cpp_func_signature_type()` 返回 None -> check_info=None ->
#     `check_signature_type_diff(arg_type_list, None)` 直接 return，跳过 V5 C 签名校验；
#   - input_args/account 占位 c_void_p 仅用于绑定 `aclnnGroupedMatmulV5GetWorkspaceSize`
#     （bind_function 只 getattr+设 argtypes，绑定后本类 __call__ 从不调用它）；
#   - output_packages 用框架支持的 FP32 aclTensor 占位，after_call() 不读它。
@register("aclnn_grouped_matmul_mxfp4_splitk")
class GroupedMatmulMxFp4SplitKAclnnApi(AclnnBaseApi):
    """MXFP4 + group_type=2（k 轴分组）TN 场景 device（pyaclnn）精度 executor。

    - npu : torch_npu.npu_grouped_matmul PTA 直调（单算子即刻执行），与 CPU golden 同源输入
    - after_call : 返回 PTA 结果（BF16->FP32 无损），与 CPU golden 同口径供框架比对
    输入在 init_by_input_data 重建为正确 FP4 layout（build_case，与 CPU golden 同 seed）。
    """

    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.case_name = task_result.case_config.name
        self._built = None
        self._out = None

    def init_by_input_data(self, input_data: InputDataset):
        # case 名驱动：按 json 的 kGroups/seed rebuild 正确 FP4 layout（与 CPU golden 同 seed 同输入）。
        m, n, k_groups, seed = _read_case_params(input_data.kwargs)
        self._built = build_case(self.case_name, m, n, k_groups, seed)
        b = self._built
        # 同步回 input_data.kwargs，使框架侧元数据与重建输入一致（名称对齐）
        input_data.kwargs["x"] = b["x"]
        input_data.kwargs["weight"] = b["weight"]
        input_data.kwargs["scale"] = b["sb"]
        input_data.kwargs["perTokenScale"] = b["sa"]
        input_data.kwargs["groupList"] = b["group_list"]

        # 输出占位 aclTensor（FP32，框架支持；after_call 不读它，仅满足 set_input_dataset
        # 的出参类型校验）。shape/dtype 与 CPU golden 输出信息 [G,m,n]/FP32 一致。
        out_t = torch.empty([b["G"], b["m"], b["n"]], dtype=torch.float32, device="npu")
        out_struct = self.backend.torch_tensor_to_acl(out_t)

        # 占位入参：仅满足 before_call 绑定 aclnnGroupedMatmulV5GetWorkspaceSize
        # （bind 后从不调用；签名校验被 get_cpp_func_signature_type->None 跳过）。
        n_args = 16
        input_args = [ctypes.c_void_p(0) for _ in range(n_args)]
        return input_args, [out_struct]

    def __call__(self):
        b = self._built
        out = _invoke(b["x"], b["weight"], b["sa"], b["sb"], b["group_list"])
        torch.npu.synchronize()
        self._out = out.contiguous()
        return None

    def after_call(self, output_packages):
        # 与 CPU golden 同口径：BF16 -> FP32（无损），供框架精度比对。
        return [self._out.to(torch.float32).contiguous()]

    def get_cpp_func_signature_type(self):
        # 返回 None -> before_call 跳过 aclnnGroupedMatmulV5 的 C 签名校验（本类走 PTA）。
        return None


# ============================================================================
# 独立 runner（保底可验证通路：直接 python3 跑，不依赖 ATK 框架调度）
# ============================================================================
def _run_one(name, params, prec=0.01):
    m, n, k_groups, seed = params
    built = build_case(name, m, n, k_groups, seed)
    golden_np = built["golden_np"]
    k_groups = built["k_groups"]

    actual = _invoke(
        built["x"], built["weight"], built["sa"], built["sb"], built["group_list"]
    )
    torch.npu.synchronize()
    actual = actual.to(torch.float32).cpu().numpy()

    # 空组必须严格为 0（详设 5.7）
    for g, k_g in enumerate(k_groups):
        if k_g == 0 and not np.all(actual[g] == 0):
            raise AssertionError(f"[{name}] 空组 g={g} 输出不全为 0")
    err = np.abs(actual - golden_np)
    max_err = float(err.max()) if err.size else 0.0
    scale_ref = float(np.abs(golden_np).max()) if golden_np.size else 1.0
    # 相对误差断言（与权威小脚本 assertRtolEqual 同口径）
    denom = np.abs(golden_np) + (np.abs(golden_np) < 1e-6) * 1e-6
    re = np.abs(actual - golden_np) / denom
    max_re = float(re.max()) if re.size else 0.0
    assert max_re <= prec, f"[{name}] max_re={max_re:.4g} > prec={prec}"
    print(
        f"  [{name}] PASS  max|err|={max_err:.4g} (ref={scale_ref:.4g}) max_re={max_re:.4g}"
    )
    return True


def run_all(names=None, prec=0.01):
    table = _load_case_params()
    names = names or list(table.keys())
    fails = []
    for name in names:
        try:
            _run_one(name, table[name], prec=prec)
        except Exception as e:
            fails.append((name, repr(e)))
            print(f"  [{name}] FAIL: {e}")
    if fails:
        print(f"\nFAILED {len(fails)}/{len(names)}: {[f[0] for f in fails]}")
        return False
    print(f"\nALL PASS ({len(names)} cases)")
    return True


if __name__ == "__main__":
    # 默认：NPU 独立 runner，执行全部用例（功能 7 + customer 6）；用例参数来自 json
    print(f"[mode] 独立 runner, NPU cases: all {len(_load_case_params())}")
    ok = run_all()
    sys.exit(0 if ok else 1)

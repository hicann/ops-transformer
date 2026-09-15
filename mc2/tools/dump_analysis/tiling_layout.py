#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MC2 (DFX) TilingData 布局计算器 (混合模式)。

动态部分: 从 set_env.sh 对应 CANN 环境的 kernel_tiling.h 实时解析
  Mc2InitTiling / Mc2CcTiling / TCubeTiling 的尺寸。
  TCubeTiling 随 CANN 版本增删字段, 会影响后续 Matmul tiling 块和结构体
  总大小, 必须动态获取。
硬编码部分: DFX 算子自有结构 (RCSTiling / MatMulV3 尾部 / QuantBmm 固定段 /
  DfxDumpInfo / WorkspaceLayoutInfo / StateDump), 源自 ops-transformer 仓库, 结构演进时
  需同步更新本文件中的 DFX 表。

CANN 头文件定位顺序:
  $ASCEND_CANN_PACKAGE_PATH > $ASCEND_HOME_PATH > $ASCEND_TOOLKIT_HOME
  > $ASCEND_AICPU_PATH (即 source set_env.sh 后的环境变量)
找不到 CANN 头文件时: 告警并使用默认尺寸 (TCubeTiling=200 等) 继续。
"""

import glob
import logging
import os
import re

# ====== 动态部分: CANN kernel_tiling.h ======
CANN_ENV_VARS = (
    "ASCEND_CANN_PACKAGE_PATH",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_AICPU_PATH",
)

# 兜底默认值 (无 CANN 环境时使用; 9.1.0 / 9.2.0 实测一致)
DEFAULT_MC2_INIT_TILING_SIZE = 64
DEFAULT_MC2_CC_TILING_SIZE = 512
DEFAULT_TCUBE_TILING_SIZE = 200

# ====== DFX 硬编码表 (ops-transformer 仓库自有结构, 结构演进时同步更新) ======

# RCSTiling — mc2/common/op_kernel/mc2_tiling_struct.h
# 16×u32 + 3×u64 + 9×u32 = 124B, 按 8 对齐 → 128B
RCS_MEMBERS = [
    ("rankDim", 4),
    ("rankID", 4),
    ("commtype", 4),
    ("subtype", 4),
    ("tileCnt", 4),
    ("tailM", 4),
    ("tailCnt", 4),
    ("biasLen", 4),
    ("isAdd", 4),
    ("rankM", 4),
    ("rankN", 4),
    ("rankK", 4),
    ("gatherIndex", 4),
    ("isTransposeA", 4),
    ("isTransposeB", 4),
    ("storageGather", 4),
    ("nd2NzWorkLen", 8),
    ("cToFloatLen", 8),
    ("gatherLen", 8),
    ("workspaceAddr4", 4),
    ("aicCoreNum", 4),
    ("needUbBuffer", 4),
    ("addX3UbCnt", 4),
    ("commWorkSpaceSize", 4),
    ("isInputCommQuantScale", 4),
    ("dataType", 4),
    ("commInt8WorkSpace", 4),
    ("dynamicQuantTempBuffSize", 4),
]

# Mc2MatMulV3TilingData — mc2/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h
# 结构 = TCubeTiling + 9×u32 (mTailCnt/nTailCnt/kTailCnt/mBaseTailSplitCnt/
#        nBaseTailSplitCnt/mTailMain/nTailMain/isHf32/aswWindowLen)
MMV3_TAIL_SIZE = 36

# Mc2QuantBatchMatmulV3TilingDataParams — mc2/3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmv3_arch35_tiling_data.h
# 结构 = DataParams(32×u32=128) + TCubeTiling + L2cache(6×u32=24) + SlidingWindow(3×u32=12)
QUANT_FIXED_SIZE = 164

# DFX 统一头部 — mc2/common/op_kernel/apace/utils/op_state_dump_struct.h
# 当前解析的是 MC2_DFX_ENABLE=1 的 Dump；关闭时 DfxDumpInfo 为空且不会生成这类 DFX Dump。
WS_SEG_INFO_SIZE = 24  # u64 offset + u64 size + u8 type + u8[7] reserved
WS_MAX_SEGMENTS = 8
WS_LAYOUT_INFO_SIZE = 208  # u64 + 2×u32 + 8×24
PEERMEM_DATA_SIZE_OFFSET = WS_LAYOUT_INFO_SIZE
PEERMEM_LAYOUT_OFFSET = PEERMEM_DATA_SIZE_OFFSET + 8
DFX_DUMP_INFO_SIZE = PEERMEM_LAYOUT_OFFSET + WS_LAYOUT_INFO_SIZE  # 424B
DFX_WORKSPACE_LAYOUT_OFFSET = 0
RUNTIME_INFO_PER_CORE_SIZE = 512  # struct alignas(512) StateDumpPerCore
RUNTIME_INFO_CORE_MAX = 128

# AlltoAllMatmulTilingInfo — mc2/allto_all_matmul/op_kernel/arch35/allto_all_matmul_tiling_data.h
# 10×u32(40) + 6×u64(48) = 88B。量化模式和 tilingKey 不再存放在该结构中。
ALLTO_ALL_TILING_INFO_SIZE = 88

# Apace (CCU/URMA 路径, 不含 TCubeTiling, 无 CANN 版本漂移)
# — mc2/common/op_kernel/apace/tiling/comm_tiling_data.h (5×u64)
# — mc2/common/op_kernel/apace/tiling/quant_matmul_tiling_data.h (定长 u32 字段)
APACE_COMM_TILING_SIZE = 40
APACE_QUANT_MATMUL_SIZE = 64

# 当前 ops-transformer 源码的 ground truth（TCubeTiling=200），用于公式回归自检。
# AGMMV2/MMRS/A2AMM 的前缀为 [Mc2InitTiling][Mc2CcTiling][DfxDumpInfo]，
# 因此 DfxDumpInfo/WorkspaceLayout 位于 offset 576；URMA 算子仍把 DfxDumpInfo 放在 offset 0。
_GROUND_TRUTH = {
    "v2.offsets.workspaceLayout": 576,
    "v2.size": 1848,
    "fp8.offsets.workspaceLayout": 576,
    "fp8.size": 2232,
}

_GT_MRSV2 = {
    "nonQuant.wsLayoutOffset": 576,
    "nonQuant.size": 1608,
    "quant.wsLayoutOffset": 576,
    "quant.size": 1864,
}

_GT_A2A = {
    "nonQuant.wsLayoutOffset": 576,
    "nonQuant.size": 1560,
    "quant.wsLayoutOffset": 576,
    "quant.size": 1816,
    "apace.wsLayoutOffset": 576,
    "apace.size": 1112,
}

PRIM_TYPES = {
    "uint8_t": (1, 1),
    "int8_t": (1, 1),
    "char": (1, 1),
    "bool": (1, 1),
    "uint16_t": (2, 2),
    "int16_t": (2, 2),
    "uint32_t": (4, 4),
    "int32_t": (4, 4),
    "float": (4, 4),
    "uint64_t": (8, 8),
    "int64_t": (8, 8),
    "double": (8, 8),
    "size_t": (8, 8),
}


class LayoutError(Exception):
    pass


def _roundup(x, a):
    return (x + a - 1) // a * a


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


class HeaderParser:
    """解析 C 头文件中的 struct 定义 / #pragma pack / 数值常量。"""

    _STRUCT_RE = re.compile(
        r"\bstruct\s+(?:alignas\s*\(\s*(\d+)\s*\)\s*)?([A-Za-z_]\w*)\s*\{"
    )
    _PACK_RE = re.compile(r"@@PACK@@\(([^\n]*)\)")
    _CONSTEXPR_RE = re.compile(r"\bconstexpr\s+\w+\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
    _MEMBER_RE = re.compile(
        r"^(?:const\s+)?([A-Za-z_][\w:]*)\s*(\*)?\s*([A-Za-z_]\w*)"
        r"(?:\s*\[\s*([^\]]*?)\s*\])?\s*(?:=\s*(.*))?$"
    )

    def __init__(self):
        self.structs = {}
        self.constants = {}

    def parse_file(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = _strip_comments(f.read())
        lines = []
        for line in text.splitlines():
            s = line.strip()
            m = re.match(r"#\s*pragma\s+pack\s*\((.*)\)", s)
            if m:
                lines.append("@@PACK@@(%s)" % m.group(1).strip())
                continue
            m = re.match(r"#\s*define\s+([A-Za-z_]\w*)\s+(\S.*)$", s)
            if m:
                self.constants.setdefault(m.group(1), m.group(2).strip())
                continue
            if s.startswith("#"):
                continue
            lines.append(line)
        code = "\n".join(lines)
        for m in self._CONSTEXPR_RE.finditer(code):
            self.constants.setdefault(m.group(1), m.group(2).strip())
        self._scan(code, path)

    def _scan(self, code, path):
        events = [(m.start(), "struct", m) for m in self._STRUCT_RE.finditer(code)]
        events += [
            (m.start(), "pack", m.group(1).strip())
            for m in self._PACK_RE.finditer(code)
        ]
        events.sort(key=lambda e: e[0])
        pack, stack = None, []
        for _, kind, payload in events:
            if kind == "pack":
                arg = payload
                if arg.startswith("push"):
                    stack.append(pack)
                    inner = arg[4:].strip().lstrip(",").strip()
                    pack = int(inner) if inner else pack
                elif arg == "pop":
                    pack = stack.pop() if stack else None
                elif arg == "":
                    pack = None
                else:
                    pack = int(arg)
                continue
            m = payload
            depth, i = 1, m.end()
            while i < len(code) and depth > 0:
                if code[i] == "{":
                    depth += 1
                elif code[i] == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                raise LayoutError("struct 花括号不配对: %s (%s)" % (m.group(2), path))
            name = m.group(2)
            if name in self.structs:
                continue
            self.structs[name] = {
                "members": self._parse_members(code[m.end() : i - 1], name, path),
                "pack": pack,
                "alignas": int(m.group(1)) if m.group(1) else None,
                "file": path,
            }

    def _parse_members(self, body, struct_name, path):
        members = []
        for raw in body.split(";"):
            decl = raw.strip()
            if not decl:
                continue
            m = self._MEMBER_RE.match(decl)
            if not m:
                raise LayoutError(
                    "无法解析成员 '%s' (struct %s, %s)" % (decl, struct_name, path)
                )
            members.append(
                {
                    "type": m.group(1),
                    "ptr": bool(m.group(2)),
                    "name": m.group(3),
                    "arr": m.group(4),
                }
            )
        return members

    def eval_const(self, expr):
        def strip_suffix(s):
            return re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", s)

        e = strip_suffix(expr.strip())
        for _ in range(8):

            def sub(m):
                v = self.constants.get(m.group(0))
                return strip_suffix(v) if v is not None else m.group(0)

            new = re.sub(r"\b[A-Za-z_]\w*\b", sub, e)
            if new == e:
                break
            e = new
        if not re.fullmatch(r"[0-9a-fA-FxX+\-*/%()<> &|~\s]*", e):
            raise LayoutError("无法求值常量表达式: %s" % expr)
        try:
            return int(eval(e, {"__builtins__": {}}, {}))
        except Exception as exc:
            raise LayoutError("常量表达式求值失败: %s (%s)" % (expr, exc))


class LayoutComputer:
    """按 C 对齐规则 (含 #pragma pack) 计算结构体尺寸与成员偏移。"""

    def __init__(self, parser):
        self.p = parser
        self._cache = {}
        self._visiting = set()

    def struct_layout(self, name):
        if name in self._cache:
            return self._cache[name]
        if name in self._visiting:
            raise LayoutError("结构体循环引用: %s" % name)
        d = self.p.structs.get(name)
        if d is None:
            raise LayoutError("未知结构体: %s (头文件缺失?)" % name)
        self._visiting.add(name)
        try:
            pack = d["pack"]
            offset, max_natural = 0, 1
            offsets = {}
            for mem in d["members"]:
                size, natural = self._member_size_align(mem)
                align = min(natural, pack) if pack is not None else natural
                offset = _roundup(offset, align)
                if mem["arr"] is not None:
                    size *= self.p.eval_const(mem["arr"])
                offsets[mem["name"]] = (offset, size)
                offset += size
                max_natural = max(max_natural, natural)
        finally:
            self._visiting.discard(name)
        align = min(pack, max_natural) if pack is not None else max_natural
        if d["alignas"]:
            align = max(align, d["alignas"])
        size = _roundup(offset, align)
        result = (size, align, offsets)
        self._cache[name] = result
        return result

    def _member_size_align(self, mem):
        if mem["ptr"]:
            return 8, 8
        bare = mem["type"].split("::")[-1]
        if bare in PRIM_TYPES:
            return PRIM_TYPES[bare]
        if bare in self.p.structs:
            s, a, _ = self.struct_layout(bare)
            return s, a
        raise LayoutError("未知成员类型 '%s' (字段 %s)" % (mem["type"], mem["name"]))


def resolve_cann_root(explicit=None):
    """按 set_env.sh 环境定位 CANN 根目录与 kernel_tiling.h。"""
    cands = []
    if explicit:
        cands.append(explicit)
    for var in CANN_ENV_VARS:
        v = os.environ.get(var)
        if v:
            cands.append(v)
    tried = list(cands)
    for c in cands:
        for arch in ("x86_64-linux", "aarch64-linux"):
            p = os.path.join(c, arch, "include", "kernel_tiling", "kernel_tiling.h")
            if os.path.isfile(p):
                return c, p
        hits = sorted(
            glob.glob(
                os.path.join(c, "*", "include", "kernel_tiling", "kernel_tiling.h")
            )
        )
        if hits:
            return c, hits[0]
    raise LayoutError("未找到 CANN kernel_tiling.h。已尝试: %s" % (tried,))


def _cann_sizes(cann):
    """从 kernel_tiling.h 解析 Mc2InitTiling/Mc2CcTiling/TCubeTiling 尺寸。"""
    _, header = resolve_cann_root(cann)
    parser = HeaderParser()
    parser.parse_file(header)
    comp = LayoutComputer(parser)
    return (
        header,
        comp.struct_layout("Mc2InitTiling")[0],
        comp.struct_layout("Mc2CcTiling")[0],
        comp.struct_layout("TCubeTiling")[0],
    )


def _base_layout(cann):
    """公共基础布局: CANN 三个尺寸 (含兜底) + RCSTiling + Workspace 常量。"""
    try:
        header, init_size, cc_size, tcube_size = _cann_sizes(cann)
        headers = [header]
    except LayoutError as e:
        init_size = DEFAULT_MC2_INIT_TILING_SIZE
        cc_size = DEFAULT_MC2_CC_TILING_SIZE
        tcube_size = DEFAULT_TCUBE_TILING_SIZE
        if os.environ.get("MC2_LAYOUT_QUIET_FALLBACK") != "1":
            logging.warning("无法从 CANN 环境获取 kernel_tiling.h (%s)", e)
            logging.warning(
                "使用默认尺寸继续: Mc2InitTiling=%d, Mc2CcTiling=%d, TCubeTiling=%d "
                "(9.1.0/9.2.0 实测值; 若 dump 来自其他版本 CANN, 偏移可能不准, "
                "请 source 对应 set_env.sh)",
                init_size,
                cc_size,
                tcube_size,
            )
        headers = ["<默认值 (未找到 CANN 头文件)>"]

    rcs_offsets = {}
    off = 0
    for name, size in RCS_MEMBERS:
        off = _roundup(off, size)
        rcs_offsets[name] = (off, size)
        off += size
    rcs_size = _roundup(off, 8)
    return {
        "headers": headers,
        "initSize": init_size,
        "ccSize": cc_size,
        "tcubeTilingSize": tcube_size,
        "rcs": {"size": rcs_size, "offsets": rcs_offsets},
        "workspace": {
            "segInfoSize": WS_SEG_INFO_SIZE,
            "layoutInfoSize": WS_LAYOUT_INFO_SIZE,
            "maxSegments": WS_MAX_SEGMENTS,
            "runtimeInfoPerCoreSize": RUNTIME_INFO_PER_CORE_SIZE,
            "runtimeInfoCoreMax": RUNTIME_INFO_CORE_MAX,
            "layoutOffset": DFX_WORKSPACE_LAYOUT_OFFSET,
            "peermemDataSizeOffset": PEERMEM_DATA_SIZE_OFFSET,
            "peermemLayoutOffset": PEERMEM_LAYOUT_OFFSET,
            "dfxDumpInfoSize": DFX_DUMP_INFO_SIZE,
        },
    }


def _self_check(layout, ground_truth):
    """三个 CANN 基础尺寸均为当前默认值时，对拍源码 ground truth。"""
    if (
        layout["tcubeTilingSize"] != DEFAULT_TCUBE_TILING_SIZE
        or layout.get("initSize") != DEFAULT_MC2_INIT_TILING_SIZE
        or layout.get("ccSize") != DEFAULT_MC2_CC_TILING_SIZE
    ):
        return
    for path, expect in ground_truth.items():
        node = layout
        try:
            for key in path.split("."):
                node = node[key]
        except (KeyError, TypeError):
            raise LayoutError(
                "布局公式自检失败: 路径 %s 取值失败 (DFX 表或组装公式有误)" % path
            )
        if node != expect:
            raise LayoutError(
                "布局公式自检失败: %s 期望 %d, 实际 %d (DFX 表或组装公式有误)"
                % (path, expect, node)
            )


def compute_layout(cann=None):
    """计算 AllGatherMatmulV2 DFX TilingData 布局。

    CANN 头文件不可用时告警并使用默认尺寸继续, 不抛异常;
    仅当公式自检失败 (bug) 时抛 LayoutError。"""
    base = _base_layout(cann)
    tcube_size = base["tcubeTilingSize"]

    # [Mc2InitTiling][Mc2CcTiling][DfxDumpInfo][RCSTiling][dataType][debugMode][3×MM tiling]
    off_init = 0
    off_cc = off_init + base["initSize"]
    off_dump_info = off_cc + base["ccSize"]
    off_param = off_dump_info + DFX_DUMP_INFO_SIZE
    off_data_type = off_param + base["rcs"]["size"]
    off_debug_mode = off_data_type + 4
    off_blocks = off_debug_mode + 4

    def build(block_size):
        return _roundup(off_blocks + 3 * block_size, 8)

    v2_size = build(tcube_size + MMV3_TAIL_SIZE)
    fp8_size = build(tcube_size + QUANT_FIXED_SIZE)

    head_offsets = {
        "dumpInfo": off_dump_info,
        "workspaceLayout": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
        "peermemDataSize": off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
        "peermemLayout": off_dump_info + PEERMEM_LAYOUT_OFFSET,
        "mc2InitTiling": off_init,
        "mc2CcTiling": off_cc,
        "param": off_param,
        "dataType": off_data_type,
        "debugMode": off_debug_mode,
        "blocks": off_blocks,
    }
    layout = {
        "headers": base["headers"],
        "initSize": base["initSize"],
        "ccSize": base["ccSize"],
        "tcubeTilingSize": tcube_size,
        "v2": {"size": v2_size, "offsets": dict(head_offsets)},
        "fp8": {"size": fp8_size, "offsets": dict(head_offsets)},
        "rcs": base["rcs"],
        "workspace": dict(
            base["workspace"],
            layoutOffset=off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            peermemDataSizeOffset=off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
            peermemLayoutOffset=off_dump_info + PEERMEM_LAYOUT_OFFSET,
        ),
    }
    _self_check(layout, _GROUND_TRUTH)
    return layout


def compute_mrs_v2_layout(cann=None):
    """计算 MatmulReduceScatterV2 TilingData 布局。

    结构: [Mc2InitTiling][Mc2CcTiling][DfxDumpInfo][RCSTiling]
          [dataType][debugMode][2×tiling 块]。"""
    base = _base_layout(cann)
    tcube_size = base["tcubeTilingSize"]

    off_init = 0
    off_cc = off_init + base["initSize"]
    off_dump_info = off_cc + base["ccSize"]
    off_param = off_dump_info + DFX_DUMP_INFO_SIZE
    off_data_type = off_param + base["rcs"]["size"]
    off_debug_mode = off_data_type + 4
    off_blocks = off_debug_mode + 4

    def build(block_size):
        return _roundup(off_blocks + 2 * block_size, 8)

    nq_size = build(tcube_size + MMV3_TAIL_SIZE)
    q_size = build(tcube_size + QUANT_FIXED_SIZE)

    layout = {
        "headers": base["headers"],
        "initSize": base["initSize"],
        "ccSize": base["ccSize"],
        "tcubeTilingSize": tcube_size,
        "nonQuant": {
            "size": nq_size,
            "wsLayoutOffset": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "blockOffset": off_blocks,
        },
        "quant": {
            "size": q_size,
            "wsLayoutOffset": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "blockOffset": off_blocks,
        },
        "offsets": {
            "dumpInfo": off_dump_info,
            "workspaceLayout": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "peermemDataSize": off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
            "peermemLayout": off_dump_info + PEERMEM_LAYOUT_OFFSET,
            "mc2InitTiling": off_init,
            "mc2CcTiling": off_cc,
            "param": off_param,
            "dataType": off_data_type,
            "debugMode": off_debug_mode,
            "blocks": off_blocks,
        },
        "rcs": base["rcs"],
        "workspace": dict(
            base["workspace"],
            layoutOffset=off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            peermemDataSizeOffset=off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
            peermemLayoutOffset=off_dump_info + PEERMEM_LAYOUT_OFFSET,
        ),
    }
    _self_check(layout, _GT_MRSV2)
    return layout


def compute_allto_all_layout(cann=None):
    """计算 AlltoAllMatmul (v1) TilingData 布局 (nonQuant/quant/apace 三分支)。

    主线结构: [Mc2InitTiling][Mc2CcTiling][DfxDumpInfo][AlltoAllMatmulTilingInfo]
              [2×tiling 块]
    apace 结构 (无 TCubeTiling, 固定):
              [Mc2InitTiling][Mc2CcTiling][DfxDumpInfo][CommTiling 40]
              [QuantMatmul 64][localMatmul 4]"""
    base = _base_layout(cann)
    tcube_size = base["tcubeTilingSize"]

    off_init = 0
    off_cc = off_init + base["initSize"]
    off_dump_info = off_cc + base["ccSize"]
    off_info = off_dump_info + DFX_DUMP_INFO_SIZE
    head = off_info + ALLTO_ALL_TILING_INFO_SIZE

    def build(block_size):
        return _roundup(head + 2 * block_size, 8)

    nq_size = build(tcube_size + MMV3_TAIL_SIZE)
    q_size = build(tcube_size + QUANT_FIXED_SIZE)

    # apace 固定布局
    apace_comm = off_info
    apace_qmm = apace_comm + APACE_COMM_TILING_SIZE
    apace_local = apace_qmm + APACE_QUANT_MATMUL_SIZE

    layout = {
        "headers": base["headers"],
        "initSize": base["initSize"],
        "ccSize": base["ccSize"],
        "tcubeTilingSize": tcube_size,
        "nonQuant": {
            "size": nq_size,
            "wsLayoutOffset": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "blockSize": tcube_size + MMV3_TAIL_SIZE,
            "tileOffset": head,
            "tailOffset": head + tcube_size + MMV3_TAIL_SIZE,
        },
        "quant": {
            "size": q_size,
            "wsLayoutOffset": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "blockSize": tcube_size + QUANT_FIXED_SIZE,
            "tileOffset": head,
            "tailOffset": head + tcube_size + QUANT_FIXED_SIZE,
        },
        "apace": {
            "size": _roundup(apace_local + 4, 8),
            "wsLayoutOffset": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "commOffset": apace_comm,
            "quantMatmulOffset": apace_qmm,
            "localMatmulOffset": apace_local,
        },
        "offsets": {
            "dumpInfo": off_dump_info,
            "workspaceLayout": off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            "peermemDataSize": off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
            "peermemLayout": off_dump_info + PEERMEM_LAYOUT_OFFSET,
            "mc2InitTiling": off_init,
            "mc2CcTiling": off_cc,
            "tilingInfo": off_info,
        },
        "workspace": dict(
            base["workspace"],
            layoutOffset=off_dump_info + DFX_WORKSPACE_LAYOUT_OFFSET,
            peermemDataSizeOffset=off_dump_info + PEERMEM_DATA_SIZE_OFFSET,
            peermemLayoutOffset=off_dump_info + PEERMEM_LAYOUT_OFFSET,
        ),
    }
    _self_check(layout, _GT_A2A)
    return layout

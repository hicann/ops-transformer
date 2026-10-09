#!/usr/bin/python3
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
把 xlsx 的 dts sheet 参数转换为
quant_flash_attn_paramset_debug.py 需要的 TEST_PARAMS 格式。

环境：无 openpyxl/pandas，xlsx 解析用本文件内置的极简读取器
（zipfile + minidom，自 tests/assets/_xlsx_minireader.py 提取所需函数，
避免目录间路径/import 依赖）。

输入 xlsx 路径必填（--xlsx，不做默认/不写死文件名）；
默认输出本脚本同目录下的 quant_flash_attn_paramset_debug.py。

用法：
  python3 excel_to_pytest_paramset.py --xlsx redline.xlsx   # 读 dts sheet, 写 debug paramset 文件
  python3 excel_to_pytest_paramset.py --xlsx xx.xlsx --sheet mxfp8d=72
  python3 excel_to_pytest_paramset.py --xlsx xx.xlsx --dry-run   # 只打印生成结果, 不改文件

字段映射（Excel colon -> debug paramset key）：
  B                 -> B
  Q_N / num_heads_q -> N_q
  KV_N/num_heads_kv -> N_kv
  Q_D / head_dim    -> D
  cu_seqlens_q_value-> cu_seqlens_q
  cu_seqlens_kv_value-> cu_seqlens_kv
  seqused_q_value   -> seqused_q
  seqused_kv_value  -> seqused_kv
  max_seqlen_q      -> max_seqlen_q
  max_seqlen_kv     -> max_seqlen_kv
  layout_kv 前缀 PA_ -> enable_pa (True/False)
  layout_kv         -> kv_cache_layout
  blocksize         -> block_size
  mask_mode         -> mask_mode
  layout_q_descale  -> q_scale_layout
  p_scale_value     -> p_scale (空则默认 1.0)
  return_softmax_lse-> enable_lse

格式约定（对齐 quant_flash_attn_paramset_debug.py 现有写法）：
  - 每个 key 的值用单元素 list 包裹:
      标量:int/float/bool/str -> [<值>]; 列表(如 cu_seqlens) -> [[...]]; 缺失 -> [None]
  - list 型参数 (cu_seqlens_*/seqused_*) 缺失时写 [None]。
"""

import argparse
import os
import re
import zipfile
import xml.dom.minidom as minidom

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SHEET = "dts"
_DEFAULT_OUT = os.path.join(_HERE, "quant_flash_attn_paramset_debug.py")


# ---------------------------------------------------------------------------
# 内置极简 xlsx 读取器 (自 tests/assets/_xlsx_minireader.py 提取所需函数),
# 避免与 assets 目录之间产生路径/import 依赖。zipfile + minidom, 无 openpyxl。
# ---------------------------------------------------------------------------
_CELL_RE = re.compile(r"^([A-Z]+)(\d+)$")


def col_to_idx(col):
    """列字母 → 0-based 列索引（A=0 ... Z=25, AA=26, AB=27 ...）"""
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def _parse_ref(ref):
    """把 Excel 单元格引用（如 "AB12"）拆成 (col_idx, row_idx)。"""
    m = _CELL_RE.match(ref)
    if not m:
        raise ValueError(f"bad cell ref: {ref!r}")
    return col_to_idx(m.group(1)), int(m.group(2)) - 1


def _cell_text(cell, shared_strings):
    """取一个 <c> 元素的文本值。返回 None 表示空单元格。"""
    t = cell.getAttribute("t")
    v_els = cell.getElementsByTagName("v")
    if t == "s" and v_els:
        idx = int(v_els[0].firstChild.data)
        return shared_strings[idx]
    if t == "b" and v_els:
        return "True" if v_els[0].firstChild.data == "1" else "False"
    if t == "inlineStr":
        is_els = cell.getElementsByTagName("is")
        if not is_els:
            return None
        ts = is_els[0].getElementsByTagName("t")
        return "".join(tt.firstChild.data if tt.firstChild else "" for tt in ts)
    if v_els:
        return v_els[0].firstChild.data
    return None


def _resolve_sheet_xml(z, sheet):
    """把 sheet 标识 (int 序号或 str 名) 解析为 xl/worksheets/sheetN.xml 路径。"""
    if isinstance(sheet, int):
        return f"xl/worksheets/sheet{sheet}.xml"

    wb_xml = z.read("xl/workbook.xml").decode("utf-8")
    wb_doc = minidom.parseString(wb_xml)
    rid = None
    for s in wb_doc.getElementsByTagName("sheet"):
        if s.getAttribute("name") == sheet:
            rid = s.getAttribute("r:id") or s.getAttribute(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            break
    if rid is None:
        names = [s.getAttribute("name") for s in wb_doc.getElementsByTagName("sheet")]
        raise ValueError(
            f"sheet name {sheet!r} not found in workbook. available: {names}"
        )

    rels_xml = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rels_doc = minidom.parseString(rels_xml)
    for rel in rels_doc.getElementsByTagName("Relationship"):
        if rel.getAttribute("Id") == rid:
            target = rel.getAttribute("Target")
            return target.lstrip("/") if target.startswith("/") else "xl/" + target
    raise ValueError(f"r:id {rid!r} (sheet {sheet!r}) not found in workbook.xml.rels")


def _parse_sheet_xml(sheet_xml, shared_strings):
    """解析单个 sheet xml 为 list[dict[col_idx]→text]"""
    sh = minidom.parseString(sheet_xml)
    rows = sh.getElementsByTagName("row")
    rows = sorted(rows, key=lambda r: int(r.getAttribute("r")))
    parsed_rows = []
    for row in rows:
        cells = {}
        for c in row.getElementsByTagName("c"):
            col_idx, _row_idx = _parse_ref(c.getAttribute("r"))
            cells[col_idx] = _cell_text(c, shared_strings)
        parsed_rows.append(cells)
    return parsed_rows


def _load_sheet_openpyxl(xlsx_path, sheet=1):
    """openpyxl 回退路径, 支持 sheet 序号或名"""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    if isinstance(sheet, int):
        ws = wb.worksheets[sheet - 1] if 1 <= sheet <= len(wb.worksheets) else wb.active
    else:
        ws = wb[sheet]
    parsed_rows = []
    for row in ws.iter_rows():
        cells = {}
        for cell in row:
            v = cell.value
            if v is None:
                continue
            cells[cell.col_idx - 1] = str(v) if not isinstance(v, str) else v
        parsed_rows.append(cells)
    return parsed_rows


def load_sheet(xlsx_path, sheet=1):
    """读取指定 sheet。sheet 可为序号 (int, 1-based) 或 sheet 名 (str)。"""
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            ss_xml = z.read("xl/sharedStrings.xml").decode("utf-8")
            sheet_path = _resolve_sheet_xml(z, sheet)
            sheet_xml = z.read(sheet_path).decode("utf-8")
    except (KeyError, FileNotFoundError):
        return _load_sheet_openpyxl(xlsx_path, sheet)

    ss_doc = minidom.parseString(ss_xml)
    shared_strings = []
    for si in ss_doc.getElementsByTagName("si"):
        ts = si.getElementsByTagName("t")
        shared_strings.append(
            "".join(t.firstChild.data if t.firstChild else "" for t in ts)
        )
    return _parse_sheet_xml(sheet_xml, shared_strings)


def list_sheet_names(xlsx_path):
    """返回 xlsx 中所有 sheet 名 (供用户查看可选 sheet)"""
    try:
        with zipfile.ZipFile(xlsx_path) as z:
            wb_xml = z.read("xl/workbook.xml").decode("utf-8")
    except (KeyError, FileNotFoundError):
        import openpyxl

        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        return wb.sheetnames
    wb_doc = minidom.parseString(wb_xml)
    return [s.getAttribute("name") for s in wb_doc.getElementsByTagName("sheet")]


_FILE_HEADER = """\
#!/usr/bin/python3
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

from quant_flash_attn_paramset_common import expand_paramset_to_cases

"""

_FILE_FOOTER = """
CASES = expand_paramset_to_cases(TEST_PARAMS)
"""


# 每个目标参数的 Excel 列名（dts sheet 表头名，按首列匹配）
_FIELDS = [
    ("B", "B"),
    ("N_q", "Q_N"),
    ("N_kv", "KV_N"),
    ("D", "Q_D"),
    ("cu_seqlens_q", "cu_seqlens_q_value"),
    ("cu_seqlens_kv", "cu_seqlens_kv_value"),
    ("seqused_q", "seqused_q_value"),
    ("seqused_kv", "seqused_kv_value"),
    ("max_seqlen_q", "max_seqlen_q"),
    ("max_seqlen_kv", "max_seqlen_kv"),
    ("enable_pa", "layout_kv"),
    ("kv_cache_layout", "layout_kv"),
    ("block_size", "blocksize"),
    ("mask_mode", "mask_mode"),
    ("q_scale_layout", "layout_q_descale"),
    ("p_scale", "p_scale_value"),
    ("enable_lse", "return_softmax_lse"),
]

# list 型参数：值为 int 列表, 缺失写 None
_LIST_FIELDS = {"cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"}

# 缺失 p_scale 时的默认值（dts sheet 的 p_scale_value 列为空, "1" 误写在 shape 列）
_P_SCALE_DEFAULT = 1.0


class _Columns:
    """按表头名查列索引, 重复列名取首个（dts sheet 的 testcase_name 出现两次）。"""

    def __init__(self, header_row: dict):
        self._by_name = {}
        for col_idx, text in sorted(header_row.items()):
            if text is None:
                continue
            key = str(text).strip()
            if key and key not in self._by_name:
                self._by_name[key] = col_idx

    def get(self, name):
        return self._by_name.get(name)


def _strip(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


def _to_int(s):
    s = _strip(s)
    if s is None:
        return None
    return int(float(s))


def _to_float(s):
    s = _strip(s)
    if s is None:
        return None
    return float(s)


def _to_int_list(s):
    s = _strip(s)
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(float(p)) for p in parts]


def _to_bool(s):
    s = _strip(s)
    if s is None:
        return None
    return s in ("True", "TRUE", "1", "true")


def _parse_row(row, cols):
    """把一行 excel 数据解析成 Debug paramset 的参数字典（值未转成 list 形式）。

    返回 dict[param_key -> python 值]；list 型缺失 → None，p_scale 空 → 1.0。
    """
    params = {"B": _to_int(row.get(cols.get("B")))}
    params["N_q"] = _to_int(row.get(cols.get("Q_N")))
    params["N_kv"] = _to_int(row.get(cols.get("KV_N")))
    params["D"] = _to_int(row.get(cols.get("Q_D")))

    params["cu_seqlens_q"] = _to_int_list(row.get(cols.get("cu_seqlens_q_value")))
    params["cu_seqlens_kv"] = _to_int_list(row.get(cols.get("cu_seqlens_kv_value")))
    params["seqused_q"] = _to_int_list(row.get(cols.get("seqused_q_value")))
    params["seqused_kv"] = _to_int_list(row.get(cols.get("seqused_kv_value")))

    params["max_seqlen_q"] = _to_int(row.get(cols.get("max_seqlen_q")))
    params["max_seqlen_kv"] = _to_int(row.get(cols.get("max_seqlen_kv")))

    layout_kv = _strip(row.get(cols.get("layout_kv")))
    params["enable_pa"] = bool(layout_kv and layout_kv.startswith("PA_"))
    params["kv_cache_layout"] = layout_kv
    params["block_size"] = _to_int(row.get(cols.get("blocksize")))
    params["mask_mode"] = _to_int(row.get(cols.get("mask_mode")))
    params["q_scale_layout"] = _strip(row.get(cols.get("layout_q_descale")))

    p_scale = _to_float(row.get(cols.get("p_scale_value")))
    params["p_scale"] = _P_SCALE_DEFAULT if p_scale is None else p_scale

    params["enable_lse"] = bool(
        _to_bool(row.get(cols.get("return_softmax_lse")))
        if _strip(row.get(cols.get("return_softmax_lse"))) is not None
        else False
    )
    return params


def _render_value(key, value):
    """把一个参数值渲染成 debug paramset 里单元素 list 的 Python 字面量。"""
    if key in _LIST_FIELDS:
        if value is None:
            return "[None]"
        return "[" + "[" + ", ".join(str(v) for v in value) + "]" + "]"
    if isinstance(value, bool):
        return "[True]" if value else "[False]"
    if isinstance(value, str):
        return f'["{value}"]'
    if isinstance(value, float):
        # 保留浮点值, 整数值写成 x.0 与现有格式一致
        if value == int(value):
            return f"[{value:.1f}]"
        return f"[{value!r}]"
    return f"[{value}]"


def build_test_params(data_rows, cols):
    """把所有数据行转成 TEST_PARAMS dict[case_name -> 参数字典]（值已渲染为字符串）。"""
    test_params = {}
    for row in data_rows:
        name = _strip(row.get(cols.get("testcase_name")))
        if not name:
            continue  # 跳过 testcase_name 为空的整行
        parsed = _parse_row(row, cols)
        test_params[name] = {key: _render_value(key, parsed[key]) for key, _ in _FIELDS}
    return test_params


def render_file(test_params):
    """把 TEST_PARAMS dict 渲染成完整 paramset 源文件内容。"""
    lines = [_FILE_HEADER, "TEST_PARAMS = {\n"]
    for name, params in test_params.items():
        lines.append(f'    "{name}": {{\n')
        for key, val in params.items():
            lines.append(f'        "{key}": {val},\n')
        lines.append("    },\n")
    lines.append("}\n")
    lines.append(_FILE_FOOTER)
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="xlsx dts sheet -> quant_flash_attn_paramset_debug.py 格式"
    )
    parser.add_argument(
        "--xlsx",
        required=True,
        help="输入 xlsx 路径 (必填)",
    )
    parser.add_argument(
        "--sheet",
        default=_DEFAULT_SHEET,
        help=f"要转换的 sheet 名/序号 (默认 {_DEFAULT_SHEET!r})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUT,
        help="输出 paramset 文件路径 (默认 quant_flash_attn_paramset_debug.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印生成结果, 不写文件",
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="列出 --xlsx 中所有 sheet 名后退出",
    )
    args = parser.parse_args()

    if args.list_sheets:
        for i, nm in enumerate(list_sheet_names(args.xlsx), 1):
            print(f"  [{i}] {nm}")
        return

    print(f"[excel_to_pytest_paramset] reading sheet={args.sheet!r} from {args.xlsx}")
    rows = load_sheet(args.xlsx, args.sheet)
    if not rows:
        raise RuntimeError(f"xlsx sheet {args.sheet!r} has no rows")
    header_row, data_rows = rows[0], rows[1:]

    cols = _Columns(header_row)
    if cols.get("testcase_name") is None:
        raise RuntimeError("sheet has no testcase_name column")

    test_params = build_test_params(data_rows, cols)
    if not test_params:
        raise RuntimeError(f"sheet {args.sheet!r} has no valid case rows")

    content = render_file(test_params)

    print(f"  generated {len(test_params)} case(s) (sheet rows: {len(data_rows)})")
    if args.dry_run:
        print(content)
        return

    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

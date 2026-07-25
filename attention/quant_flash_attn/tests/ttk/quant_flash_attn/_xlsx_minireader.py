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
"""极简 xlsx 读取器：zipfile + xml.dom.minidom，无 openPyXL/pandas 依赖。

供 excel_to_csv.py 使用。只读 xl/worksheets/sheet1.xml（mxfp8 sheet）+
xl/sharedStrings.xml，返回 list[dict[col_idx]→text]，空单元格 → None。
"""

import re
import zipfile
import xml.dom.minidom as minidom

_CELL_RE = re.compile(r"^([A-Z]+)(\d+)$")


def col_to_idx(col: str) -> int:
    """列字母 → 0-based 列索引（A=0 ... Z=25, AA=26, AB=27 ...）"""
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def _parse_ref(ref: str):
    """把 Excel 单元格引用（如 "AB12"）拆成 (col_idx, row_idx)。"""
    m = _CELL_RE.match(ref)
    if not m:
        raise ValueError(f"bad cell ref: {ref!r}")
    return col_to_idx(m.group(1)), int(m.group(2)) - 1


def _cell_text(cell, shared_strings):
    """取一个 <c> 元素的文本值。返回 None 表示空单元格。

    - t="s" + <v>i</v>      → shared_strings[i]
    - t="inlineStr" + <is><t> → inline string
    - t="b" + <v>0/1</v>    → bool 字符串 "True"/"False"
    - <v>num</v>            → 数值字符串（保留 Excel 原样，避免精度丢失）
    - <f>...</f> + <v>      → 公式缓存值，取 <v>
    - 空                    → None
    """
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
        # 数值或公式缓存值
        return v_els[0].firstChild.data
    return None


def load_sheet1(xlsx_path: str):
    """读 xlsx 的 sheet1（按行号排序），返回 list[dict[col_idx]→text]。

    空单元格不出现在 dict 里，调用方用 dict.get(idx) 得 None。
    """
    with zipfile.ZipFile(xlsx_path) as z:
        ss_xml = z.read("xl/sharedStrings.xml").decode("utf-8")
        sheet_xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8")

    ss_doc = minidom.parseString(ss_xml)
    shared_strings = []
    for si in ss_doc.getElementsByTagName("si"):
        ts = si.getElementsByTagName("t")
        shared_strings.append(
            "".join(t.firstChild.data if t.firstChild else "" for t in ts)
        )

    sh = minidom.parseString(sheet_xml)
    rows = sh.getElementsByTagName("row")
    rows = sorted(rows, key=lambda r: int(r.getAttribute("r")))
    parsed_rows = []
    for row in rows:
        cells = {}
        for c in row.getElementsByTagName("c"):
            ref = c.getAttribute("r")
            col_idx, _row_idx = _parse_ref(ref)
            cells[col_idx] = _cell_text(c, shared_strings)
        parsed_rows.append(cells)
    return parsed_rows

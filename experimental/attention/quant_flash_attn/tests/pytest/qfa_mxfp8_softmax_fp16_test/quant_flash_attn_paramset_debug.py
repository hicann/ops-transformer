#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 调试用例集（当前为空）
# =====================================================================
# 修复存档（本文件历史上承载的 FAIL/探索用例及根因，均已修复并收编 func_rdv）
# =====================================================================
# ① 2026-09-24 S1 非对齐·V1 部分列（QS192/QS444）：LoadPToL0B 呈现宽度契约修复
# ② 2026-09-24 QS1_KVS832 C2_V2 释放竞态：非末块释放改挂 PIPE_V
# ③ 2026-09-24 S2 任意长度（KVS100/65/388/556）：MmadMx k 对齐 64 + P 零填充 + softmax 尾行链
# ④ 2026-09-24 per-batch seqused：golden 逐 batch 循环 + 对比无效行掩码（后由 ClearOutput 取代）
# ⑤ 2026-09-27 S2=1 边界系列 6 用例：kernel 零修改即正确，迁入 func_rdv
# ⑥ 2026-09-27 seq_used=0 空批系列 8 用例：golden 空批保护 + kernel 全量预清零（对齐主线
#    ClearOutput）后 8/8 全过，迁入 func_rdv
# ⑦ 2026-09-27 边值 fuzz 扫描 565 用例（S1/S2 边值、G/B 扫描、零长模式、维度交叉）：
#    暴露 metadata BNSD 变长 padding 漏判（Qstep/Qmix 10 用例失败）→ 补判定后全过；
#    全量收编 func_rdv 全维度矩阵（每用例 LSE/非LSE 双版本）
# ⑧ 2026-09-27 softmaxLse 规格验证 11 用例：VfCalcLse 落地 + 空批 -inf 语义，收编 func_rdv
# =====================================================================
CASES = []

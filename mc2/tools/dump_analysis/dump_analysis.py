#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file dump_analysis.py
#  \brief

import os
import sys
import csv
import ast
import logging
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
import dump_analysis_A2 as A2

SOC_VERSION_950 = "950"
SOC_VERSION_910_93 = "910_93"
SOC_VERSION_910B = "910B"

csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


@dataclass
class WinData:
    win_data_list_01: np.ndarray | list
    win_data_list_02: np.ndarray | list
    win_data_01: int
    win_data_02: int
    card_num_class: int
    bs_class: int
    k_class: int


# 根据文件名判断挂在dispatch还是combine
def check_dis_com_version(target_path: str) -> str:
    dis_com_func = "None"
    dis_com_version_func = "None"
    for filename_func in os.listdir(os.path.join(target_path)):
        if filename_func.lower().endswith("host.o"):
            if "V3" in filename_func:
                logging.info("1.1 根据文件名%s,识别版本为 V3", filename_func)
                dis_com_version_func = "V3"
            if "dispatch" in filename_func.lower():
                logging.info(
                    "1.1 根据dump数据文件名%s,判断该卡挂在dispatch", filename_func
                )
                dis_com_func = "dispatch"
            elif "combine" in filename_func.lower():
                logging.info(
                    "1.1 根据dump数据文件名%s,判断该卡挂在combine", filename_func
                )
                dis_com_func = "combine"
            else:
                logging.warning(
                    "1.1 根据dump数据文件名%s,该卡没有挂在dispatch/combine算子上",
                    filename_func,
                )
                break
    return dis_com_func, dis_com_version_func


def check_mask(mask, moe_num_func: int, expert_ids_reshape):
    if np.any(mask):
        row_func, col_func = np.where(mask)
        logging.error(
            "1.3 检测到expertids中有%d个异常输入, expertids中的值应该大于0 且小于moe专家数%d",
            len(row_func),
            moe_num_func,
        )
        for r, c in zip(row_func, col_func):
            logging.error(
                "1.3 expertids中下标[%d,%d]的值:%d超范围",
                r,
                c,
                expert_ids_reshape[r][c],
            )
    else:
        logging.info("1.3 未检测到expertids中有超范围的异常输入")


def check_duplicate_per_row(expert_ids_reshape_func):
    has_duplicate = False
    batch_size, topk = expert_ids_reshape_func.shape

    for row_idx in range(batch_size):
        per_expertids = expert_ids_reshape_func[row_idx]
        # 检查这一行是否有重复数字
        if len(np.unique(per_expertids)) == len(per_expertids):
            continue  # 无重复，直接跳过

        # 有重复
        has_duplicate = True
        seen = set()
        duplicates = set()
        for num in per_expertids:
            if num in seen:
                duplicates.add(num)
            seen.add(num)
        logging.error(
            "1.3 第[%s]行出现重复expert_id: %s, 重复的值=%s",
            row_idx,
            list(per_expertids),
            list(duplicates),
        )

    if not has_duplicate:
        logging.info("1.3 所有行均无重复expert_id")


# 判断topk是否超范围(<0 or >moe专家数)
def check_topk(
    target_path: str,
    moe_num_func: int,
    bs_func: int,
    sp_moe_num_func: int,
    dis_com_version_func: str,
):
    k_func = 0
    if dis_com_version_func == "V3":
        target_filie_num = "input.2.bin"
    else:
        target_filie_num = "input.1.bin"

    for filename_func in os.listdir(os.path.join(target_path)):
        if target_filie_num in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            expert_ids = np.fromfile(file_path_func, dtype=np.int32)
            k_func = int(expert_ids.shape[0] / bs_func)
            expert_ids_reshape = expert_ids.reshape(bs_func, k_func)
            logging.info("1.3 该卡的输入expertids为%s\n", expert_ids_reshape)
            mask = (expert_ids_reshape < 0) | (
                expert_ids_reshape >= (moe_num_func + sp_moe_num_func)
            )
            check_mask(mask, moe_num_func, expert_ids_reshape)
            check_duplicate_per_row(expert_ids_reshape)
            return k_func, expert_ids_reshape
    logging.warning(
        "1.3 该卡未发现输入expertids对应的%s文件, 无法分析输入expertids",
        target_filie_num,
    )
    return k_func, np.array([])


# 获取a
def get_a(target_path: str, dis_com_version_func: str):
    a_func = 0
    if dis_com_version_func == "V3":
        target_filie_num = "input.3.bin"
    else:
        target_filie_num = "input.2.bin"

    for filename_func in os.listdir(os.path.join(target_path)):
        if target_filie_num in filename_func:
            assist_info_for_combine_path = os.path.join(target_path, filename_func)
            assist_info_for_combine = np.fromfile(
                assist_info_for_combine_path, dtype=np.int32
            )
            a_func = int(assist_info_for_combine.shape[0] / 128)
            break
    return a_func


# 获取h
def get_h(
    target_path: str, a_func: int, tp_worldsize_func: int, dis_com_version_func: str
):
    h_func = 0
    if dis_com_version_func == "V3":
        target_filie_num = "input.1.bin"
    else:
        target_filie_num = "input.0.bin"

    for filename_func in os.listdir(os.path.join(target_path)):
        if target_filie_num in filename_func:
            expands_path = os.path.join(target_path, filename_func)
            expands = np.fromfile(expands_path, dtype=np.int16)
            h_func = int(expands.shape[0] / (tp_worldsize_func * a_func))
            break
    return h_func


# 获取本卡专家数
def get_local_expert_num(
    moe_num_func: int,
    share_expert_card_count_func: int,
    card_num_func: int,
    all_card_num_func: int,
):
    if (card_num_func + 1) <= share_expert_card_count_func:
        logging.info("1.2 该卡为共享专家卡")
        local_expert_num_func = 1  # 共享专家卡:本卡专家数 = 1
        logging.info("1.2 本卡专家数为:%d", local_expert_num_func)
    else:
        logging.info("1.2 该卡不为共享专家卡")
        # 非共享专家卡:本卡专家数 = moe专家数 / moe卡数
        local_expert_num_func = int(
            moe_num_func / (all_card_num_func - share_expert_card_count_func)
        )
        logging.info("1.2 本卡专家数为:%d", local_expert_num_func)
    return local_expert_num_func


# 根据expertids计算epsendcnt
def count_all_card_epsendcnt(all_card_expertids_func, all_local_expetrnum_func):
    cards = sorted(all_card_expertids_func.keys(), key=lambda x: int(x))
    current_id = 0
    card_id_experts = {}
    for card in cards:
        num = all_local_expetrnum_func[card]
        card_id_experts[card] = list(range(current_id, current_id + num))
        current_id += num
    all_expertids = []
    for per_expertids in all_card_expertids_func.values():
        for arr_func in per_expertids:
            all_expertids.extend(arr_func)
    result = {}
    for card in cards:
        expert_idx = card_id_experts[card]
        count_list = [all_expertids.count(e) for e in expert_idx]
        cumulative = []
        total = 0
        for num in count_list:
            total += num
            cumulative.append(total)
        result[card] = cumulative
    return result


def check_epsendcnt(epsendcnt_count_func, epsendcnt_dump_func):
    for card in sorted(epsendcnt_count_func.keys(), key=lambda x: int(x)):
        if len(epsendcnt_dump_func[card]) == 0:
            logging.info(
                "卡%s 算子执行流程未卡死在combine算子,不进行epsendcnt对比", card
            )
            continue
        logging.info("卡%s 计算得出的epsendcnt为:%s", card, epsendcnt_count_func[card])
        logging.info(
            "卡%s dump数据获取的epsendcnt为:%s", card, epsendcnt_dump_func[card]
        )
        if len(epsendcnt_count_func[card]) != len(epsendcnt_dump_func[card]):
            logging.error(
                "7. 该卡根据输入expertids计算得到的epsendcnt的大小:%d与dump数据取得的epsendcnt的大小:%d不同",
                len(epsendcnt_count_func[card]),
                len(epsendcnt_dump_func[card]),
            )
            continue
        if epsendcnt_count_func[card] != epsendcnt_dump_func[card]:
            logging.error(
                "7 该卡根据输入expertids计算得到的epsendcnt的值与dump数据取得的epsendcnt的值不同"
            )
        else:
            logging.info(
                "7. 该卡根据输入expertids计算得到的epsendcnt的值与dump数据取得的epsendcnt的值相同"
            )


# 从input.3.bin中获取epsendcnt
def get_dump_epsendcnt(target_path: str, dis_com_version_func: str):
    if dis_com_version_func == "V3":
        target_filie_num = "input.4.bin"
    else:
        target_filie_num = "input.3.bin"

    epsendcnt_input = np.array([], dtype=np.int32)
    for filename_func in os.listdir(os.path.join(target_path)):
        if target_filie_num in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            epsendcnt_input = np.fromfile(file_path_func, dtype=np.int32)
            return epsendcnt_input
    return epsendcnt_input


# 获取该卡dump数据的从input.2.bin中获取expandidx
def get_dump_expandidx(target_path: str, dis_com_version_func: str):
    expandidx = []
    if dis_com_version_func == "V3":
        target_filie_num = "input.3.bin"
    else:
        target_filie_num = "input.2.bin"

    for filename_func in os.listdir(os.path.join(target_path)):
        if target_filie_num in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            expandidx_dump = np.fromfile(file_path_func, dtype=np.int32).tolist()
            expandidx_dump_reshape = [
                expandidx_dump[i : i + 3] for i in range(0, len(expandidx_dump), 3)
            ]
            return expandidx_dump_reshape
    return expandidx


# 识别使用了多少个核,通过查看每512B的前9*4B的位置是否全为0来判断
def analysis_core_num(arr_func: np.ndarray) -> int:
    per_core = 128  # 512B转为int32
    judge_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 标识区内全为0,识别是否走到最后一个核
    card_num_func = 0
    for i in range(100):
        if (
            judge_arr == arr_func[i * per_core : (len(judge_arr) + i * per_core)]
        ).all():
            card_num_func = i
            break
    return card_num_func


# 识别是否跑了单算子,识别dis&com使用核数是否相同，将dis的使用核数与com的使用核数对比
def compare_core_num(
    dis_core_num_func: int, com_core_num_func: int, card_num_func: int
) -> dict:
    core_error_dict = {}
    logging.info(
        "1. 该卡dispatch&combine使用核数为 dispatch:%d, combine:%d",
        dis_core_num_func,
        com_core_num_func,
    )
    if dis_core_num_func == 0 or com_core_num_func == 0:
        logging.info("1. 仅跑了单个算子")
        return core_error_dict
    else:
        if dis_core_num_func != com_core_num_func:
            core_error_dict[f"d{card_num_func}"] = (
                f"dispatch的使用核数:{dis_core_num_func} 与 combine的使用核数:{com_core_num_func} 应该相等"
            )
            logging.error(
                "dispatch的使用核数:%d 与 combine的使用核数:(%d) 应该相等",
                dis_core_num_func,
                com_core_num_func,
            )
            return core_error_dict
        else:
            return core_error_dict


# 获取每个核的执行次数，将出现次数最多的执行次数作为该卡的执行次数，并将与该卡的执行次数不同的核记录下来
def analysis_run_num(arr_func: np.ndarray, card_num_func: int):
    per_core = 128
    run_num_list = []
    diff_indices = []  # 存放与大多数执行次数不同的核的下标
    for i in range(card_num_func):
        run_num_list.append(arr_func[6 + (i * per_core)])
    run_num_count = Counter(run_num_list)
    max_run_num = max(run_num_count, key=run_num_count.get)
    diff_indices = [idx for idx, val in enumerate(run_num_list) if val != max_run_num]
    return run_num_list, diff_indices, max_run_num


# 对比com&dis的执行次数, 并将与次数最多的执行次数不同的核打印，并根据dsi&com的执行次数关系给出结论
def compare_run_num(
    parms: WinData, dis_diff_indices_func: list, com_diff_indices_func: list, d_c: str
) -> dict:
    dis_run_num_list_func = parms.win_data_list_01
    com_run_num_list_func = parms.win_data_list_02
    max_dis_run_num_func = parms.win_data_01
    max_com_run_num_func = parms.win_data_02
    card_num_func = parms.card_num_class
    run_num_error_dict = {}
    dis_max_run_num = len(dis_run_num_list_func) - len(dis_diff_indices_func)
    com_max_run_num = len(com_run_num_list_func) - len(com_diff_indices_func)
    if len(dis_diff_indices_func) != 0:
        for x in dis_diff_indices_func:
            run_num_error_dict[f"d{card_num_func}_{x}core_dispatch"] = (
                f"{x} core run_num should be {max_dis_run_num_func}"
                f"({dis_max_run_num}/{len(dis_run_num_list_func)}) but got {dis_run_num_list_func[x]}"
            )
    if len(com_diff_indices_func) != 0:
        for x in com_diff_indices_func:
            run_num_error_dict[f"d{card_num_func}_{x}core_combine"] = (
                f"{x} core run_num should be {max_com_run_num_func}"
                f"({com_max_run_num}/{len(com_run_num_list_func)}) but got {com_run_num_list_func[x]}"
            )

    if max_dis_run_num_func == max_com_run_num_func:
        logging.warning(
            "2.2 dispatch执行次数:%d = combine执行次数:%d ,挂在combine上",
            max_dis_run_num_func,
            max_com_run_num_func,
        )
        run_num_error_dict[f"d{card_num_func}"] = (
            f"dispatch执行次数:{max_dis_run_num_func} = "
            f"combine执行次数:{max_dis_run_num_func},挂在combine上"
        )
        if d_c != "combine":
            logging.warning("2.2 根据文件判断该卡挂在%s上,与执行序结果分析不同")
    elif max_dis_run_num_func == (max_com_run_num_func + 1):
        logging.warning(
            "2.2 dispatch执行次数:%d = combine执行次数:%d + 1,挂在dispatch上",
            max_dis_run_num_func,
            max_com_run_num_func,
        )
        run_num_error_dict[f"d{card_num_func}"] = (
            f"dispatch执行次数:{max_dis_run_num_func} = "
            f"combine执行次数:{max_dis_run_num_func} + 1,挂在diapatch上"
        )
        if d_c != "dispatch":
            logging.warning("2.2 根据文件判断该卡挂在%s上,与执行序结果分析不同")
    else:
        logging.warning(
            "2.2 dispatch执行次数:%d, combine执行次数:%d 无法判断",
            max_dis_run_num_func,
            max_com_run_num_func,
        )
        run_num_error_dict[f"d{card_num_func}"] = (
            f"dispatch执行次数:{max_dis_run_num_func},combine执行次数:{max_dis_run_num_func},无法判断"
        )
        logging.warning("2.2 根据文件名判断该卡挂在文件 %s 对应的算子上", d_c)
    return run_num_error_dict


# 获取每个核的moe专家数,globalBs,将出现最多次数的moe专家数,globalBs作为该卡的moe专家数,globalBs
def get_moe_globalbs(arr_func: np.ndarray, card_num_func: int, d_c: str):
    per_core = 128
    moe_num = []
    globalbs_num = []
    for i in range(card_num_func):
        moe_num.append(arr_func[3 + (i * per_core)])
        globalbs_num.append(arr_func[5 + (i * per_core)])
    logging.info("1.1 %s各核moe专家数:%s\n", d_c, moe_num)
    logging.info("1.1 %s各核globalbs:%s\n", d_c, globalbs_num)
    moe_num_count = Counter(moe_num)
    globalbs_num_count = Counter(globalbs_num)
    max_moe_num = max(moe_num_count, key=moe_num_count.get)
    max_globalbs_num = max(globalbs_num_count, key=globalbs_num_count.get)
    diff_moe = [idx for idx, val in enumerate(globalbs_num) if val != max_globalbs_num]
    diff_globalbs = [idx for idx, val in enumerate(moe_num) if val != max_moe_num]
    if diff_moe != []:
        logging.error(
            "1.1 %s有如下下标的核的moe专家数与其他核不相等%s,共%d个核\n",
            diff_moe,
            len(diff_moe),
        )
    if diff_globalbs != []:
        logging.error(
            "1.1 %s有如下下标的核的globalbs与其他核不相等%s,共%d个核\n",
            diff_globalbs,
            len(diff_globalbs),
        )
    return moe_num, max_moe_num, globalbs_num, max_globalbs_num


# 获取每个核的rankid,epworldsize,将出现最多次数的epworldsize,rankid作为该卡的epworldsize,rankid
def get_rankid_ep(arr_func: np.ndarray, card_num_func: int, d_c: str):
    per_core = 128
    rankid_num = []
    ep_num = []
    for i in range(card_num_func):
        rankid_num.append(arr_func[2 + (i * per_core)])
        ep_num.append(arr_func[4 + (i * per_core)])
    logging.info("1.1 %s各核rankid:%s\n", d_c, rankid_num)
    logging.info("1.1 %s各核epworldsize:%s\n", d_c, ep_num)
    rankid_num_count = Counter(rankid_num)
    ep_num_count = Counter(ep_num)
    max_rankid_num = max(rankid_num_count, key=rankid_num_count.get)
    max_ep_num = max(ep_num_count, key=ep_num_count.get)
    diff_ep = [idx for idx, val in enumerate(ep_num) if val != max_ep_num]
    diff_rankid = [idx for idx, val in enumerate(rankid_num) if val != max_rankid_num]
    if diff_rankid != []:
        logging.error(
            "1.1 %s有如下下标的核的rankid与其他核不相等%s,共%d个核\n",
            diff_rankid,
            len(diff_rankid),
        )
    if diff_ep != []:
        logging.error(
            "1.1 %s有如下下标的核的epworldsize与其他核不相等%s,共%d个核\n",
            diff_ep,
            len(diff_ep),
        )
    return rankid_num, max_rankid_num, ep_num, max_ep_num


# 获取每个核的hcclcontext中rankid,epworldsize,将出现最多次数的rankid,epworldsize作为该卡的epworldsize,rankid
def get_hccl_rankid_ep(arr_func: np.ndarray, card_num_func: int, d_c: str):
    per_core = 128
    hccl_rankid_num = []
    hccl_ep_num = []
    for i in range(card_num_func):
        hccl_rankid_num.append(arr_func[8 + (i * per_core)])
        hccl_ep_num.append(arr_func[9 + (i * per_core)])
    logging.info("1.1 %s各核建立hccl通信链路时的输入rankid:%s\n", d_c, hccl_rankid_num)
    logging.info("1.1 %s各核建立hccl通信链路时的输入epworldsize:%s\n", d_c, hccl_ep_num)
    hccl_rankid_num_count = Counter(hccl_rankid_num)
    hccl_ep_num_count = Counter(hccl_ep_num)
    hccl_max_rankid_num = max(hccl_rankid_num_count, key=hccl_rankid_num_count.get)
    hccl_max_ep_num = max(hccl_ep_num_count, key=hccl_ep_num_count.get)
    diff_hccl_ep = [
        idx for idx, val in enumerate(hccl_ep_num) if val != hccl_max_ep_num
    ]
    diff_hccl_rankid = [
        idx for idx, val in enumerate(hccl_rankid_num) if val != hccl_max_rankid_num
    ]
    if diff_hccl_rankid != []:
        logging.error(
            "1.1 %s有如下下标的核的建立hccl通信链路时的输入rankid与其他核不相等%s,共%d个核\n",
            diff_hccl_rankid,
            len(diff_hccl_rankid),
        )
    if diff_hccl_ep != []:
        logging.error(
            "1.1 %s有如下下标的核的建立hccl通信链路时的输入epworldsize与其他核不相等%s,共%d个核\n",
            diff_hccl_ep,
            len(diff_hccl_ep),
        )
    return hccl_rankid_num, hccl_max_rankid_num, hccl_ep_num, hccl_max_ep_num


# 获取每个核的执行位置,并以第一个核的0/1标识位作为该卡的0/1标识区,并记录没等到状态的核位置
def get_status_info(arr_func: np.ndarray, card_num_func: int):
    per_core = 128
    list_status = []
    list_0_1_list = []
    unwait_index = []  # 存放没等到状态的核的下标
    for i in range(card_num_func):
        list_status.append(arr_func[1 + (i * per_core)])
        list_0_1_list.append(arr_func[0 + (i * per_core)])
    list_0_1_count = Counter(list_0_1_list)
    max_0_1 = max(list_0_1_count, key=list_0_1_count.get)
    diff_0_1 = [idx for idx, val in enumerate(list_0_1_list) if val != max_0_1]
    unwait_index = [index for index, value in enumerate(list_status) if value == 1]
    return list_status, max_0_1, unwait_index, diff_0_1, list_0_1_list


# 判断是否有核0/1标识位与其他核不相等
def get_diff_0_1(
    arr_func: np.ndarray, diff_0_1: list, card_num_func: int, d_c: str, core_0_1
) -> dict:
    diff_error_dict = {}
    if diff_0_1 != []:
        if d_c == "dispatch":
            logging.error(
                "3.1 dispatch有如下下标的核的状态位与其他核不相等%s,共%d个核",
                diff_0_1,
                len(diff_0_1),
            )
        else:
            logging.error(
                "3.1 combine有如下下标的核的状态位与其他核不相等%s,共%d个核",
                diff_0_1,
                len(diff_0_1),
            )
        for x in diff_0_1:
            diff_error_dict[f"d{card_num_func}_{x}core_{str}"] = (
                f"{x} core 0/1标识位:{arr_func[x]} 与其他核:{core_0_1}不相等"
            )
    return diff_error_dict


# 判断是否为共享专家卡,来分析该卡dis的总状态位数量
def get_dis_status_num(
    dis_epworldsize_func: int,
    dis_moe_num_func: int,
    all_card_num_func: int,
    card_num_func: int,
    share_expert_card_count_func: int,
):
    status_num = 0
    if (card_num_func + 1) <= share_expert_card_count_func:
        logging.info("3.1 该卡为共享专家卡")
        status_num = dis_epworldsize_func  # 共享专家卡:总状态位数量 = ep_worldsize
    else:
        logging.info("3.1 该卡不为共享专家卡")
        # 非共享专家卡:总状态位数量 = ep_worldsize * 单卡moe专家数
        status_num = int(
            (dis_moe_num_func / (all_card_num_func - share_expert_card_count_func))
            * dis_epworldsize_func
        )
    logging.info("3.1 dispatch总状态位数量:%d", status_num)

    return status_num


# 根据共享专家数量,分析该卡com的总状态位数量
def get_com_status_num(
    bs_func: int, k_func: int, share_expert_num_func: int, len_status: int
):
    status_num = 0
    status_num = bs_func * (
        k_func + share_expert_num_func
    )  # 总状态位数量=BS*(K+共享专家数)
    logging.info("3.1 combine总状态位数量:%d", status_num)
    if len_status < (status_num * 8):
        logging.error(
            "3.1 计算得出的总状态位*8(%d) 大于combine状态区数据(int32)shape(%d), 请检查BS,K是否输入正确",
            (status_num * 8),
            len_status,
        )
    return status_num


# dis状态位分析，通过传入的未等到状态的核下标，在对应的0/1状态区查找具体哪个状态没有等到
def dis_status_analysis(
    parms: WinData, dis_core_num_func: int, dis_unwait_index_func: list
):
    int32_status_data = parms.win_data_list_01
    dis_status_list_func = parms.win_data_list_02
    dis_status_num_func = parms.win_data_01
    dis_0_1_func = parms.win_data_02
    card_num_func = parms.card_num_class
    # 状态位数量平均分配给每个核，若不能整除，余数平均分配给前per_core_status_num_any个核
    per_core_status_num = int(dis_status_num_func // dis_core_num_func)
    per_core_status_num_any = int(dis_status_num_func % dis_core_num_func)
    dis_status_core = []
    dis_status_error_dict = {}
    if per_core_status_num_any != 0:
        dis_status_core += [per_core_status_num + 1] * per_core_status_num_any
        dis_status_core += [per_core_status_num] * (
            dis_core_num_func - per_core_status_num_any
        )
    else:
        dis_status_core += [per_core_status_num] * dis_core_num_func
    logging.info("3.2 dispatch 中各核分配到状态位数量%s", dis_status_core)
    if dis_unwait_index_func == []:
        return dis_status_error_dict
    logging.error(
        "3.2 dispatch有如下下标的核未等到状态%s,(共%d个核)",
        dis_unwait_index_func,
        len(dis_unwait_index_func),
    )
    # 在未等到状态的核的对应0/1状态区查找具体哪个状态没有等到
    for i in dis_unwait_index_func:
        if sum(dis_status_core[: i + 1]) > dis_status_num_func:
            break
        if dis_status_list_func[i] != 1:
            continue
        for core_num_func in range(dis_status_core[i]):
            index_status = (
                sum(dis_status_core[: i + 1]) - dis_status_core[i] + core_num_func
            ) * 8
            if index_status >= len(int32_status_data):
                break
            if int32_status_data[index_status] == 0:
                dis_status_error_dict[
                    f"d{card_num_func}_第{i}个核_第{core_num_func}状态位_dispatch{dis_0_1_func}"
                ] = "状态位未等到"
    return dis_status_error_dict


# com状态位分析，通过传入的未等到状态的核下标，在对应的0/1状态区查找具体哪个状态没有等到
def com_status_analysis(
    parms: WinData,
    com_core_num_func: int,
    share_expert_num_func: int,
    com_unwait_index_func: list,
):
    int32_status_data = parms.win_data_list_01
    com_status_list_func = parms.win_data_list_02
    com_status_num_func = parms.win_data_01
    com_0_1_func = parms.win_data_02
    bs_func = parms.bs_class
    k_func = parms.k_class
    card_num_func = parms.card_num_class
    # 状态位数量按BS平均分配给每个核，若不能整除，余数的BS平均分配给前per_core_status_num_any个核
    per_bs_status = k_func + share_expert_num_func  # 单个BS的状态位数量
    per_core_status_num = int(bs_func // com_core_num_func)
    per_core_status_num_any = int(bs_func % com_core_num_func)
    com_status_core = []
    com_statu_error_dict = {}
    if per_core_status_num_any != 0:
        com_status_core += [
            (per_core_status_num + 1) * per_bs_status
        ] * per_core_status_num_any
        com_status_core += [per_core_status_num * per_bs_status] * (
            com_core_num_func - per_core_status_num_any
        )
    else:
        com_status_core += [
            per_core_status_num * per_core_status_num
        ] * com_core_num_func
    logging.info("3.2 combine 中各核分配到状态位数量%s", com_status_core)
    if com_unwait_index_func == []:
        return com_statu_error_dict
    logging.error(
        "3.2 combine有如下下标的核未等到状态%s,(共%d个核)",
        com_unwait_index_func,
        len(com_unwait_index_func),
    )
    # 在未等到状态的核的对应0/1状态区查找具体哪个状态没有等到
    for i in com_unwait_index_func:
        if sum(com_status_core[: i + 1]) > com_status_num_func:
            break
        if com_status_list_func[i] != 1:
            continue
        for core_num_func in range(com_status_core[i]):
            index_status = (
                sum(com_status_core[: i + 1]) - com_status_core[i] + core_num_func
            ) * 8
            if index_status >= len(int32_status_data):
                break
            if int32_status_data[index_status] == 0:
                com_statu_error_dict[
                    f"d{card_num_func}_第{i}个核_第{core_num_func}个状态位_combine{com_0_1_func}区"
                ] = "状态位未等到"
    return com_statu_error_dict


def convert_triple_structure(data):
    nums = []
    for item in data:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                nums.append(list(sub_item))
    return nums


# main
sp_moe_num = int(sys.argv[1])
tp_worldsize = int(sys.argv[2])
share_expert_card_count = int(sys.argv[3])
share_expert_num = int(sys.argv[4])
all_card_num = int(sys.argv[5])
card_num = int(sys.argv[6])
floder_path = sys.argv[7]
soc_version = sys.argv[8]
error_dict = {}
max_dis_run_num = 0
max_com_run_num = 0
dis_moe_num = 0
com_moe_num = 0
dis_globalbs = 0
com_globalbs = 0
dis_bs = 0
com_bs = 0
bs = 0
k = 0
expertids = np.array([])
dump_expandidx = []
dump_epsendcnt = []
start_idx = 0
dis_com_version = "None"
dis_com = "None"
dis_epworldsize = 0
com_epworldsize = 0

if (soc_version == SOC_VERSION_950) or (soc_version == SOC_VERSION_910B):
    perfix = "mc2_"
    endfix = ""
elif soc_version == SOC_VERSION_910_93:
    perfix = "exception_info"
    endfix = ".workspace.1.bin.npy"
else:
    logging.error(
        "soc_version:%d 非法输入, soc_version should be: %s or %s",
        soc_version,
        SOC_VERSION_950,
        SOC_VERSION_910_93,
    )

for filename in os.listdir(os.path.join(floder_path)):
    if filename.startswith(perfix) and filename.endswith(endfix):
        logging.info("开始分析卡%d数据", card_num)
        file_path = os.path.join(floder_path, filename)
        logging.info("解析文件:%s\n", file_path)
        with open(file_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.int8)
        last_1M = arr[-1024 * 1024 :]
        dis_0_status = last_1M[0 : 64 * 1024]
        com_0_status = last_1M[64 * 1024 : 384 * 1024]
        dis_1_status = last_1M[384 * 1024 : 448 * 1024]
        com_1_status = last_1M[448 * 1024 : 768 * 1024]
        dis_win_data = last_1M[768 * 1024 : 818 * 1024]
        com_win_data = last_1M[818 * 1024 : 868 * 1024]

        int32_arr = last_1M.view(np.int32)
        int32_dis_0_status = dis_0_status.view(np.int32)
        int32_com_0_status = com_0_status.view(np.int32)
        int32_dis_1_status = dis_1_status.view(np.int32)
        int32_com_1_status = com_1_status.view(np.int32)
        int32_dis_win_data = dis_win_data.view(np.int32)
        int32_com_win_data = com_win_data.view(np.int32)

        # 判断卡上 dis&com使用核数是否一致
        logging.info("1. 开始进行输入异常分析\n")
        dis_core_num = analysis_core_num(int32_dis_win_data)
        com_core_num = analysis_core_num(int32_com_win_data)
        error_dict.update(compare_core_num(dis_core_num, com_core_num, card_num))
        dis_com, dis_com_version = check_dis_com_version(floder_path)
        # 输入错误分析
        if dis_core_num != 0:
            dis_rankid_list, dis_rankid, dis_epworldsize_list, dis_epworldsize = (
                get_rankid_ep(int32_dis_win_data, dis_core_num, "dispatch")
            )
            (
                dis_hccl_rankid_list,
                dis_hccl_rankid,
                dis_hccl_epworldsize_list,
                dis_hccl_epworldsize,
            ) = get_hccl_rankid_ep(int32_dis_win_data, dis_core_num, "dispatch")
            dis_moe_list, dis_moe_num, dis_globalbs_list, dis_globalbs = (
                get_moe_globalbs(int32_dis_win_data, dis_core_num, "dispatch")
            )
            if dis_epworldsize != 0:
                dis_bs = int(dis_globalbs / dis_epworldsize)
            logging.info(
                "1.1 dispatch_rankid:%d, dispatch_hccl_rankid:%d, dispatch epworldsize:%d,"
                "dispatch_hccl epworldsize:%d, dispatch moe专家数:%d, dispatch globalbs:%d, 根据dispatch输入计算的bs:%d",
                dis_rankid,
                dis_hccl_rankid,
                dis_epworldsize,
                dis_hccl_epworldsize,
                dis_moe_num,
                dis_globalbs,
                dis_bs,
            )
            if dis_rankid != dis_hccl_rankid:
                logging.error(
                    "1.1 dispatch win区数据中的rankid:%d 与建立hccl通信链路时的rankid输入:%s 不同",
                    dis_rankid,
                    dis_hccl_rankid,
                )
            if dis_epworldsize != dis_hccl_epworldsize:
                logging.error(
                    "1.1 dispatch win区数据中的epworldsize:%d 与建立hccl通信链路时的epworldsize输入:%s 不同",
                    dis_epworldsize,
                    dis_hccl_epworldsize,
                )
        else:
            logging.info(
                "1. 未调用到dispatch算子不进行dispatch的rankid,moe专家输入分析"
            )
        if com_core_num != 0:
            com_rankid_list, com_rankid, com_epworldsize_list, com_epworldsize = (
                get_rankid_ep(int32_com_win_data, com_core_num, "combine")
            )
            (
                com_hccl_rankid_list,
                com_hccl_rankid,
                com_hccl_epworldsize_list,
                com_hccl_epworldsize,
            ) = get_hccl_rankid_ep(int32_com_win_data, com_core_num, "combine")
            com_moe_list, com_moe_num, com_globalbs_list, com_globalbs = (
                get_moe_globalbs(int32_com_win_data, com_core_num, "combine")
            )
            if com_epworldsize != 0:
                com_bs = int(com_globalbs / com_epworldsize)
            logging.info(
                "1.1 combine_rankid:%d, combine_hccl_rankid:%d, combine epworldsize:%d,"
                "combine_hccl epworldsize:%d, combine moe专家数:%d, combine globalbs:%d, 根据combine输入计算的bs:%d",
                com_rankid,
                com_hccl_rankid,
                com_epworldsize,
                com_hccl_epworldsize,
                com_moe_num,
                com_globalbs,
                com_bs,
            )
            if com_rankid != com_hccl_rankid:
                logging.error(
                    "1.1 combine win区数据中的rankid:%d 与建立hccl通信链路时的rankid输入:%s 不同",
                    com_rankid,
                    com_hccl_rankid,
                )
            if com_epworldsize != com_hccl_epworldsize:
                logging.error(
                    "1.1 combine win区数据中的epworldsize:%d 与建立hccl通信链路时的epworldsize输入:%s 不同",
                    com_epworldsize,
                    com_hccl_epworldsize,
                )
        else:
            logging.info("1.1 未调用到combine算子不进行combine的rankid,moe专家输入分析")

        if soc_version == SOC_VERSION_910B:
            if dis_core_num != 0:
                all_card_num = dis_epworldsize
            elif com_core_num != 0:
                all_card_num = com_epworldsize

        # 总卡数以win区获取的ep_worldsize为准，如没有则以文件数判断
        if dis_epworldsize != 0 and com_epworldsize != 0:
            if dis_epworldsize == com_epworldsize:
                all_card_num = dis_epworldsize

        if dis_moe_num != 0 and com_moe_num != 0:
            if dis_moe_num != com_moe_num:
                logging.error(
                    "1.2 dispatch得出的moe专家数:%d与combine得出的moe专家数:%d不同",
                    dis_moe_num,
                    com_moe_num,
                )
            else:
                local_expert_num = get_local_expert_num(
                    com_moe_num, share_expert_card_count, card_num, all_card_num
                )
        elif dis_moe_num != 0 and com_moe_num == 0:
            local_expert_num = get_local_expert_num(
                dis_moe_num, share_expert_card_count, card_num, all_card_num
            )
        else:
            local_expert_num = get_local_expert_num(
                com_moe_num, share_expert_card_count, card_num, all_card_num
            )

        if dis_bs == 0:
            bs = com_bs
            logging.info("1.2 根据计算得出该卡的bs=%d", bs)
        elif com_bs == 0:
            bs = dis_bs
            logging.info("1.2 根据计算得出该卡的bs=%d", bs)
        else:
            if com_bs == dis_bs:
                bs = dis_bs
                logging.info("1.2 根据计算得出该卡的bs=%d", bs)
            else:
                logging.error(
                    "1.2 根据dispatch输入计算得出的bs:%d与根据combine输入计算得出的bs:%d不同",
                    dis_bs,
                    com_bs,
                )

        if dis_com == "dispatch" and dis_core_num != 0:
            k, expertids = check_topk(
                floder_path, dis_moe_num, dis_bs, sp_moe_num, dis_com_version
            )
        if dis_com == "combine" and com_core_num != 0:
            k, expertids = check_topk(
                floder_path, com_moe_num, com_bs, sp_moe_num, dis_com_version
            )
        logging.info("1.3 根据计算得出该卡的k为:%d", k)

        if dis_com == "combine" and com_core_num != 0:
            a = get_a(floder_path, dis_com_version)
            h = get_h(floder_path, a, tp_worldsize, dis_com_version)
            logging.info(
                "1.4 根据combine输入计算得出的A:%d, 根据combine输入计算得出的H:%d", a, h
            )
            dump_epsendcnt = get_dump_epsendcnt(floder_path, dis_com_version)
            dump_expandidx = get_dump_expandidx(floder_path, dis_com_version)
        logging.info("1.5 输入异常分析完成\n")

        # 执行序分析
        logging.info("2. 开始执行序分析")
        if dis_core_num != 0:
            dis_run_num_list, dis_diff_indices, max_dis_run_num = analysis_run_num(
                int32_dis_win_data, dis_core_num
            )
            logging.info("2. dispatch各核执行次数:%s", dis_run_num_list)
        if com_core_num != 0:
            com_run_num_list, com_diff_indices, max_com_run_num = analysis_run_num(
                int32_com_win_data, com_core_num
            )
            logging.info("2. combine各核执行次数:%s", com_run_num_list)
        if dis_core_num != 0 and com_core_num != 0:
            run_num_class = WinData(
                win_data_list_01=dis_run_num_list,
                win_data_list_02=com_run_num_list,
                win_data_01=max_dis_run_num,
                win_data_02=max_com_run_num,
                card_num_class=card_num,
                bs_class=bs,
                k_class=k,
            )
            error_dict.update(
                compare_run_num(
                    run_num_class, dis_diff_indices, com_diff_indices, dis_com
                )
            )
            logging.info("2. 执行序分析完成\n")
        else:
            logging.info("2. 单个算子的调用场景不进行执行序分析\n")
        # 状态位分析
        if dis_core_num != 0:
            logging.info("3. 开始dispatch状态位分析")
            (
                dis_status_list,
                dis_0_1,
                dis_unwait_index,
                dis_diff_0_1_list,
                dis_0_1_list,
            ) = get_status_info(int32_dis_win_data, dis_core_num)
            error_dict.update(
                get_diff_0_1(
                    dis_0_1_list, dis_diff_0_1_list, card_num, "dispatch", dis_0_1
                )
            )
            logging.info("3.1 dispatch各核执行位置情况:%s", dis_status_list)
            dis_status_num = get_dis_status_num(
                dis_epworldsize,
                dis_moe_num,
                all_card_num,
                card_num,
                share_expert_card_count,
            )
            if len(int32_dis_0_status) < (dis_status_num * 8):
                logging.error(
                    "3.1 计算得出的总状态位*8(%d) 大于dispatch状态区数据(int32)shape(%d)",
                    (dis_status_num * 8),
                    len(int32_dis_0_status),
                )
            logging.info("3.2 状态区实际挂的区域与win区获取的区域相反")
            if dis_0_1 == 0:
                logging.info("3.2 dispatch 1区状态区数据:%s", int32_dis_1_status.dtype)
                logging.info(
                    "3.2 dispatch 1区状态区数据 shape:%d", len(int32_dis_1_status)
                )
                logging.info("3.2 dispatch 1区状态区数据:%s", int32_dis_1_status)
                dis_status_class = WinData(
                    win_data_list_01=int32_dis_1_status,
                    win_data_list_02=dis_status_list,
                    win_data_01=dis_status_num,
                    win_data_02=dis_0_1,
                    card_num_class=card_num,
                    bs_class=bs,
                    k_class=k,
                )
                error_dict.update(
                    dis_status_analysis(
                        dis_status_class, dis_core_num, dis_unwait_index
                    )
                )
                logging.info("3.3 dispatch状态区分析完成\n")
            elif dis_0_1 == 1:
                logging.info("3.2 dispatch 0区状态区数据:%s", int32_dis_0_status.dtype)
                logging.info(
                    "3.2 dispatch 0区状态区数据 shape:%d", len(int32_dis_0_status)
                )
                logging.info("3.2 dispatch 0区状态区数据:%s", int32_dis_0_status)
                dis_status_class = WinData(
                    win_data_list_01=int32_dis_0_status,
                    win_data_list_02=dis_status_list,
                    win_data_01=dis_status_num,
                    win_data_02=dis_0_1,
                    card_num_class=card_num,
                    bs_class=bs,
                    k_class=k,
                )
                error_dict.update(
                    dis_status_analysis(
                        dis_status_class, dis_core_num, dis_unwait_index
                    )
                )
                logging.info("3.3 dispatch状态区分析完成\n")
            else:
                logging.error(
                    "3. dispatch 0/1标识位 should be 0/1 but got %d\n", dis_0_1
                )
        else:
            logging.info("3. 未调用dispatch,不进行dispatch状态区分析\n")

        if com_core_num != 0:
            logging.info("4. 开始combine状态位分析")
            (
                com_status_list,
                com_0_1,
                com_unwait_index,
                com_diff_0_1_list,
                com_0_1_list,
            ) = get_status_info(int32_com_win_data, com_core_num)
            error_dict.update(
                get_diff_0_1(
                    com_0_1_list, com_diff_0_1_list, card_num, "combine", com_0_1
                )
            )
            logging.info("4.1 combine各核执行位置情况:%s", com_status_list)
            com_status_num = get_com_status_num(
                bs, k, share_expert_num, len(int32_dis_0_status)
            )
            logging.info("4.2 状态区实际挂的区域与win区获取的区域相反")
            if com_0_1 == 0:
                logging.info("4.2 combine 1区状态区数据:%s", int32_com_1_status.dtype)
                logging.info(
                    "4.2 combine 1区状态区数据 shape:%d", len(int32_com_1_status)
                )
                logging.info("4.2 combine 1区状态区数据:%s", int32_com_1_status)
                com_status_class = WinData(
                    win_data_list_01=int32_com_1_status,
                    win_data_list_02=com_status_list,
                    win_data_01=com_status_num,
                    win_data_02=com_0_1,
                    bs_class=bs,
                    k_class=k,
                    card_num_class=card_num,
                )
                error_dict.update(
                    com_status_analysis(
                        com_status_class,
                        com_core_num,
                        share_expert_num,
                        com_unwait_index,
                    )
                )
                logging.info("4.3 combine状态区分析完成\n")
            elif com_0_1 == 1:
                logging.info("4.2 combine 0区状态区数据:%s", int32_com_0_status.dtype)
                logging.info(
                    "4.2 combine 0区状态区数据 shape:%d", len(int32_com_0_status)
                )
                logging.info("4.2 combine 0区状态区数据:%s", int32_com_0_status)
                com_status_class = WinData(
                    win_data_list_01=int32_com_0_status,
                    win_data_list_02=com_status_list,
                    win_data_01=com_status_num,
                    win_data_02=com_0_1,
                    bs_class=bs,
                    k_class=k,
                    card_num_class=card_num,
                )
                error_dict.update(
                    com_status_analysis(
                        com_status_class,
                        com_core_num,
                        share_expert_num,
                        com_unwait_index,
                    )
                )
                logging.info("4.3 combine状态区分析完成\n")
            else:
                logging.error(
                    "4. combine 0/1标识位 should be 0/1 but got %d\n", com_0_1
                )
        else:
            logging.info("4. 未调用com,不进行com状态区分析\n")
        if soc_version == SOC_VERSION_910B:
            if dis_com == "dispatch":
                A2_ep_rankid = dis_rankid
                A2_ep_worldsize = dis_epworldsize
                A2_moe_expert_num = dis_moe_num
                A2_run_num_list = dis_status_list
                buffer_id = dis_0_1
            else:
                A2_ep_rankid = com_rankid
                A2_ep_worldsize = com_epworldsize
                A2_moe_expert_num = com_moe_num
                A2_run_num_list = com_status_list
                buffer_id = com_0_1
            A2.analyze_single_card(
                arr_func_dis=int32_dis_win_data,
                arr_func_com=int32_com_win_data,
                core_num=max(dis_core_num, com_core_num),
                ep_rankid=A2_ep_rankid,
                d_c=dis_com,
                ep_worldsize=A2_ep_worldsize,
                runpos=A2_run_num_list,
                moe_expert_num=A2_moe_expert_num,
                k=k,
                buffer_id=buffer_id,
                target_path=floder_path,
                expertids=expertids,
                dis_status=dis_0_status,
                com_status=com_0_status,
                dis_run_cnts=max_dis_run_num,
                com_run_cnts=max_com_run_num,
            )
            break

        # 输出
        logging.info("5. 数据归档")
        if dis_core_num != 0:
            file_status_list_exists = os.path.exists("win_status_list.csv")
            file_exists = os.path.exists("win_data.csv")
            file_list_exists = os.path.exists("win_data_list.csv")
            dis_core_info = pd.DataFrame(
                [
                    [
                        dis_core_num,
                        dis_moe_num,
                        dis_globalbs,
                        dis_rankid,
                        dis_hccl_rankid,
                        dis_epworldsize,
                        dis_hccl_epworldsize,
                        dis_bs,
                        k,
                        dis_0_1,
                    ]
                ],
                columns=[
                    "使用核数",
                    "moe专家数",
                    "globals",
                    "rankid",
                    "hccl_rankid",
                    "ep_worldsize",
                    "hccl_ep_worldsize",
                    "bs",
                    "k",
                    "0/1标识",
                ],
                index=[f"d{card_num}_dispatch"],
            )

            dis_core_info.to_csv(
                "win_data.csv",
                index=True,
                mode="a",
                header=not file_exists,
                encoding="gbk",
            )
            dis_status_list_info = pd.DataFrame(
                [dis_status_list], index=[f"d{card_num}_dispatch各核执行位置数据"]
            )
            dis_status_list_info.to_csv(
                "win_data_list.csv",
                index=True,
                mode="a",
                header=not file_list_exists,
                encoding="gbk",
            )
            dis_0_1_list_info = pd.DataFrame(
                [dis_0_1_list], index=[f"d{card_num}_dispatch各核0/1标识区数据"]
            )
            dis_0_1_list_info.to_csv(
                "win_data_list.csv",
                index=True,
                mode="a",
                header=not file_list_exists,
                encoding="gbk",
            )

            if dis_0_1 == 0:
                int32_dis_1_status_info = pd.DataFrame(
                    [int32_dis_1_status], index=[f"d{card_num}_dispatch 1区状态区数据"]
                )
                int32_dis_1_status_info.to_csv(
                    "win_status_list.csv",
                    index=True,
                    mode="a",
                    header=not file_status_list_exists,
                    encoding="gbk",
                )
            else:
                int32_dis_0_status_info = pd.DataFrame(
                    [int32_dis_0_status], index=[f"d{card_num}_dispatch 0区状态区数据"]
                )
                int32_dis_0_status_info.to_csv(
                    "win_status_list.csv",
                    index=True,
                    mode="a",
                    header=not file_status_list_exists,
                    encoding="gbk",
                )

        if com_core_num != 0:
            file_status_list_exists = os.path.exists("win_status_list.csv")
            file_exists = os.path.exists("win_data.csv")
            file_list_exists = os.path.exists("win_data_list.csv")
            com_core_info = pd.DataFrame(
                [
                    [
                        com_core_num,
                        com_moe_num,
                        com_globalbs,
                        com_rankid,
                        com_hccl_rankid,
                        com_epworldsize,
                        com_hccl_epworldsize,
                        com_bs,
                        k,
                        com_0_1,
                    ]
                ],
                columns=[
                    "使用核数",
                    "moe专家数",
                    "globals",
                    "rankid",
                    "hccl_rankid",
                    "ep_worldsize",
                    "hccl_ep_worldsize",
                    "bs",
                    "k",
                    "0/1标识",
                ],
                index=[f"d{card_num}_combine"],
            )
            com_core_info.to_csv(
                "win_data.csv",
                index=True,
                mode="a",
                header=not file_exists,
                encoding="gbk",
            )

            com_status_list_info = pd.DataFrame(
                [com_status_list], index=[f"d{card_num}_combine各核执行位置数据"]
            )
            com_status_list_info.to_csv(
                "win_data_list.csv",
                index=True,
                mode="a",
                header=not file_list_exists,
                encoding="gbk",
            )
            com_0_1_list_info = pd.DataFrame(
                [com_0_1_list], index=[f"d{card_num}_combine各核0/1标识区数据"]
            )
            com_0_1_list_info.to_csv(
                "win_data_list.csv",
                index=True,
                mode="a",
                header=not file_list_exists,
                encoding="gbk",
            )

            if com_0_1 == 0:
                int32_com_1_status_info = pd.DataFrame(
                    [int32_com_1_status], index=[f"d{card_num}_combine 1区状态区数据"]
                )
                int32_com_1_status_info.to_csv(
                    "win_status_list.csv",
                    index=True,
                    mode="a",
                    header=not file_status_list_exists,
                    encoding="gbk",
                )
            else:
                int32_com_0_status_info = pd.DataFrame(
                    [int32_com_0_status], index=[f"d{card_num}_combine 0区状态区数据"]
                )
                int32_com_0_status_info.to_csv(
                    "win_status_list.csv",
                    index=True,
                    mode="a",
                    header=not file_status_list_exists,
                    encoding="gbk",
                )

        logging.info(
            "5. 该卡的dispatch&combine的使用核数、moe专家数、globals、rankid、hccl_rankid、ep_worldsize、"
            "hccl_ep_worldsize、bs、k、0/1标识区数据已归档至win_data.csv"
        )
        logging.info(
            "5. 该卡的中各核的dispatch&combine的执行位置、0/1标识区数据已归档至win_data_list.csv"
        )
        logging.info(
            "5. 该卡的中所使用的dispatch&combine状态区数据已归档至win_status_list.csv"
        )
        file_error_exists = os.path.exists("win_analysis_error.csv")
        error_rows = [[key, value] for key, value in error_dict.items()]
        df = pd.DataFrame(error_rows, columns=["card_num", "error"])
        df.to_csv(
            "win_analysis_error.csv",
            index=True,
            mode="a",
            header=not file_error_exists,
            encoding="gbk",
        )
        logging.info("5. 分析出的错误详细信息归档至win_analysis_error.csv\n")
        file_all_card_dat_exists = os.path.exists("win_all_card_data.csv")

        all_card_run_num_info = pd.DataFrame(
            [
                [
                    max_dis_run_num,
                    max_com_run_num,
                    dis_moe_num,
                    com_moe_num,
                    dis_globalbs,
                    com_globalbs,
                ]
            ],
            columns=[
                "dispatch执行次数",
                "combine执行次数",
                "dispatch_moe专家数",
                "combine_moe专家数",
                "dispatch_globalbs",
                "combine_globalbs",
            ],
            index=[f"d{card_num}"],
        )
        all_card_run_num_info.to_csv(
            "win_all_card_data.csv",
            index=True,
            mode="a",
            header=not file_all_card_dat_exists,
            encoding="gbk",
        )

        file_all_card_expandx_exists = os.path.exists("win_all_card_expandidx.csv")
        all_card_expandx_info = pd.DataFrame(
            [
                [
                    expertids.tolist(),
                    dump_expandidx,
                    dump_epsendcnt,
                    local_expert_num,
                    dis_com,
                ]
            ],
            columns=[
                "expertids",
                "dump数据中读取的expandidx",
                "dump数据中读取的epsendcnt",
                "本卡专家数",
                "挂在哪个算子上",
            ],
            index=[card_num],
        )
        all_card_expandx_info.to_csv(
            "win_all_card_expandidx.csv",
            index=True,
            mode="a",
            header=not file_all_card_expandx_exists,
            encoding="gbk",
        )
        # 当分析到最后一张卡时，进行多卡的dispatch、combine数据对比
        if card_num == (all_card_num - 1):
            logging.info("6. 开始多卡的dispatch、combine数据对比")
            all_card_dis_run_num = []
            all_card_com_run_num = []
            all_card_dis_moe_num = []
            all_card_com_moe_num = []
            all_card_dis_globalbs_num = []
            all_card_com_globalbs_num = []
            with open("win_all_card_data.csv", "r", encoding="gbk") as f_run_num:
                reader = csv.DictReader(f_run_num)
                for row in reader:
                    all_card_dis_run_num.append(int(row["dispatch执行次数"]))
                    all_card_com_run_num.append(int(row["combine执行次数"]))
                    all_card_dis_moe_num.append(int(row["dispatch_moe专家数"]))
                    all_card_com_moe_num.append(int(row["combine_moe专家数"]))
                    all_card_dis_globalbs_num.append(int(row["dispatch_globalbs"]))
                    all_card_com_globalbs_num.append(int(row["combine_globalbs"]))
            dis_run_judge = all(
                x == all_card_dis_run_num[0]
                for x in all_card_dis_run_num[:all_card_num]
            )
            com_run_judge = all(
                y == all_card_com_run_num[0]
                for y in all_card_com_run_num[:all_card_num]
            )
            dis_moe_judge = all(
                x == all_card_dis_moe_num[0]
                for x in all_card_dis_moe_num[:all_card_num]
            )
            com_moe_judge = all(
                y == all_card_com_moe_num[0]
                for y in all_card_com_moe_num[:all_card_num]
            )
            dis_bs_judge = all(
                x == all_card_dis_globalbs_num[0]
                for x in all_card_dis_globalbs_num[:all_card_num]
            )
            com_bs_judge = all(
                y == all_card_com_globalbs_num[0]
                for y in all_card_com_globalbs_num[:all_card_num]
            )

            if not dis_run_judge:
                logging.error("6. 多卡对比中,有卡的dispatch的执行次数与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的dispatch的执行次数完全相同")
            if not com_run_judge:
                logging.error("6. 多卡对比中,有卡的combine的执行次数与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的combine的执行次数完全相同")

            if not dis_moe_judge:
                logging.error("6. 多卡对比中,有卡的dispatch的moe专家数与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的dispatch的moe专家数完全相同")
            if not com_moe_judge:
                logging.error("6. 多卡对比中,有卡的combine的moe专家数与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的combine的moe专家数完全相同")

            if not dis_bs_judge:
                logging.error("6. 多卡对比中,有卡的dispatch的globalbs与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的dispatch的globalbs完全相同")
            if not com_bs_judge:
                logging.error("6. 多卡对比中,有卡的combine的globalbs与其他卡不相同")
            else:
                logging.info("6. 多卡对比中,各卡的combine的globalbs完全相同")

            logging.info(
                "6. 各卡中dispatch的执行次数:%s", all_card_dis_run_num[:all_card_num]
            )
            logging.info(
                "6. 各卡中combine的执行次数:%s", all_card_com_run_num[:all_card_num]
            )
            logging.info(
                "6. 各卡中dispatch的moe专家数:%s", all_card_dis_moe_num[:all_card_num]
            )
            logging.info(
                "6. 各卡中combine的moe专家数:%s", all_card_com_moe_num[:all_card_num]
            )
            logging.info(
                "6. 各卡中dispatch的globalbs:%s",
                all_card_dis_globalbs_num[:all_card_num],
            )
            logging.info(
                "6. 各卡中combine的globalbs:%s",
                all_card_com_globalbs_num[:all_card_num],
            )
            logging.info(
                "6. 各卡中dispatch、combine的执行次数、moe专家数、globalbs数据已归档至win_all_card_data.csv"
            )

            logging.info("7. 开始所有卡的expandidx,epsendcnt对比")
            expandidx_count = {}
            all_card_expertids = {}
            all_card_expandidx = {}
            all_card_dis_com = {}
            all_local_expetrnum = {}
            all_card_epsendcnt = {}
            with open(
                "win_all_card_expandidx.csv", "r", encoding="gbk"
            ) as f_expandidx_num:
                reader = csv.DictReader(f_expandidx_num)
                row_index_col = reader.fieldnames[0]
                for row in reader:
                    row_index = row[row_index_col].strip()
                    expertids_data = ast.literal_eval(row["expertids"])
                    expandidx_data = ast.literal_eval(row["dump数据中读取的expandidx"])
                    raw_data = (
                        row["dump数据中读取的epsendcnt"]
                        .replace("[ ", "[")
                        .replace("] ", "]")
                    )
                    epsendcnt_data = ast.literal_eval(
                        raw_data.replace(" ", ",").strip(",").replace(",,", ",")
                    )
                    dis_com = row["挂在哪个算子上"]
                    expetrnum = int(row["本卡专家数"])
                    all_card_expertids[row_index] = expertids_data
                    all_card_expandidx[row_index] = expandidx_data
                    all_card_epsendcnt[row_index] = epsendcnt_data
                    all_card_dis_com[row_index] = dis_com
                    all_local_expetrnum[row_index] = expetrnum
            sorted_card_ids = sorted(all_card_expertids.keys())
            for card_id in sorted_card_ids:
                expertids_one_card = all_card_expertids[card_id]
                for bs_idx, row in enumerate(expertids_one_card):
                    for topkid, expertid in enumerate(row):
                        if expertid not in expandidx_count:
                            expandidx_count[expertid] = []
                        triple = (int(card_id), bs_idx, topkid)
                        expandidx_count[expertid].append(triple)
            expandidx_count = dict(sorted(expandidx_count.items()))
            epsendcnt_count = count_all_card_epsendcnt(
                all_card_expertids, all_local_expetrnum
            )
            logging.info("7. 所有卡的计算的epsendcnt如下:%s", epsendcnt_count)
            check_epsendcnt(epsendcnt_count, all_card_epsendcnt)
            logging.info("7. 所有专家计算得到的的expandidx如下:%s", expandidx_count)
            for card_id in sorted_card_ids:
                compare_expandidx = []
                local_expert_num_one_card = all_local_expetrnum[card_id]
                if all_card_dis_com[card_id] == "combine":
                    dump_expandidx_one_card = all_card_expandidx[card_id]
                    for expert_id in range(
                        start_idx, start_idx + local_expert_num_one_card
                    ):
                        triples = expandidx_count.get(expert_id, [])
                        if triples:
                            compare_expandidx.append(triples)
                    dump_expandidx = dump_expandidx_one_card[
                        0 : len(compare_expandidx) + 1
                    ]
                    compare_expandidx_final = convert_triple_structure(
                        compare_expandidx
                    )
                    judge_expandidx = True
                    for expand_idx, (triple1, triple2) in enumerate(
                        zip(compare_expandidx_final, dump_expandidx)
                    ):
                        if triple1 != triple2:
                            judge_expandidx = False
                            logging.error(
                                "7. 第%d个三元组不一致:计算的expandidx:%s,dump数据取出的expandidx%s",
                                expand_idx,
                                triple1,
                                triple2,
                            )
                    if judge_expandidx == True:
                        logging.info("7. 卡%s的expandidx没有异常", card_id)
                else:
                    logging.info(
                        "7. 卡%s的算子执行流程并未卡死combine算子上,不进行分析", card_id
                    )
                    continue
                start_idx += local_expert_num_one_card
            logging.info(
                "7. 各卡的expertids,dump数据中读取的expandidx,epsendcnt,本卡专家数,该卡挂在哪个算子上已归档至"
                "win_all_card_expandidx.csv"
            )
            logging.info("7. 所有卡的expandidx,epsendcnt对比结束")

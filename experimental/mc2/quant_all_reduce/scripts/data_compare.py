# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import csv
import logging
import argparse
from typing import Tuple, Dict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import numpy as np
from ml_dtypes import float8_e5m2, float8_e8m0fnu, bfloat16

# 3. 日志配置（全局初始化）
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ========== 封装精度指标为数据类 ==========
@dataclass
class AccuracyMetrics:
    """强相关的精度参数"""

    mae: float  # 平均绝对误差
    max_ae: float  # 最大绝对误差
    mse: float  # 均方误差
    rmse: float  # 均方根误差
    cos_sim: float  # 余弦相似度

    def round_all(self, decimals: int = 6) -> None:
        """将所有指标四舍五入到指定小数位"""
        self.mae = round(self.mae, decimals)
        self.max_ae = round(self.max_ae, decimals)
        self.mse = round(self.mse, decimals)
        self.rmse = round(self.rmse, decimals)
        self.cos_sim = round(self.cos_sim, decimals)


class QuantAllReduceAccuracyChecker:
    """
    quant_all_reduce accuracy checker
    对比CPU Golden值与NPU计算值的精度指标, 输出结果并写入CSV
    """

    # 类常量：数据类型精度配置（所有实例共享）
    TYPE_CONFIG: Dict[str, Dict] = {
        "int32_t": {"dtype": np.int32, "mae": 1e-3, "cos": 0.9999},
        "float16_t": {"dtype": np.float16, "mae": 1e-3, "cos": 0.9999},
        "float32_t": {"dtype": np.float32, "mae": 1e-3, "cos": 0.9999},
        "int8_t": {"dtype": np.int8, "mae": 1e-3, "cos": 0.9999},
        "bfloat16_t": {"dtype": bfloat16, "mae": 5e-3, "cos": 0.999},
        "fp8_e5m2_t": {"dtype": float8_e5m2, "mae": 1e-2, "cos": 0.99},
        "fp8_e8m0_t": {"dtype": float8_e8m0fnu, "mae": 8e-3, "cos": 0.995},
    }

    def __init__(self, args):
        """
        初始化精度校验器
        :param args: 命令行参数Namespace对象
        """
        # 保存命令行参数为实例属性
        self.case_name = args.case_name
        self.bs = args.bs
        self.hidden_size = args.hidden_size
        self.output_type = args.output_type
        self.ranksize = args.ranksize
        # 初始化Golden目录
        self.golden_dir = (
            f"./golden/quantallreduce_{self.case_name}_{self.bs}_{self.hidden_size}"
        )
        # 校验输出类型是否支持
        if self.output_type not in self.TYPE_CONFIG:
            raise ValueError(
                f"不支持的输出类型：{self.output_type}，支持类型：{list(self.TYPE_CONFIG.keys())}"
            )
        # 获取当前类型的精度配置
        self.type_info = self.TYPE_CONFIG[self.output_type]
        self.target_dtype = self.type_info["dtype"]

    def write_to_result_csv(
        self, metrics: AccuracyMetrics, is_pass: bool, csv_path: str = "./result.csv"
    ) -> bool:
        """
        将精度对比结果写入CSV文件
        :param metrics: 精度指标数据类实例
        :param is_pass: 是否通过校验
        :param csv_path: CSV文件路径
        :return: 是否写入成功
        """
        # 对指标四舍五入（避免小数位过长）
        metrics.round_all(decimals=6)
        # 构造数据行
        header = [
            "case_name",
            "mae",
            "max_ae",
            "mse",
            "rmse",
            "cos_sim",
            "is_pass",
            "timestamp",
        ]
        data_row = [
            self.case_name,
            metrics.mae,
            metrics.max_ae,
            metrics.mse,
            metrics.rmse,
            metrics.cos_sim,
            is_pass,
            self._get_utc8_timestamp(),
        ]
        # 写入CSV文件
        try:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(data_row)
            logging.info(f"results written to CSV file: {csv_path}")
            return True
        except Exception as e:
            logging.error(f"failed to write CSV file: {str(e)}")
            return False

    def calculate_metrics(
        self, cpu_data: np.ndarray, npu_data: np.ndarray
    ) -> AccuracyMetrics:
        """
        计算CPU与NPU数据的精度指标
        :param cpu_data: CPU Golden数据
        :param npu_data: NPU计算数据
        :return: 精度指标数据类实例
        """
        # 提升精度到float32计算指标（避免低精度误差）
        cpu_data_32 = cpu_data.astype(np.float32)
        npu_data_32 = npu_data.astype(np.float32)
        # 计算绝对误差
        abs_error = np.abs(cpu_data_32 - npu_data_32)
        mae = float(np.mean(abs_error))
        max_ae = float(np.max(abs_error))
        # 计算均方误差
        squared_error = np.square(cpu_data_32 - npu_data_32)
        mse = float(np.mean(squared_error))
        rmse = float(np.sqrt(mse)) if mse > 0 else 0.0
        # 计算余弦相似度（处理零范数边界情况）
        cpu_norm = float(np.linalg.norm(cpu_data_32))
        npu_norm = float(np.linalg.norm(npu_data_32))
        if cpu_norm < 1e-10 or npu_norm < 1e-10:
            cos_sim = 1.0 if np.allclose(cpu_data_32, npu_data_32, atol=1e-3) else 0.0
        else:
            dot_product = float(np.dot(cpu_data_32.flatten(), npu_data_32.flatten()))
            cos_sim = dot_product / (cpu_norm * npu_norm)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 防止数值溢出
        # 返回封装后的精度指标
        return AccuracyMetrics(
            mae=mae, max_ae=max_ae, mse=mse, rmse=rmse, cos_sim=cos_sim
        )

    def compare(self) -> bool:
        """
        执行所有Rank的精度对比
        :return: 是否所有Rank都通过校验
        """
        logging.info("=" * 50)
        logging.info("start accuracy comparison")
        logging.info(
            f"case: {self.case_name} | output type: {self.output_type} | rank count: {self.ranksize}"
        )
        logging.info("=" * 50)
        # 逐Rank对比NPU数据
        all_pass = True
        final_metrics = AccuracyMetrics(
            mae=0.0, max_ae=0.0, mse=0.0, rmse=0.0, cos_sim=0.0
        )
        for rank in range(self.ranksize):
            logging.info(f"\n---------- compare Rank-{rank} NPU data ----------")
            # 加载数据
            npu_data, load_success = self._load_data(rank, "npu")
            if not load_success:
                all_pass = False
                continue
            cpu_data, load_success = self._load_data(rank, "cpu")
            if not load_success:
                all_pass = False
                continue
            # 校验数据长度
            if len(cpu_data) != len(npu_data):
                logging.error(
                    f"Rank-{rank} data length mismatch | CPU: {len(cpu_data)} | NPU: {len(npu_data)}"
                )
                all_pass = False
                continue
            # 计算精度指标（返回封装后的对象）
            metrics = self.calculate_metrics(cpu_data, npu_data)
            # 记录最后一个有效Rank的指标（用于写入CSV）
            final_metrics = metrics
            # 打印精度指标
            logging.info(f"Rank-{rank} accuracy metrics:")
            logging.info(
                f"  MAE: {metrics.mae:.6f} (threshold: {self.type_info['mae']})"
            )
            logging.info(f"  MaxAE: {metrics.max_ae:.6f}")
            logging.info(f"  MSE: {metrics.mse:.6f}")
            logging.info(f"  RMSE: {metrics.rmse:.6f}")
            logging.info(
                f"  余弦相似度：{metrics.cos_sim:.6f} (阈值：{self.type_info['cos']})"
            )
            # 判断当前Rank是否通过
            rank_pass = (metrics.mae < self.type_info["mae"]) and (
                metrics.cos_sim > self.type_info["cos"]
            )
            if rank_pass:
                logging.info(f"[PASS] Rank-{rank} accuracy check passed")
            else:
                logging.error(f"[FAIL] Rank-{rank} accuracy check failed")
                all_pass = False
        # 输出最终结果
        logging.info("\n" + "=" * 50)
        if all_pass:
            logging.info(
                f"[PASS] all ranks (type: {self.output_type}) accuracy check passed"
            )
        else:
            logging.error(
                f"[FAIL] some ranks (type: {self.output_type}) accuracy check failed"
            )
        logging.info("=" * 50)
        # 写入CSV文件（传递封装后的指标对象）
        self.write_to_result_csv(final_metrics, all_pass)
        return all_pass

    def _load_data(self, rank: int, name: str) -> Tuple[np.ndarray | None, bool]:
        """
        加载指定Rank的数据文件
        :param rank: 要加载的Rank编号
        :return: (加载成功返回NPU数据数组, 失败返回None) , (是否加载成功: True/False)
        """
        file = os.path.join(self.golden_dir, f"output_{name}_{rank}.bin")
        # 1. 检查文件是否存在
        if not os.path.exists(file):
            logging.error(f"Rank-{rank} {name} data file not found: {file}")
            return None, False
        # 2. 加载NPU数据（带异常处理）
        try:
            data = np.fromfile(file, dtype=self.target_dtype)
            logging.info(f"Rank-{rank} {name} data loaded | length: {len(data)}")
            return data, True
        except Exception as e:
            logging.error(f"failed to load Rank-{rank} {name} data: {e}")
            return None, False

    def _get_utc8_timestamp(self) -> datetime:
        """私有方法: 获取UTC+8时间戳"""
        utc_time = datetime.now(tz=timezone.utc)
        return utc_time + timedelta(hours=8)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数（独立函数，便于测试和扩展）
    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="量化AllReduce算子精度对比工具")
    # 必选参数（按顺序）
    parser.add_argument("case_name", type=str, help="测试用例名称")
    parser.add_argument("bs", type=int, help="Batch Size")
    parser.add_argument("hidden_size", type=int, help="Hidden Size")
    parser.add_argument("output_type", type=str, help="输出数据类型")
    parser.add_argument("ranksize", type=int, help="Rank数量")
    return parser.parse_args()


if __name__ == "__main__":
    """主函数：解析参数 → 初始化校验器 → 执行精度对比"""
    try:
        # 解析命令行参数
        args = parse_args()
        # 创建精度校验器实例
        checker = QuantAllReduceAccuracyChecker(args)
        # 执行精度对比
        result = checker.compare()
        # 退出码：0=通过，1=失败
        exit(0 if result else 1)
    except Exception as e:
        logging.error(f"program execution failed: {e}")
        exit(1)

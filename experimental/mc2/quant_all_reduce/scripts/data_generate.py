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
import logging
import argparse
from typing import Union, Dict, Type
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ml_dtypes import float8_e5m2, float8_e8m0fnu, bfloat16

# 3. 日志配置（仅初始化一次，全局生效）
logging.basicConfig(level=logging.INFO, format="%(message)s")

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"


class QuantAllReduceGoldenGenerator:
    """
    quant_all_reduce golden data generator
    CPU计算逻辑实现：dequant(x*scale) → all_reduce → 保存结果
    """

    # 类常量：数据类型范围映射（所有实例共享）
    DTYPE_RANGE: Dict[Type, tuple] = {
        np.int8: (-128, 127),
        np.int16: (-32768, 32767),
        np.float16: (-65504, 65504),
        np.float32: (-1e38, 1e38),
        bfloat16: (-1e38, 1e38),
        float8_e5m2: (1e-9, 1e6),
        float8_e8m0fnu: (2**-126, 2**127),
    }

    # 类常量：数据类型映射表（命令行参数→实际类型）
    TYPE_MAP: Dict[str, Type] = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "float32_t": np.float32,
        "int8_t": np.int8,
        "fp8_e5m2_t": float8_e5m2,
        "fp8_e8m0_t": float8_e8m0fnu,
        "bfloat16_t": bfloat16,
    }

    def __init__(self, rank: int, args: argparse.Namespace):
        """
        初始化generator
        :param args: 命令行参数Namespace对象
        """
        # 保存命令行参数为实例属性
        self.case_name = args.case_name
        self.bs = args.bs
        self.hidden_size = args.hidden_size
        self.input_tensor_range = args.input_tensor_range
        self.input_tensor_type = self.TYPE_MAP.get(args.input_tensor_type, np.float16)
        self.scales_range = args.scales_range
        self.scales_type = self.TYPE_MAP.get(args.scales_type, np.float16)
        self.output_type = self.TYPE_MAP.get(args.output_type, np.float16)
        self.ranksize = args.ranksize
        self.reduce_op = args.reduce_op.lower()  # 统一小写，避免大小写问题
        self.is_mxfp = args.mxfp
        self.seed = args.seed
        # 计算输入/scale长度（适配mxFp模式）
        self.input_len = self.bs * self.hidden_size
        self.scale_len = self._calc_scale_len()
        # 初始化Golden数据目录（对齐参考代码的output_path）
        self.output_path = (
            f"./golden/quantallreduce_{self.case_name}_{self.bs}_{self.hidden_size}"
        )
        self._init_golden_dir()
        # 设置随机种子（保证结果可复现）
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # CPU分布式相关属性
        self.rank = rank  # 当前rank（默认0，实际由分布式环境赋值）
        self.x = None  # 输入x张量
        self.scale = None  # 输入scale张量
        self._init_distributed_env()  # 初始化CPU分布式环境

    def gen_random_data(
        self, size: int, dtype: Union[np.dtype, Type], drange: str
    ) -> np.ndarray:
        """
        根据数据类型和数据范围生成随机数据
        :param size: 数据长度
        :param dtype: 目标数据类型
        :param drange: 数值范围
        :return: 符合要求的随机数组
        """
        # 解析数值范围
        try:
            low, high = map(float, drange.split())
            if low >= high:
                raise ValueError(f"无效的数值范围! low={low} >= high={high}")
        except Exception as e:
            raise ValueError(f"解析数值范围失败：{e}") from e
        # 校验数据类型并裁剪范围
        if dtype not in self.DTYPE_RANGE:
            clip_low, clip_high = low, high
            logging.info(f"no preset range for {dtype}, using input range: {drange}")
        else:
            dtype_low, dtype_high = self.DTYPE_RANGE[dtype]
            clip_low = max(low, dtype_low)
            clip_high = min(high, dtype_high)
            if clip_low >= clip_high:
                raise ValueError(
                    f"{dtype}的预设范围{self.DTYPE_RANGE[dtype]}与输入范围{drange}无交集"
                )
        # 生成随机数（先按float32生成，再做类型转换）
        random_data = np.random.uniform(low=clip_low, high=clip_high, size=size).astype(
            np.float32
        )
        # 特殊类型处理（适配FP8格式）
        if dtype == float8_e8m0fnu:
            log2_data = np.log2(np.abs(random_data) + 1e-10)  # 避免log2(0)报错
            round_log2 = np.round(log2_data)
            random_data = np.power(2, round_log2) * np.sign(random_data)
        elif "float8" in str(dtype):
            random_data = np.clip(random_data, dtype_low, dtype_high)
        # 转换为目标类型并检查NaN
        target_data = random_data.astype(dtype)
        nan_count = np.isnan(target_data.astype(np.float32)).sum()
        if nan_count > 0:
            raise RuntimeError(f"生成的数据包含{nan_count}个NaN值, 请检查参数!")
        return target_data

    def input_generate(
        self, data_name: str, data_len: int, data_type: Type, drange: str
    ) -> torch.Tensor:
        """
        生成输入数据并保存为bin文件, 返回torch张量
        :param data_name: 数据名称
        :param data_len: 数据长度
        :param data_type: 数据类型
        :param drange: 数值范围
        :return: 当前rank的输入torch张量
        """
        # 每个rank生成独立的随机数据
        input_np = self.gen_random_data(data_len, dtype=data_type, drange=drange)
        # 保存当前rank的bin文件
        file_path = os.path.join(self.output_path, f"input_{data_name}_{self.rank}.bin")
        input_np.tofile(file_path)
        # 转换为torch float32张量（对齐参考代码的x.to(torch.float32)）
        input_tensor = torch.from_numpy(input_np).to(torch.float32).cpu()
        # 仅主rank打印日志
        logging.info(f"{data_name} data generated, path: {self.output_path}")
        return input_tensor

    def cpu_dequant(
        self, x: np.ndarray, scale: np.ndarray, group_size: int
    ) -> torch.Tensor:
        """
        完全对齐参考代码的反量化逻辑
        :param x: 输入numpy数组
        :param scale: 缩放因子numpy数组
        :param group_size: 分组大小
        :return: 反量化后的torch张量
        """
        repeated_scale = np.repeat(scale, group_size, axis=-1)
        x = torch.from_numpy(x)
        repeated_scale = torch.from_numpy(repeated_scale)
        return x * repeated_scale

    def save(self, tensor: torch.Tensor, save_path: str, file_name: str) -> None:
        """
        对齐参考代码的保存逻辑: 保存torch张量为bin文件
        :param tensor: 要保存的张量
        :param save_path: 保存目录
        :param file_name: 文件名
        """
        # 转换为目标输出类型的numpy数组
        save_np = tensor.numpy().astype(self.output_type)
        # 保存为bin文件
        save_file = os.path.join(save_path, file_name)
        save_np.tofile(save_file)
        logging.info(f"Rank {self.rank}: results saved to {save_file}")

    def get_cpu(self) -> torch.Tensor:
        """
        完全对齐参考代码的CPU计算核心逻辑
        :return: all_reduce后的输出张量
        """
        # 1. 构建reduce操作映射（对齐参考代码）
        reduce_op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        op = reduce_op_map.get(self.reduce_op, dist.ReduceOp.SUM)  # 默认sum
        # 2. 转换为numpy（对齐参考代码的x.numpy()/scale.numpy()）
        x_np = self.x.numpy()
        scale_np = self.scale.numpy()
        # 3. 执行反量化
        group_size = 32 if self.is_mxfp else 128
        output = self.cpu_dequant(x_np, scale_np, group_size)
        # 4. 执行all_reduce
        dist.all_reduce(output, op)
        # 6. 保存结果（仅主rank保存，避免多rank覆盖）
        self.save(output, self.output_path, f"output_cpu_{self.rank}.bin")
        return output

    def run(self):
        """
        RUN 完整执行流程：生成数据 → 执行CPU计算 → 输出结果
        """
        # 1. 生成x输入数据并赋值给self.x（对齐参考代码）
        self.x = self.input_generate(
            data_name="x",
            data_len=self.input_len,
            data_type=self.input_tensor_type,
            drange=self.input_tensor_range,
        )
        # 2. 生成scale输入数据并赋值给self.scale（对齐参考代码）
        self.scale = self.input_generate(
            data_name="scale",
            data_len=self.scale_len,
            data_type=self.scales_type,
            drange=self.scales_range,
        )
        # 3. 打印数据形状
        logging.info(f"self.x.shape: {self.x.shape}")
        logging.info(f"self.scale.shape: {self.scale.shape}")
        # 4. 执行核心CPU计算逻辑
        output_cpu = self.get_cpu()
        logging.info(f"output_cpu.shape: {output_cpu.shape}")
        logging.info(f"all golden data generated, save dir: {self.output_path}")
        # 6. 销毁分布式环境（仅主rank执行）
        if self.rank == 0:
            dist.destroy_process_group()
            logging.info("CPU distributed env destroyed")
        return output_cpu

    def _calc_scale_len(self) -> int:
        """私有方法: 根据不同的量化类型计算scale长度"""
        if self.is_mxfp == 0:
            return self.bs * (self.hidden_size // 128)
        else:
            return self.bs * (self.hidden_size // 64) * 2

    def _init_golden_dir(self) -> None:
        """私有方法: 初始化golden的数据存放目录"""
        # 创建目录（不存在则创建，存在则不报错）
        os.makedirs(self.output_path, exist_ok=True)
        # 清空目录原有文件（避免残留旧数据）
        for file in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _init_distributed_env(self):
        """私有方法: 初始化CPU分布式环境"""
        # 设置环境变量（每个进程都要设置）
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.ranksize)
        # 初始化Gloo后端（CPU）
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.ranksize,
                init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
            )
        logging.info(
            f"Rank {self.rank}: distributed env initialized (total processes: {self.ranksize})"
        )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="quant_all_reduce golden generator (对齐指定CPU逻辑)"
    )
    parser.add_argument("case_name", type=str, help="测试用例名称")
    parser.add_argument("bs", type=int, help="Batch Size")
    parser.add_argument("hidden_size", type=int, help="Hidden Size")
    parser.add_argument("input_tensor_range", type=str, help="input tensor范围")
    parser.add_argument("input_tensor_type", type=str, help="input tensor类型")
    parser.add_argument("scales_range", type=str, help="scale tensor范围")
    parser.add_argument("scales_type", type=str, help="scale tensor类型")
    parser.add_argument("output_type", type=str, help="输出类型")
    parser.add_argument("ranksize", type=int, help="Rank数量")
    parser.add_argument("reduce_op", type=str, help="Reduce操作 (sum/max/min)")
    parser.add_argument("mxfp", type=int, help="MXFP模式 (0/1)")
    parser.add_argument("seed", type=int, help="随机种子（保证结果可复现）")
    return parser.parse_args()


def run_worker(rank: int, args: argparse.Namespace):
    """每个进程的执行函数"""
    try:
        generator = QuantAllReduceGoldenGenerator(rank, args)
        generator.run()
    except Exception as e:
        logging.error(f"Rank {rank}: execution failed, error: {e}")
        raise


if __name__ == "__main__":
    """主函数：解析参数 → 实例化生成器 → 执行指定CPU逻辑"""
    # 1. 解析命令行参数
    args = parse_args()
    # 2. 设置多进程启动方式（避免Windows/Linux兼容问题）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    # 3. 启动多个进程
    processes = []
    for rank in range(args.ranksize):
        # 关键：给每个进程传 rank 和 args 两个参数
        p = mp.Process(target=run_worker, args=(rank, args))
        p.start()
        processes.append(p)
    # 4. 等待所有进程完成
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"进程 {p.pid} 执行失败，退出码：{p.exitcode}")
    # 5. 最终提示
    logging.info("\n===== all processes completed =====")
    logging.info(
        f"golden data save directory: ./golden/quantallreduce_{args.case_name}_{args.bs}_{args.hidden_size}"
    )

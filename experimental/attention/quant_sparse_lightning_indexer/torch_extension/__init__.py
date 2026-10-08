# 注意: 本包目录名、包内模块名与模块内导出的入口函数名同为 quant_sparse_lightning_indexer,
# 因此必须显式从子模块导入(不能用 `from . import <同名>` 或包级 __getattr__ 懒加载, 会自递归);
# 导入后本包的 quant_sparse_lightning_indexer 属性是入口函数(覆盖同名子模块的模块属性), 这是有意为之。
from .quant_sparse_lightning_indexer import (  # noqa: F401
    quant_sparse_lightning_indexer,
)

__all__ = ["quant_sparse_lightning_indexer"]

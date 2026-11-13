#!/usr/bin/env python3
"""
PG-UNAGI: Prior-Guided UNAGI — 领域可插拔先验嵌入框架

模块化结构:
  pg_unagi/
    __init__.py
    config.py          — 全局配置、TF基因集、颜色方案
    tf_database.py     — TRRUST/STRING 数据库下载与先验矩阵构建
    patcher.py         — UNAGI 源码克隆与三项补丁应用
    wrapper.py         — CustomUNAGIWrapper 核心训练封装
    scripts/
      train_neuro.py   — UNAGI-Neuro (GSE153162 神经发育) 训练脚本
      train_lung.py    — UNAGI-Lung (GSE136831 肺纤维化) 训练脚本
      train_blood.py   — UNAGI-Blood (造血分化) 训练脚本
"""

# 这个 __init__.py 只标记模块存在
__version__ = "1.0.0"
__author__ = "Andre et al."

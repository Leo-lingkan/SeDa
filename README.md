# SeDa - Scientific Dataset Aggregation System

> **说明**：本仓库展示了数据集管理系统中的三个核心模块：**死链检测**、**数据去重**和**主题标签生成**。这些模块是完整系统的关键组成部分，用于保障数据集的可用性、唯一性和可发现性。

---

## 模块说明

### 1. 死链检测模块 (deadlink/)

**功能**：基于自适应分层采样的数据源存活性监控，持续验证数据集URL的可访问性。

**执行入口**：`deadlink/deadlink_detection.py`

**配置文件**：`deadlink/config.py`

---

### 2. 数据去重模块 (deduplication/)

**功能**：基于 SimHash + LSH + 语义向量的大规模数据集去重，支持分批并行处理。

**执行入口**：`deduplication/run_deduplicate.py`

**配置文件**：`deduplication/config.py`

---

### 3. 主题标签模块 (tagging/)

**功能**：基于频率-位置加权、共现图扩散和LLM语义过滤的主题标签生成与关联。

**执行入口**：`tagging/run_tagging.py`

**配置文件**：`tagging/config.py`

---

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置数据库
编辑各模块的 `config.py` 文件，修改 `MYSQL_CONFIG` 中的数据库连接信息。

### 运行模块
```bash
# 死链检测
python deadlink/deadlink_detection.py

# 数据去重
python deduplication/run_deduplicate.py

# 主题标签生成
python tagging/run_tagging.py
```

---

## 项目结构

```
SeDa/
├── deadlink/              # 死链检测模块
│   ├── config.py
│   ├── deadlink_core.py
│   ├── deadlink_repo.py
│   └── deadlink_detection.py
│
├── deduplication/         # 数据去重模块
│   ├── config.py
│   ├── deduplicate.py
│   └── run_deduplicate.py
│
├── tagging/               # 主题标签模块
│   ├── config.py
│   ├── tag_extraction.py
│   ├── tag_graph.py
│   ├── tag_engine.py
│   └── run_tagging.py
│
├── requirements.txt
└── README.md
```


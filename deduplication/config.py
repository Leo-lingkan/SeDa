# -*- coding: utf-8 -*-

# ===== 数据字段定义 =====
DATASET_NAME = "dataset_name"
DATASET_DESC = "dataset_desc"
DATASET_URL  = "dataset_url"
EXT_HOST     = "source"
ID_FIELD     = "id"
SELECTED_FIELDS = [DATASET_NAME, DATASET_DESC, DATASET_URL, EXT_HOST]

# ===== Elasticsearch 配置 =====
ES_HOSTS = ["http://127.0.0.1:9200"]
ES_INDEX = "seda_dataset"
ES_SCROLL_SIZE = 5000
ES_SCROLL_TTL = "3m"

# ===== Milvus 配置 =====
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "seda_dataset_vectors"
MILVUS_VECTOR_FIELD = "embedding"
MILVUS_ID_FIELD = "id"
MILVUS_FETCH_BATCH = 3000

# ===== 算法参数 =====
SIMHASH_BITS   = 64
SIMHASH_BANDS  = 8
LSH_NUM_PLANES = 16
SIM_THRESHOLD  = 0.82
TOPK_PER_QUERY = 10
MAX_BUCKET_SIZE = 500  # 桶大小上限

# ===== 批量与并行控制 =====
BATCH_SIZE = 5000
MAX_WORKERS = 4  # 批内并行数

# ===== 代表记录选择指标 =====
REP_FIELDS_FOR_COMPLETENESS = [DATASET_NAME, DATASET_DESC, DATASET_URL, EXT_HOST]

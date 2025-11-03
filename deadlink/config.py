MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "seda",
    "charset": "utf8mb4",
}

# Elasticsearch配置
ES_HOSTS = ["http://127.0.0.1:9200"]
ES_INDEX = "seda_dataset"

API_URL = "http://127.0.0.1:18080/internal-api/gateway/radar-number-five/dataset/sample-list"

# API二次验证站点
API_VERIFIED_SITES = ["opendatalab.org.cn"]

# 浏览器渲染站点
BROWSER_RENDERED_SITES = ["example.com"]

TAU = 0.9              # 存活率阈值 τ
K_MIN = 20             # k_min
C_MAX = 2.0            # k_max 系数 c（论文范围 [1.5, 2.5]）
BUDGET = 2000          # K_total
LAMBDA1 = 3.0          # λ₁ 历史波动敏感度
LAMBDA2 = 5.0          # λ₂ 增长活跃度敏感度
EPS = 1e-6             # ε 数值稳定项
TIMEOUT = 5.0
RETRY = 1
FAIL_THRESHOLD = 3     # 连续失败阈值（触发隐藏前的检测轮数）

TABLE_SEDA_SOURCE = "seda_source"
TABLE_LINK_DETECT = "link_detect"
TABLE_DATASET_HOST = "source_status"

COL_SOURCE_NAME = "source_name"

ES_FIELD_SOURCE = "source"
ES_FIELD_VISIBLE = "application_visibility"
ES_FIELD_TRACE = "traceability"

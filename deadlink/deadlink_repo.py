import pymysql
import uuid, time
from datetime import datetime
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch, helpers
from config import (
    MYSQL_CONFIG, ES_HOSTS, ES_INDEX,
    TABLE_SEDA_SOURCE, TABLE_LINK_DETECT, TABLE_DATASET_HOST,
    COL_SOURCE_NAME, ES_FIELD_SOURCE, ES_FIELD_VISIBLE, ES_FIELD_TRACE
)

class MySQL:
    def __init__(self):
        self.cfg = MYSQL_CONFIG
    def _conn(self):
        return pymysql.connect(**self.cfg)

class SiteRepo(MySQL):
    def fetch_sites(self) -> List[str]:
        sql = f"SELECT ext_host_name FROM {TABLE_DATASET_HOST} WHERE application_visibility=1 OR application_visibility IS NULL"
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

    def fetch_site_counts(self) -> Dict[str, int]:
        """从MySQL站点表读取各站点的数据集数量（N字段）"""
        sql = f"SELECT site_name, N FROM {TABLE_SEDA_SOURCE}"
        out: Dict[str, int] = {}
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            for name, n in cur.fetchall():
                if name and n is not None:
                    out[name] = int(n)
        return out

    def fetch_site_snapshot(self, site: str) -> Optional[dict]:
        sql = f"""
        SELECT site_name, N, alive_rate, alive_var, last_N, delta_N, status,
               last_detect_date, fail_streak, is_hidden, last_hidden_date
        FROM {TABLE_SEDA_SOURCE}
        WHERE site_name=%s
        """
        with self._conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, (site,))
            return cur.fetchone()

    def upsert_site_snapshot(
        self, site: str, N: int, alive_rate: float, alive_var: float,
        delta_N: int, status: str, detect_date: datetime,
        fail_streak: int, is_hidden: int, last_hidden_date: Optional[datetime]
    ):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_SEDA_SOURCE} WHERE site_name=%s", (site,))
            exists = cur.fetchone()[0] > 0
            now = datetime.now()
            if exists:
                sql = f"""
                UPDATE {TABLE_SEDA_SOURCE}
                SET N=%s, alive_rate=%s, alive_var=%s, last_N=%s, delta_N=%s,
                    status=%s, last_detect_date=%s, fail_streak=%s, is_hidden=%s,
                    last_hidden_date=%s, update_time=%s
                WHERE site_name=%s
                """
                cur.execute(sql, (N, alive_rate, alive_var, N, delta_N, status,
                                  detect_date, fail_streak, is_hidden, last_hidden_date, now, site))
            else:
                code = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{site}-{time.time()}")).replace("-", "")
                sql = f"""
                INSERT INTO {TABLE_SEDA_SOURCE}
                (id, site_name, N, alive_rate, alive_var, last_N, delta_N,
                 status, last_detect_date, fail_streak, is_hidden, last_hidden_date,
                 create_time, update_time)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """
                cur.execute(sql, (code, site, N, alive_rate, alive_var, N, delta_N,
                                  status, detect_date, fail_streak, is_hidden,
                                  last_hidden_date, now, now))
            conn.commit()

    def insert_link_history(self, site: str, link_num: int, link_rate: float, status: str, detect_date: datetime):
        with self._conn() as conn, conn.cursor() as cur:
            code = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{site}-{detect_date.timestamp()}")).replace("-", "")
            sql = f"""
            INSERT INTO {TABLE_LINK_DETECT}
            (id, ext_host_name, link_num, link_rate, status, detect_date, create_time, update_time)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """
            now = datetime.now()
            cur.execute(sql, (code, site, link_num, link_rate, status, detect_date, now, now))
            conn.commit()

    def fetch_recent_link_rates(self, site: str, window: int = 5) -> List[float]:
        sql = f"""
        SELECT link_rate FROM {TABLE_LINK_DETECT}
        WHERE ext_host_name=%s ORDER BY detect_date DESC LIMIT %s
        """
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (site, window))
            rates = [float(r[0]) for r in cur.fetchall()]
        return list(reversed(rates))

    def purge_old_history(self, keep: int = 10):
        with self._conn() as conn, conn.cursor() as cur:
            sql = f"""
            DELETE ld FROM {TABLE_LINK_DETECT} ld
            JOIN (
              SELECT ext_host_name, MIN(detect_date) AS cutoff
              FROM (
                SELECT ext_host_name, detect_date,
                       ROW_NUMBER() OVER (PARTITION BY ext_host_name ORDER BY detect_date DESC) AS rn
                FROM {TABLE_LINK_DETECT}
              ) t
              WHERE rn <= %s
              GROUP BY ext_host_name
            ) c
            ON ld.ext_host_name = c.ext_host_name AND ld.detect_date < c.cutoff
            """
            try:
                cur.execute(sql, (keep,))
                conn.commit()
            except Exception:
                pass

    def hide_site_datasets(self, site: str):
        """隐藏指定站点的所有数据集（更新ES）"""
        es = Elasticsearch(ES_HOSTS, request_timeout=120)
        
        # 查询该站点的所有数据集
        query = {
            "query": {
                "term": {
                    f"{ES_FIELD_SOURCE}.keyword": site
                }
            }
        }
        
        # 使用update_by_query批量更新
        update_body = {
            "script": {
                "source": f"ctx._source.{ES_FIELD_VISIBLE} = 0; ctx._source.{ES_FIELD_TRACE} = 'untraceable'",
                "lang": "painless"
            },
            "query": query["query"]
        }
        
        try:
            es.update_by_query(index=ES_INDEX, body=update_body, refresh=True)
        except Exception as e:
            pass

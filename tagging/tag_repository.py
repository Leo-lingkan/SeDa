# -*- coding: utf-8 -*-
import pymysql
import numpy as np
from scipy import sparse
from contextlib import contextmanager
import config as cfg


class TagRepository:
    
    def __init__(self):
        self.conn_config = cfg.MYSQL_CONFIG
    
    @contextmanager
    def _get_conn(self):
        conn = pymysql.connect(**self.conn_config)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_tables(self):
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {cfg.TABLE_TAG_VOCABULARY} (
                    tag_id INT PRIMARY KEY AUTO_INCREMENT,
                    tag_name VARCHAR(255) UNIQUE NOT NULL,
                    tag_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_tag_name (tag_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {cfg.TABLE_TAG_COOCCURRENCE} (
                    tag_id_1 INT NOT NULL,
                    tag_id_2 INT NOT NULL,
                    weight FLOAT NOT NULL,
                    PRIMARY KEY (tag_id_1, tag_id_2),
                    FOREIGN KEY (tag_id_1) REFERENCES {cfg.TABLE_TAG_VOCABULARY}(tag_id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id_2) REFERENCES {cfg.TABLE_TAG_VOCABULARY}(tag_id) ON DELETE CASCADE,
                    INDEX idx_tag1 (tag_id_1),
                    INDEX idx_tag2 (tag_id_2)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            
            conn.commit()
    
    def save_vocabulary(self, tag_vocab, tag_counts):
        """返回 tag_name -> db_tag_id 映射"""
        with self._get_conn() as conn, conn.cursor() as cur:
            tag_to_id = {}
            
            for tag in tag_vocab:
                count = tag_counts.get(tag, 0)
                cur.execute(f"""
                    INSERT INTO {cfg.TABLE_TAG_VOCABULARY} (tag_name, tag_count)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE tag_count = %s
                """, (tag, count, count))
                
                cur.execute(f"""
                    SELECT tag_id FROM {cfg.TABLE_TAG_VOCABULARY} WHERE tag_name = %s
                """, (tag,))
                
                tag_id = cur.fetchone()[0]
                tag_to_id[tag] = tag_id
            
            conn.commit()
            return tag_to_id
    
    def save_cooccurrence_edges(self, edges):
        if not edges:
            return
        
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {cfg.TABLE_TAG_COOCCURRENCE}")
            cur.executemany(f"""
                INSERT INTO {cfg.TABLE_TAG_COOCCURRENCE} (tag_id_1, tag_id_2, weight)
                VALUES (%s, %s, %s)
            """, edges)
            conn.commit()
    
    def load_vocabulary(self):
        """返回 (tag_vocab, tag_to_id[连续索引], db_id_to_idx, tag_counts)"""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT tag_id, tag_name, tag_count
                FROM {cfg.TABLE_TAG_VOCABULARY}
                ORDER BY tag_id
            """)
            
            rows = cur.fetchall()
            tag_vocab = []
            tag_to_id = {}
            db_id_to_idx = {}
            tag_counts = {}
            
            for idx, (db_tag_id, tag_name, tag_count) in enumerate(rows):
                tag_vocab.append(tag_name)
                tag_to_id[tag_name] = idx
                db_id_to_idx[db_tag_id] = idx
                tag_counts[tag_name] = tag_count
            
            return tag_vocab, tag_to_id, db_id_to_idx, tag_counts
    
    def load_cooccurrence_sparse_matrix(self, db_id_to_idx, n_tags):
        """加载共现边并构建行归一化稀疏邻接矩阵"""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT tag_id_1, tag_id_2, weight
                FROM {cfg.TABLE_TAG_COOCCURRENCE}
            """)
            edges = cur.fetchall()
        
        if not edges:
            return sparse.csr_matrix((n_tags, n_tags))
        
        rows, cols, data = [], [], []
        for db_id_1, db_id_2, weight in edges:
            idx1 = db_id_to_idx.get(db_id_1)
            idx2 = db_id_to_idx.get(db_id_2)
            if idx1 is not None and idx2 is not None:
                rows.extend([idx1, idx2])
                cols.extend([idx2, idx1])
                data.extend([weight, weight])
        
        A = sparse.coo_matrix((data, (rows, cols)), shape=(n_tags, n_tags)).tocsr()
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        A_norm = sparse.diags(1.0 / row_sums) @ A
        
        return A_norm
    
    def add_new_tags(self, new_tags):
        """增量添加新标签（孤立节点）"""
        with self._get_conn() as conn, conn.cursor() as cur:
            tag_to_id = {}
            for tag in new_tags:
                cur.execute(f"""
                    INSERT IGNORE INTO {cfg.TABLE_TAG_VOCABULARY} (tag_name, tag_count)
                    VALUES (%s, 0)
                """, (tag,))
                cur.execute(f"""
                    SELECT tag_id FROM {cfg.TABLE_TAG_VOCABULARY} WHERE tag_name = %s
                """, (tag,))
                result = cur.fetchone()
                if result:
                    tag_to_id[tag] = result[0]
            conn.commit()
            return tag_to_id
    
    def get_tag_count(self):
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {cfg.TABLE_TAG_VOCABULARY}")
            return cur.fetchone()[0]


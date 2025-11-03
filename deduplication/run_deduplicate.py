# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch, helpers
import config as cfg
import deduplicate as engine

log_file = datetime.now().strftime('%Y%m%d_deduplication.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def iter_es_batches(batch_size=cfg.BATCH_SIZE):
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    query = {"query": {"match_all": {}}}
    page = es.search(index=cfg.ES_INDEX, body=query, size=batch_size, scroll=cfg.ES_SCROLL_TTL)
    scroll_id = page["_scroll_id"]
    while True:
        hits = page["hits"]["hits"]
        if not hits:
            break
        batch = []
        for h in hits:
            src = h["_source"]
            rid = str(src.get(cfg.ID_FIELD) or h["_id"])
            src[cfg.ID_FIELD] = rid
            batch.append(src)
        yield batch
        page = es.scroll(scroll_id=scroll_id, scroll=cfg.ES_SCROLL_TTL)
    es.clear_scroll(scroll_id=scroll_id)

def process_batch(records):
    rec_idx = {}
    for r in records:
        rid = str(r.get(cfg.ID_FIELD))
        r[cfg.DATASET_NAME] = engine.normalize_name(r.get(cfg.DATASET_NAME, ""))
        r[cfg.DATASET_URL] = engine.canonicalize_url(r.get(cfg.DATASET_URL, ""))
        rec_idx[rid] = r
    
    explicit_pairs = engine.stage_explicit_match(rec_idx)
    sim_map, lsh_map, rid_list = engine.stage_blocking(rec_idx)
    semantic_pairs = engine.stage_semantic(rec_idx, sim_map, lsh_map, rid_list)
    all_pairs = explicit_pairs | semantic_pairs
    merged, clusters, mapping = engine.stage_merge(rec_idx, all_pairs)
    return merged, mapping

def global_merge(all_batches):
    logging.info(f"全局合并开始，共 {len(all_batches)} 个批次")
    
    global_mapping = {}
    for merged, mapping in all_batches:
        global_mapping.update(mapping)
    
    bucket_groups = {}
    for merged, mapping in all_batches:
        for rec in merged:
            name = rec.get(cfg.DATASET_NAME, "")
            url = rec.get(cfg.DATASET_URL, "")
            bucket_key = hash(f"{name}:{url}") % 100
            if bucket_key not in bucket_groups:
                bucket_groups[bucket_key] = []
            bucket_groups[bucket_key].append(rec)
    
    logging.info(f"分为 {len(bucket_groups)} 个桶进行去重")
    
    all_final_records = []
    bucket_mappings = {}
    processed = 0
    total_buckets = len(bucket_groups)
    
    for bucket_id, records in bucket_groups.items():
        processed += 1
        if processed % 20 == 0 or processed == total_buckets:
            logging.info(f"全局合并进度: {processed}/{total_buckets}")
        
        if len(records) <= 1:
            all_final_records.extend(records)
            continue
        
        rec_idx = {r[cfg.ID_FIELD]: r for r in records}
        explicit_pairs = engine.stage_explicit_match(rec_idx)
        sim_map, lsh_map, rid_list = engine.stage_blocking(rec_idx)
        semantic_pairs = engine.stage_semantic(rec_idx, sim_map, lsh_map, rid_list)
        all_pairs = explicit_pairs | semantic_pairs
        merged, clusters, mapping = engine.stage_merge(rec_idx, all_pairs)
        all_final_records.extend(merged)
        bucket_mappings.update(mapping)
    
    logging.info(f"全局合并完成，最终实体数: {len(all_final_records)}")
    
    final_mapping = {}
    for orig_id, local_rep in global_mapping.items():
        final_rep = bucket_mappings.get(local_rep, local_rep)
        final_mapping[orig_id] = final_rep
    
    return all_final_records, final_mapping

def es_bulk_update(mapping):
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    actions = [
        {
            "_op_type": "update",
            "_index": cfg.ES_INDEX,
            "_id": rid,
            "doc": {"canonical_id": rep},
            "doc_as_upsert": True
        }
        for rid, rep in mapping.items()
    ]
    helpers.bulk(es, actions, chunk_size=2000, request_timeout=120)
    logging.info(f"写回ES完成，更新 {len(actions)} 条记录")

def run_parallel_dedup():
    logging.info("去重任务启动")
    
    results = []
    batch_count = 0
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
        futures = []
        for batch in iter_es_batches(cfg.BATCH_SIZE):
            futures.append(executor.submit(process_batch, batch))
            batch_count += 1
        
        logging.info(f"已提交 {batch_count} 个批次任务")
        completed = 0
        for f in as_completed(futures):
            results.append(f.result())
            completed += 1
            if completed % 50 == 0 or completed == batch_count:
                logging.info(f"批次处理进度: {completed}/{batch_count}")
    
    logging.info("批次处理完成")
    
    merged_records, final_mapping = global_merge(results)
    
    logging.info("写回ES")
    es_bulk_update(final_mapping)
    logging.info(f"去重任务完成，最终实体数: {len(merged_records)}")

if __name__ == "__main__":
    run_parallel_dedup()

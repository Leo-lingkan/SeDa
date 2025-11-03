# -*- coding: utf-8 -*-
import logging
import pickle
from datetime import datetime
from elasticsearch import Elasticsearch
import config as cfg
import tag_extraction as extraction
import tag_graph as graph
import tag_engine as engine
import tag_repository as repo

log_file = datetime.now().strftime('%Y%m%d_tagging.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def save_tag_embeddings_cache(tag_embeddings, filepath="tag_embeddings.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(tag_embeddings, f)
    logging.info(f"标签向量缓存已保存至 {filepath}")

def load_tag_embeddings_cache(filepath="tag_embeddings.pkl"):
    try:
        with open(filepath, 'rb') as f:
            tag_embeddings = pickle.load(f)
        logging.info(f"标签向量缓存已从 {filepath} 加载")
        return tag_embeddings
    except FileNotFoundError:
        logging.warning(f"标签向量缓存文件 {filepath} 不存在，将从Milvus重新加载")
        return None

def get_tagged_dataset_ids():
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    query = {"query": {"exists": {"field": cfg.TAGS_FIELD}}}
    
    tagged_ids = set()
    page = es.search(index=cfg.ES_INDEX, body=query, size=1000, scroll=cfg.ES_SCROLL_TTL)
    scroll_id = page["_scroll_id"]
    
    while True:
        hits = page["hits"]["hits"]
        if not hits:
            break
        
        for hit in hits:
            dataset_id = str(hit["_source"].get(cfg.ID_FIELD) or hit["_id"])
            tagged_ids.add(dataset_id)
        
        page = es.scroll(scroll_id=scroll_id, scroll=cfg.ES_SCROLL_TTL)
    
    es.clear_scroll(scroll_id=scroll_id)
    logging.info(f"已有 {len(tagged_ids)} 个数据集存在标签")
    return tagged_ids

def load_seed_datasets():
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    query = {"query": {"match_all": {}}, "size": cfg.SEED_DATASET_SIZE}
    result = es.search(index=cfg.ES_INDEX, body=query)
    
    datasets = []
    for hit in result["hits"]["hits"]:
        src = hit["_source"]
        src[cfg.ID_FIELD] = str(src.get(cfg.ID_FIELD) or hit["_id"])
        datasets.append(src)
    
    logging.info(f"加载 {len(datasets)} 个种子数据集")
    return datasets

def load_all_datasets():
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    query = {"query": {"match_all": {}}}
    
    datasets = []
    page = es.search(index=cfg.ES_INDEX, body=query, size=1000, scroll=cfg.ES_SCROLL_TTL)
    scroll_id = page["_scroll_id"]
    
    while True:
        hits = page["hits"]["hits"]
        if not hits:
            break
        
        for hit in hits:
            src = hit["_source"]
            src[cfg.ID_FIELD] = str(src.get(cfg.ID_FIELD) or hit["_id"])
            datasets.append(src)
        
        page = es.scroll(scroll_id=scroll_id, scroll=cfg.ES_SCROLL_TTL)
    
    es.clear_scroll(scroll_id=scroll_id)
    logging.info(f"加载 {len(datasets)} 个数据集")
    return datasets

def build_seed_vocabulary(seed_datasets):
    all_tags = set()
    
    for dataset in seed_datasets:
        title = dataset.get(cfg.TITLE_FIELD, "")
        description = dataset.get(cfg.DESCRIPTION_FIELD, "")
        
        single_tags = extraction.extract_candidate_tags(description, title)
        phrase_tags = extraction.extract_phrase_tags(description)
        merged = extraction.merge_candidates(single_tags, phrase_tags)
        
        all_tags.update(merged)
    
    tag_vocab = list(all_tags)
    logging.info(f"提取 {len(tag_vocab)} 个候选标签")
    return tag_vocab

def build_tag_graph_system(datasets, tag_vocab):
    logging.info("构建标签共现图")
    repository = repo.TagRepository()
    repository.init_tables()
    
    tag_embeddings = engine.get_tag_embeddings(tag_vocab)
    if tag_embeddings:
        normalized_tags, merge_map = graph.normalize_tags_by_embedding(tag_vocab, tag_embeddings)
        logging.info(f"同义词合并后剩余 {len(normalized_tags)} 个标签")
    else:
        normalized_tags = tag_vocab
    
    db_tag_to_id = repository.save_vocabulary(normalized_tags, {})
    tag_counts, edges = graph.build_cooccurrence_graph(datasets, normalized_tags, db_tag_to_id)
    logging.info(f"标签图构建完成：{len(normalized_tags)} 节点, {len(edges)} 边")
    
    repository.save_cooccurrence_edges(edges)
    db_tag_to_id = repository.save_vocabulary(normalized_tags, tag_counts)
    
    tag_to_id = {tag: idx for idx, tag in enumerate(normalized_tags)}
    db_id_to_idx = {db_id: idx for idx, (tag, db_id) in enumerate(db_tag_to_id.items())}
    idx_edges = [(db_id_to_idx[e[0]], db_id_to_idx[e[1]], e[2]) for e in edges]
    A_norm = graph.build_sparse_adjacency_matrix(idx_edges, len(normalized_tags))
    
    return normalized_tags, tag_to_id, A_norm, tag_embeddings

def initialize_tag_system():
    logging.info("=== 初始化模式：构建标签系统 ===")
    seed_datasets = load_seed_datasets()
    tag_vocab = build_seed_vocabulary(seed_datasets)
    all_datasets = load_all_datasets()
    tag_vocab, tag_to_id, A_norm, tag_embeddings = build_tag_graph_system(all_datasets, tag_vocab)
    save_tag_embeddings_cache(tag_embeddings)
    results = process_datasets(all_datasets, tag_vocab, tag_to_id, A_norm, tag_embeddings, generate_new_vectors=False)
    write_results_to_es(results)
    logging.info("标签系统初始化完成")

def load_new_datasets():
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    query = {
        "query": {
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": cfg.TAGS_FIELD}}}},
                    {"term": {cfg.TAGS_FIELD: ""}}
                ]
            }
        }
    }
    
    datasets = []
    page = es.search(index=cfg.ES_INDEX, body=query, size=1000, scroll=cfg.ES_SCROLL_TTL)
    scroll_id = page["_scroll_id"]
    
    while True:
        hits = page["hits"]["hits"]
        if not hits:
            break
        
        for hit in hits:
            src = hit["_source"]
            src[cfg.ID_FIELD] = str(src.get(cfg.ID_FIELD) or hit["_id"])
            datasets.append(src)
        
        page = es.scroll(scroll_id=scroll_id, scroll=cfg.ES_SCROLL_TTL)
    
    es.clear_scroll(scroll_id=scroll_id)
    logging.info(f"发现 {len(datasets)} 个新数据集")
    return datasets

def increment_tag_new_datasets():
    logging.info("=== 增量模式：为新数据集生成标签 ===")
    repository = repo.TagRepository()
    tag_vocab, tag_to_id, db_id_to_idx, tag_counts = repository.load_vocabulary()
    
    if not tag_vocab:
        logging.error("标签系统未初始化，请先运行初始化模式")
        return
    
    logging.info(f"加载标签词汇表：{len(tag_vocab)} 个标签")
    n_tags = len(tag_vocab)
    A_norm = repository.load_cooccurrence_sparse_matrix(db_id_to_idx, n_tags)
    logging.info(f"加载稀疏邻接矩阵：{A_norm.nnz} 个非零元素")
    
    tag_embeddings = load_tag_embeddings_cache()
    if tag_embeddings is None:
        tag_embeddings = engine.get_tag_embeddings(tag_vocab)
        save_tag_embeddings_cache(tag_embeddings)
    
    new_datasets = load_new_datasets()
    if not new_datasets:
        logging.info("没有新数据集需要处理")
        return
    
    results = process_datasets(new_datasets, tag_vocab, tag_to_id, A_norm, tag_embeddings, generate_new_vectors=True)
    if tag_embeddings:
        save_tag_embeddings_cache(tag_embeddings)
    write_results_to_es(results)
    logging.info("增量标签生成完成")

def process_datasets(datasets, tag_vocab, tag_to_id, A_norm, tag_embeddings, generate_new_vectors=False):
    logging.info(f"开始为 {len(datasets)} 个数据集生成标签")
    
    results = []
    processed = 0
    total = len(datasets)
    
    for dataset in datasets:
        processed += 1
        if processed % 100 == 0 or processed == total:
            logging.info(f"标签生成进度: {processed}/{total}")
        
        try:
            if generate_new_vectors:
                dataset_emb = engine.get_dataset_embedding(dataset.get(cfg.ID_FIELD))
                if dataset_emb is None:
                    logging.info(f"为新数据集 {dataset.get(cfg.ID_FIELD)} 生成向量")
                    dataset_emb = engine.generate_and_save_dataset_embedding(dataset)
            
            final_tags, new_tags = engine.generate_tags_for_dataset(
                dataset, tag_vocab, tag_to_id, A_norm, tag_embeddings
            )
            
            if new_tags:
                repository = repo.TagRepository()
                new_tag_db_ids = repository.add_new_tags(new_tags)
                for tag in new_tags:
                    if tag not in tag_vocab:
                        new_idx = len(tag_vocab)
                        tag_vocab.append(tag)
                        tag_to_id[tag] = new_idx
                new_tag_embeddings = engine.generate_and_save_tag_embeddings(new_tags)
                tag_embeddings.update(new_tag_embeddings)
                logging.info(f"新增标签: {', '.join(new_tags)}")
            
            results.append({
                "dataset_id": dataset.get(cfg.ID_FIELD),
                "tags": final_tags
            })
        
        except Exception as e:
            logging.warning(f"数据集 {dataset.get(cfg.ID_FIELD)} 标签生成失败: {e}")
            continue
    
    logging.info(f"标签生成完成，最终词汇表包含 {len(tag_vocab)} 个标签")
    return results

def write_results_to_es(results):
    es = Elasticsearch(cfg.ES_HOSTS, request_timeout=120)
    for result in results:
        es.update(
            index=cfg.ES_INDEX,
            id=result["dataset_id"],
            body={"doc": {cfg.TAGS_FIELD: result["tags"]}}
        )
    logging.info(f"写回ES完成，更新 {len(results)} 条记录")

def main(mode=cfg.MODE_INIT):
    logging.info(f"标签生成任务启动 (模式: {mode})")
    
    if mode == cfg.MODE_INIT:
        initialize_tag_system()
    elif mode == cfg.MODE_INCREMENT:
        increment_tag_new_datasets()
    else:
        logging.error(f"未知模式: {mode}")
    
    logging.info("标签生成任务完成")

if __name__ == "__main__":
    # 使用方式：
    # 1. 首次运行（初始化）：python run_tagging.py
    # 2. 增量更新：修改下面的mode参数为 cfg.MODE_INCREMENT
    
    main(mode=cfg.MODE_INIT)  # 首次运行用 MODE_INIT
    # main(mode=cfg.MODE_INCREMENT)  # 后续增量更新用 MODE_INCREMENT

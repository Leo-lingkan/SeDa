# -*- coding: utf-8 -*-
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from pymilvus import connections, Collection
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
import config as cfg
import prompts

# 全局向量模型（延迟加载）
_embedding_model = None

def _get_embedding_model():
    """获取向量模型单例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL_PATH)
    return _embedding_model


def compute_semantic_similarity(dataset_emb, tag_embeddings):
    similarities = {}
    for tag, tag_emb in tag_embeddings.items():
        sim = cosine_similarity([dataset_emb], [tag_emb])[0][0]
        similarities[tag] = float(sim)
    return similarities


def compute_lexical_similarity(dataset_text, tags):
    corpus = [dataset_text.lower().split()]
    bm25 = BM25Okapi(corpus)
    
    similarities = {}
    for tag in tags:
        query = tag.lower().split()
        score = bm25.get_scores(query)[0]
        similarities[tag] = float(score)
    
    max_score = max(similarities.values()) if similarities else 1.0
    if max_score > 0:
        similarities = {k: v / max_score for k, v in similarities.items()}
    
    return similarities


def fuse_scores(sem_scores, lex_scores, gamma=cfg.GAMMA):
    all_tags = set(sem_scores.keys()) | set(lex_scores.keys())
    fused = {}
    for tag in all_tags:
        s_sem = sem_scores.get(tag, 0.0)
        s_lex = lex_scores.get(tag, 0.0)
        fused[tag] = gamma * s_sem + (1 - gamma) * s_lex
    return fused


def mmr_selection(scores, tag_embeddings, k=cfg.NUM_SEED_TAGS, lambda_param=0.5):
    selected = []
    candidates = list(scores.keys())
    
    if not candidates:
        return selected
    
    first = max(candidates, key=lambda t: scores[t])
    selected.append(first)
    candidates.remove(first)
    
    while len(selected) < k and candidates:
        mmr_scores = {}
        for cand in candidates:
            relevance = scores[cand]
            
            max_sim = 0.0
            for sel in selected:
                if cand in tag_embeddings and sel in tag_embeddings:
                    sim = cosine_similarity([tag_embeddings[cand]], [tag_embeddings[sel]])[0][0]
                    max_sim = max(max_sim, sim)
            
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores[cand] = mmr
        
        best = max(mmr_scores, key=lambda t: mmr_scores[t])
        selected.append(best)
        candidates.remove(best)
    
    return selected


def pagerank_diffusion(A_norm, seed_tags, tag_to_id, iterations=cfg.PAGERANK_ITERATIONS, damping=cfg.LAMBDA_DAMPING):
    n = A_norm.shape[0]
    s = np.zeros(n)
    s0 = np.zeros(n)
    
    for tag in seed_tags:
        if tag in tag_to_id:
            s0[tag_to_id[tag]] = 1.0 / len(seed_tags)
    
    s = s0.copy()
    
    for _ in range(iterations):
        s = (1 - damping) * (A_norm.T @ s) + damping * s0
    
    return s


def llm_semantic_filter(dataset_name, dataset_desc, candidate_tags, top_k=2):
    client = ZhipuAI(api_key=cfg.GLM_API_KEY)
    prompt = prompts.get_tag_selection_prompt(dataset_name, dataset_desc, candidate_tags, top_k)
    
    st = time.time()
    try:
        response = client.chat.completions.create(
            model=cfg.GLM_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=cfg.GLM_TEMPERATURE,
        )
        
        parsed = json.loads(response.choices[0].message.content)
        selected_tags = parsed.get("tags", [])[:top_k]
        new_tags = parsed.get("new_tags", [])
        return selected_tags, new_tags
    except:
        return candidate_tags[:top_k], []


def get_dataset_embedding(dataset_id):
    connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    
    try:
        col = Collection(cfg.MILVUS_COLLECTION)
        expr = f"{cfg.MILVUS_ID_FIELD} == '{dataset_id}'"
        results = col.query(expr, output_fields=[cfg.MILVUS_VECTOR_FIELD])
        
        if results:
            vec = np.array(results[0][cfg.MILVUS_VECTOR_FIELD], dtype=np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
    except Exception as e:
        pass
    
    return None


def generate_and_save_dataset_embedding(dataset):
    model = _get_embedding_model()
    
    dataset_id = dataset.get(cfg.ID_FIELD)
    dataset_name = dataset.get(cfg.TITLE_FIELD, "")
    dataset_desc = dataset.get(cfg.DESCRIPTION_FIELD, "")
    
    text = f"{dataset_name}. {dataset_desc}"
    embedding = model.encode(text, convert_to_numpy=True)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
    embedding = embedding.astype(np.float32)
    connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    try:
        col = Collection(cfg.MILVUS_COLLECTION)
        data = [[dataset_id], [embedding.tolist()]]
        col.insert(data)
        col.flush()
    except Exception as e:
        pass
    
    return embedding


def get_tag_embeddings(tags):
    connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    
    try:
        col = Collection(cfg.MILVUS_TAG_COLLECTION)
    except:
        return {}
    
    tag_embeddings = {}
    for tag in tags:
        expr = f"{cfg.MILVUS_TAG_ID_FIELD} == '{tag}'"
        results = col.query(expr, output_fields=[cfg.MILVUS_VECTOR_FIELD])
        
        if results:
            vec = np.array(results[0][cfg.MILVUS_VECTOR_FIELD], dtype=np.float32)
            tag_embeddings[tag] = vec / (np.linalg.norm(vec) + 1e-9)
    
    return tag_embeddings


def generate_and_save_tag_embeddings(tags):
    if not tags:
        return {}
    
    model = _get_embedding_model()
    
    embeddings = model.encode(tags, batch_size=cfg.EMBEDDING_BATCH_SIZE, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    embeddings = embeddings.astype(np.float32)
    connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    try:
        col = Collection(cfg.MILVUS_TAG_COLLECTION)
        data = [tags, [emb.tolist() for emb in embeddings]]
        col.insert(data)
        col.flush()
    except:
        pass
    
    return {tag: emb for tag, emb in zip(tags, embeddings)}


def generate_tags_for_dataset(dataset, tag_vocab, tag_to_id, A_norm, tag_embeddings):
    dataset_id = dataset.get(cfg.ID_FIELD)
    dataset_name = dataset.get(cfg.TITLE_FIELD, "")
    dataset_desc = dataset.get(cfg.DESCRIPTION_FIELD, "")
    
    dataset_emb = get_dataset_embedding(dataset_id)
    if dataset_emb is None:
        return [], []
    
    dataset_text = f"{dataset_name} {dataset_desc}"
    sem_scores = compute_semantic_similarity(dataset_emb, tag_embeddings)
    lex_scores = compute_lexical_similarity(dataset_text, tag_vocab)
    fused_scores = fuse_scores(sem_scores, lex_scores)
    
    seed_tags = mmr_selection(fused_scores, tag_embeddings)
    
    if not seed_tags:
        return [], []
    
    diffusion_scores = pagerank_diffusion(A_norm, seed_tags, tag_to_id)
    
    top_indices = np.argsort(diffusion_scores)[::-1][:cfg.TOP_K_AFTER_DIFFUSION]
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    candidate_tags = [id_to_tag[i] for i in top_indices if i in id_to_tag]
    
    final_tags, new_tags = llm_semantic_filter(dataset_name, dataset_desc, candidate_tags)
    
    return final_tags, new_tags




# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import config as cfg


def normalize_tags_by_embedding(tags, embeddings):
    """基于嵌入相似度合并同义标签"""
    if len(tags) < 2:
        return tags, {t: t for t in tags}
    
    emb_matrix = np.array([embeddings[t] for t in tags])
    sim_matrix = cosine_similarity(emb_matrix)
    
    merged_map = {}
    used = set()
    
    for i, tag_i in enumerate(tags):
        if tag_i in used:
            continue
        
        cluster = [tag_i]
        for j, tag_j in enumerate(tags):
            if i != j and tag_j not in used and sim_matrix[i][j] >= cfg.SIMILARITY_THRESHOLD:
                cluster.append(tag_j)
                used.add(tag_j)
        
        rep = min(cluster, key=len)
        for t in cluster:
            merged_map[t] = rep
        used.add(tag_i)
    
    unique_tags = list(set(merged_map.values()))
    return unique_tags, merged_map


def build_cooccurrence_graph(datasets, tag_vocab, tag_to_id):
    """构建标签共现图，返回标签计数和边列表"""
    n_tags = len(tag_vocab)
    tag_counts_array = np.zeros(n_tags)
    cooccur_counts = defaultdict(int)
    
    for dataset in datasets:
        tags_in_doc = set()
        text = f"{dataset.get(cfg.TITLE_FIELD, '')} {dataset.get(cfg.DESCRIPTION_FIELD, '')}".lower()
        
        for tag in tag_vocab:
            if tag in text:
                tags_in_doc.add(tag)
        
        for tag in tags_in_doc:
            tag_counts_array[tag_to_id[tag]] += 1
        
        tags_list = list(tags_in_doc)
        for i in range(len(tags_list)):
            for j in range(i + 1, len(tags_list)):
                pair = tuple(sorted([tag_to_id[tags_list[i]], tag_to_id[tags_list[j]]]))
                cooccur_counts[pair] += 1
    
    tag_counts = {tag: int(tag_counts_array[tag_to_id[tag]]) for tag in tag_vocab}
    
    edges = []
    for (i, j), count in cooccur_counts.items():
        if tag_counts_array[i] > 0 and tag_counts_array[j] > 0:
            weight = count / np.sqrt(tag_counts_array[i] * tag_counts_array[j])
            if weight >= cfg.CO_OCCURRENCE_THRESHOLD:
                tag_id_1, tag_id_2 = min(i, j), max(i, j)
                edges.append((tag_id_1, tag_id_2, float(weight)))
    
    return tag_counts, edges


def build_sparse_adjacency_matrix(edges, n_tags):
    """从边列表构建行归一化的稀疏邻接矩阵"""
    if not edges:
        return sparse.csr_matrix((n_tags, n_tags))
    
    rows = []
    cols = []
    data = []
    
    for tag_id_1, tag_id_2, weight in edges:
        rows.extend([tag_id_1, tag_id_2])
        cols.extend([tag_id_2, tag_id_1])
        data.extend([weight, weight])
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_tags, n_tags)).tocsr()
    
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = sparse.diags(1.0 / row_sums) @ A
    
    return A_norm


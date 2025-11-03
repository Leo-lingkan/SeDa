# -*- coding: utf-8 -*-
import re
import hashlib
from urllib.parse import urlparse
from collections import defaultdict
import itertools
import numpy as np
import networkx as nx
from pymilvus import connections, Collection
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
import config as cfg


def normalize_name(name):
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"v\d+$", "", name)
    return name.strip()

def canonicalize_url(u):
    if not u:
        return ""
    try:
        parsed = urlparse(u)
        domain = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        return f"{domain}{path}"
    except Exception:
        return str(u).strip().lower().rstrip("/")

def concat_fields(rec, fields):
    parts = []
    for f in fields:
        val = rec.get(f, "")
        if val:
            parts.append(str(val))
    return " ".join(parts).strip().lower()

def _hash_token(token):
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    i = int(h, 16)
    return i & ((1 << cfg.SIMHASH_BITS) - 1)

def simhash(text, bits=64):
    if not text:
        return 0
    tokens = re.findall(r"\w+", text.lower())
    v = [0] * bits
    for t in tokens:
        hv = _hash_token(t)
        for i in range(bits):
            bitmask = 1 << i
            if hv & bitmask:
                v[i] += 1
            else:
                v[i] -= 1
    fp = 0
    for i in range(bits):
        if v[i] >= 0:
            fp |= 1 << i
    return fp

def simhash_bands(fp, bits=64, bands=8):
    segment = bits // bands
    keys = []
    for b in range(bands):
        part = (fp >> (b * segment)) & ((1 << segment) - 1)
        keys.append(f"band{b}:{part}")
    return keys

class LSHProjector:
    def __init__(self, dim, planes, seed=42):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(size=(planes, dim))

    def signature(self, vec):
        proj = self.P @ vec
        bits = (proj >= 0).astype(np.int8).tolist()
        v = 0
        for b in bits[::-1]:
            v = (v << 1) | int(b)
        return f"lsh:{v}"

def build_block_maps(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=20000)
    X = vec.fit_transform(texts)
    projector = LSHProjector(dim=X.shape[1], planes=cfg.LSH_NUM_PLANES)
    sim_map, lsh_map = defaultdict(list), defaultdict(list)
    for i, t in enumerate(texts):
        fp = simhash(t, bits=cfg.SIMHASH_BITS)
        for k in simhash_bands(fp, cfg.SIMHASH_BITS, cfg.SIMHASH_BANDS):
            sim_map[k].append(i)
        v = X[i].toarray().ravel()
        lkey = projector.signature(v)
        lsh_map[lkey].append(i)
    return sim_map, lsh_map

def _milvus_connect():
    try:
        connections.get_connection_addr("default")
    except Exception:
        connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)

def get_vectors_from_milvus(id_list):
    _milvus_connect()
    col = Collection(cfg.MILVUS_COLLECTION)
    id_to_vec = {}
    for i in range(0, len(id_list), cfg.MILVUS_FETCH_BATCH):
        batch = id_list[i:i+cfg.MILVUS_FETCH_BATCH]
        expr = f"{cfg.MILVUS_ID_FIELD} in {batch}"
        results = col.query(expr, output_fields=[cfg.MILVUS_ID_FIELD, cfg.MILVUS_VECTOR_FIELD])
        for r in results:
            rid = str(r[cfg.MILVUS_ID_FIELD])
            v = np.array(r[cfg.MILVUS_VECTOR_FIELD], dtype=np.float32)
            id_to_vec[rid] = v / (np.linalg.norm(v) + 1e-9)
    return id_to_vec

def cosine_sim(a, b): return float(np.dot(a, b))

def search_bucket_with_faiss(idxs, rid_list, vec_map):
    if len(idxs) < 2:
        return set()
    
    vectors = []
    valid_indices = []
    id_to_idx = {}
    
    for i in idxs:
        rid = rid_list[i]
        if rid in vec_map:
            vectors.append(vec_map[rid])
            valid_indices.append(i)
            id_to_idx[i] = len(valid_indices) - 1
    
    if len(vectors) < 2:
        return set()
    
    vectors = np.array(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    n = vectors.shape[0]
    
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    k = min(50, n)
    D, I = index.search(vectors, k)
    
    pairs = set()
    for i in range(n):
        for j_idx, sim in enumerate(D[i]):
            j = I[i][j_idx]
            if j > i and sim >= cfg.SIM_THRESHOLD:
                rid_i = rid_list[valid_indices[i]]
                rid_j = rid_list[valid_indices[j]]
                pairs.add(tuple(sorted((rid_i, rid_j))))
    
    return pairs

def search_bucket_vectors(idxs, rid_list, vec_map, depth=0):
    """大桶二次分桶，优先使用FAISS"""
    pairs = set()
    
    if depth > 3:
        return pairs
    
    # 大桶：二次分桶
    if len(idxs) > cfg.MAX_BUCKET_SIZE:
        sub_buckets = defaultdict(list)
        for i in idxs:
            if rid_list[i] in vec_map:
                vec = vec_map[rid_list[i]]
                hash_key = int(np.sum(vec[:10]) * 1000) % 50
                sub_buckets[hash_key].append(i)
        
        for sub_idxs in sub_buckets.values():
            pairs |= search_bucket_vectors(sub_idxs, rid_list, vec_map, depth + 1)
        return pairs
    
    # 小桶：FAISS或暴力比较
    if FAISS_AVAILABLE:
        return search_bucket_with_faiss(idxs, rid_list, vec_map)
    
    for ii, i in enumerate(idxs):
        vi = vec_map.get(rid_list[i])
        if vi is None: continue
        for j in idxs[ii+1:]:
            vj = vec_map.get(rid_list[j])
            if vj is None: continue
            s = cosine_sim(vi, vj)
            if s >= cfg.SIM_THRESHOLD:
                pairs.add(tuple(sorted((rid_list[i], rid_list[j]))))
    return pairs

def select_canonical(records_idx, ids):
    if not ids:
        return None
    def score(rec):
        vals = [rec.get(f, "") for f in cfg.REP_FIELDS_FOR_COMPLETENESS]
        non_empty = sum(1 for v in vals if v)
        total_len = sum(len(v) for v in vals if isinstance(v, str))
        return non_empty * 10 + total_len
    best_id = max(ids, key=lambda rid: score(records_idx[rid]))
    return best_id

def merge_attributes(base, others):
    for o in others:
        for k, v in o.items():
            if k not in base or not base[k]:
                base[k] = v
    return base

def build_components(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)
    return [list(c) for c in nx.connected_components(G)]

def stage_merge(records_idx, all_pairs):
    if not all_pairs:
        merged = list(records_idx.values())
        mapping = {rid: rid for rid in records_idx}
        return merged, [], mapping

    comps = build_components(all_pairs)
    merged, clusters, mapping = [], [], {}
    for comp in comps:
        rep_id = select_canonical(records_idx, comp)
        members = [records_idx[rid] for rid in comp]
        rep_rec = merge_attributes(records_idx[rep_id].copy(), members)
        merged.append(rep_rec)
        clusters.append(comp)
        for rid in comp:
            mapping[rid] = rep_id

    used = set(mapping)
    for rid, rec in records_idx.items():
        if rid not in used:
            merged.append(rec)
            mapping[rid] = rid
            clusters.append([rid])

    return merged, clusters, mapping

def stage_explicit_match(records_idx):
    """Stage 1: 显式标识符匹配"""
    name_map = defaultdict(list)
    url_map = defaultdict(list)
    
    for rid, rec in records_idx.items():
        name = rec.get(cfg.DATASET_NAME, "")
        url = rec.get(cfg.DATASET_URL, "")
        if name:
            name_map[name].append(rid)
        if url:
            url_map[url].append(rid)
    
    pairs = set()
    for group in name_map.values():
        if len(group) > 1:
            pairs.update(itertools.combinations(sorted(group), 2))
    for group in url_map.values():
        if len(group) > 1:
            pairs.update(itertools.combinations(sorted(group), 2))
    
    return pairs

def stage_blocking(records_idx):
    texts = [concat_fields(records_idx[rid], cfg.SELECTED_FIELDS) for rid in records_idx]
    sim_map, lsh_map = build_block_maps(texts)
    rid_list = list(records_idx.keys())
    return sim_map, lsh_map, rid_list

def stage_semantic(records_idx, sim_map, lsh_map, rid_list):
    vec_map = get_vectors_from_milvus(rid_list)
    pairs = set()
    
    for _, idxs in sim_map.items():
        pairs |= search_bucket_vectors(idxs, rid_list, vec_map)
    
    for _, idxs in lsh_map.items():
        pairs |= search_bucket_vectors(idxs, rid_list, vec_map)
    
    return pairs

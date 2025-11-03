# -*- coding: utf-8 -*-
import re
from collections import Counter
import config as cfg


def extract_candidate_tags(text, title=""):
    """频率-位置加权提取候选标签"""
    if not text:
        return []
    
    full_text = f"{title} {text}".lower()
    words = re.findall(r'\b[a-z]{3,}\b', full_text)
    if not words:
        return []
    
    term_freq = Counter(words)
    first_pos = {}
    for i, word in enumerate(words):
        if word not in first_pos:
            first_pos[word] = i + 1
    
    candidates = []
    for term, freq in term_freq.items():
        if freq < cfg.MIN_TERM_FREQ:
            continue
        pos = first_pos[term]
        weight = cfg.ALPHA * freq + cfg.BETA * (1.0 / pos)
        candidates.append((term, weight))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in candidates[:cfg.MAX_CANDIDATE_TAGS]]


def extract_phrase_tags(text, max_phrases=20):
    """复合概念提取"""
    if not text:
        return []
    
    text = text.lower()
    patterns = [
        r'\b[a-z]+\s+(?:recognition|detection|classification|generation|analysis|processing)\b',
        r'\b(?:natural|computer|machine|deep)\s+[a-z]+\b',
        r'\b[a-z]+\s+(?:learning|vision|network|model)\b',
    ]
    
    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phrases.extend(matches)
    
    unique_phrases = list(dict.fromkeys(phrases))
    return unique_phrases[:max_phrases]


def merge_candidates(single_tags, phrase_tags):
    """合并单词和短语标签"""
    all_tags = phrase_tags + single_tags
    seen = set()
    merged = []
    for tag in all_tags:
        tag_clean = tag.strip()
        if tag_clean and tag_clean not in seen:
            seen.add(tag_clean)
            merged.append(tag_clean)
    return merged


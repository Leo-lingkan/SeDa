import math, re, requests
from statistics import variance
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from config import (
    TAU, K_MIN, C_MAX, BUDGET, LAMBDA1, LAMBDA2, EPS,
    TIMEOUT, RETRY, FAIL_THRESHOLD, API_URL,
    API_VERIFIED_SITES, BROWSER_RENDERED_SITES
)

def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def fetch_sample_urls(host_name: str, size: int) -> List[Tuple[str, str]]:
    params = {'host_name': host_name, 'size': size}
    r = requests.get(API_URL, params=params, timeout=10)
    r.raise_for_status()
    result = r.json()
    if result.get('success'):
        items = result['data']['search_result']
        pairs = [(it.get('dataset_url'), it.get('dataset_name')) for it in items if it.get('dataset_url')]
        return [p for p in pairs if p[0]]
    raise ValueError(f"Failed to fetch data: {result.get('message')}")

def _head_then_get(url: str) -> tuple[int | None, bytes | None]:
    for _ in range(RETRY + 1):
        try:
            r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
            return r.status_code, None
        except requests.RequestException:
            pass
    try:
        r = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
        return r.status_code, r.content
    except requests.RequestException:
        return None, None

# API二次验证策略
def _policy_api_verified(url: str) -> bool:
    code, _ = _head_then_get(url)
    if code != 200:
        return False
    h = host_of(url)
    if "opendatalab.org.cn" in h:
        parts = url.rstrip("/").split("/")
        if len(parts) < 2:
            return False
        dataset_name = parts[-1]
        dataset_origin = parts[-2]
        second_url = f"https://opendatalab.org.cn/datasets/api/v2/datasets/{dataset_origin},{dataset_name}"
        try:
            r = requests.get(second_url, allow_redirects=True, timeout=TIMEOUT)
            return r.status_code == 200
        except requests.RequestException:
            return False
    return True

# 浏览器渲染
def _policy_browser_check(url: str, timeout: float = 12.0) -> bool:
    driver = None
    try:
        opt = Options()
        opt.add_argument("--headless=new")
        opt.add_argument("--no-sandbox")
        opt.add_argument("--disable-gpu")
        opt.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=opt)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        title = (driver.title or "").strip()
        return bool(title)
    except Exception:
        return False
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass

SITE_POLICIES_REGISTRY: List[Dict] = [
    {
        "name": "APIVerifiedSites",
        "type": "API_SECONDARY",
        "patterns": [fr"(^|\.){h}$" for h in API_VERIFIED_SITES],
        "handler": _policy_api_verified,
    },
    {
        "name": "BrowserRenderedSites",
        "type": "BROWSER_REQUIRED",
        "patterns": [fr"(^|\.){h}$" for h in BROWSER_RENDERED_SITES],
        "handler": _policy_browser_check,
    },
]

def _match_any(host: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, host):
            return True
    return False

def build_runtime_policies() -> List[Dict]:
    return list(SITE_POLICIES_REGISTRY)

def check_one_url_with_policies(url: str, policies: List[Dict]) -> bool:
    #检测URL存活状态
    if not url:
        return False
    h = host_of(url)
    for pol in policies:
        if pol["type"] == "API_SECONDARY" and _match_any(h, pol["patterns"]):
            return pol["handler"](url)
    for pol in policies:
        if pol["type"] == "BROWSER_REQUIRED" and _match_any(h, pol["patterns"]):
            return pol["handler"](url)
    code, _ = _head_then_get(url)
    return code in {200, 301, 302}

def alive_rate(bools: List[bool]) -> float:
    return round((sum(1 for b in bools if b) / len(bools)) if bools else 0.0, 4)

def k_max_from_N(N: int) -> int:
    return max(1, int(round(C_MAX * math.sqrt(max(0, N)))))

# 权重计算
def compute_weight(N: int, hist_rates: List[float], deltaN: int) -> float:
    sig2 = variance(hist_rates) if len(hist_rates) >= 2 else 0.0
    return math.log1p(max(0, N)) * (1 + LAMBDA1 * sig2) * (1 + LAMBDA2 * (deltaN / (max(N, 0) + EPS)))

# 分配
def allocate(
    sites: List[str], weights: Dict[str, float], Ns: Dict[str, int], hidden_flags: Dict[str, int]
) -> Dict[str, int]:
    W = sum(max(w, 0.0) for w in weights.values()) or 1.0
    out: Dict[str, int] = {}
    for s in sites:
        if hidden_flags.get(s, 0) == 1:
            out[s] = K_MIN
            continue
        k = int(round((weights.get(s, 0.0) / W) * BUDGET))
        k = max(K_MIN, k)
        k = min(k, k_max_from_N(Ns.get(s, 0)))
        out[s] = k
    return out

# 主流程：采样、检测、更新状态、隐藏标记
def run_once(site_repo) -> Dict[str, dict]:
    sites = site_repo.fetch_sites()
    Ns = site_repo.fetch_site_counts()
    policies = build_runtime_policies()

    deltaNs: Dict[str, int] = {}
    hist_rates_map: Dict[str, List[float]] = {}
    fail_streaks: Dict[str, int] = {}
    hidden_flags: Dict[str, int] = {}
    last_hidden_dates: Dict[str, Optional[datetime]] = {}

    for s in sites:
        snap = site_repo.fetch_site_snapshot(s)
        last_N = int(snap["last_N"]) if snap and snap.get("last_N") is not None else 0
        deltaNs[s] = int(Ns.get(s, 0)) - last_N
        hist_rates_map[s] = site_repo.fetch_recent_link_rates(s, window=5)
        fail_streaks[s] = int(snap["fail_streak"]) if snap and snap.get("fail_streak") is not None else 0
        hidden_flags[s] = int(snap["is_hidden"]) if snap and snap.get("is_hidden") is not None else 0
        last_hidden_dates[s] = snap.get("last_hidden_date") if snap else None

    weights = {s: compute_weight(Ns.get(s, 0), hist_rates_map.get(s, []), deltaNs.get(s, 0)) for s in sites}
    allocs = allocate(sites, weights, Ns, hidden_flags)

    summary: Dict[str, dict] = {}
    now = datetime.now()

    for s in sites:
        k = allocs[s]
        pairs = fetch_sample_urls(s, k)
        results = [check_one_url_with_policies(u, policies) for (u, _name) in pairs]
        rate = alive_rate(results)
        status_ok = (rate >= TAU)
        status = "合格" if status_ok else "不合格"

        site_repo.insert_link_history(s, link_num=len(pairs), link_rate=rate, status=status, detect_date=now)

        recent_rates = site_repo.fetch_recent_link_rates(s, window=5)
        sig2 = variance(recent_rates) if len(recent_rates) >= 2 else 0.0

        # 隐藏
        if status_ok:
            fail_streaks[s] = 0
        else:
            fail_streaks[s] += 1
            if fail_streaks[s] >= FAIL_THRESHOLD and hidden_flags.get(s, 0) == 0:
                site_repo.hide_site_datasets(s)
                hidden_flags[s] = 1
                last_hidden_dates[s] = now

        site_repo.upsert_site_snapshot(
            site=s, N=Ns.get(s, 0), alive_rate=rate, alive_var=round(sig2, 6),
            delta_N=deltaNs[s], status=status, detect_date=now,
            fail_streak=fail_streaks[s], is_hidden=hidden_flags[s], last_hidden_date=last_hidden_dates[s]
        )

        summary[s] = {
            "N": Ns.get(s, 0),
            "k": k,
            "delta_N": deltaNs[s],
            "alive_rate": rate,
            "alive_var": round(sig2, 6),
            "status": status,
            "fail_streak": fail_streaks[s],
            "is_hidden": hidden_flags[s],
        }

    site_repo.purge_old_history(keep=10)
    summary["_policies"] = {
        "APIVerifiedSites": API_VERIFIED_SITES,
        "BrowserRenderedSites": BROWSER_RENDERED_SITES,
        "FailThreshold": FAIL_THRESHOLD,
    }
    return summary

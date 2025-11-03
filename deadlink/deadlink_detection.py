from deadlink_repo import SiteRepo
from deadlink_core import run_once
from datetime import datetime

def main():
    repo = SiteRepo()
    summary = run_once(repo)
    print(f"=== Deadlink Check @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    for s, info in summary.items():
        if s.startswith("_"):
            continue
        print(f"{s:25s} N={info['N']:6d} ΔN={info['delta_N']:5d} k={info['k']:4d} "
              f"alive={info['alive_rate']:.2%} var={info['alive_var']:.4f} "
              f"streak={info['fail_streak']} hidden={info['is_hidden']} → {info['status']}")
    if "_policies" in summary:
        pol = summary["_policies"]
        print("\nSpecial handling applied for:")
        print(f"  • APIVerifiedSites: {pol['APIVerifiedSites'][0]}")
        print(f"  • BrowserRenderedSites: {pol['BrowserRenderedSites'][0]}")
        print(f"  • FailThreshold: {pol['FailThreshold']}")

if __name__ == "__main__":
    main()

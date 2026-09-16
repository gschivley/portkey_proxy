#!/usr/bin/env python3
"""
Print a monthly usage & cost report for the Portkey proxy.

Reads the JSONL usage log directly, so it works whether or not the proxy is
running. Costs were computed when each request completed, using rates from
Portkey's public pricing API (see usage.py).

Usage:
    python usage_report.py                 # current month
    python usage_report.py --month 2026-08
    python usage_report.py --json          # raw report as JSON
    python usage_report.py --log PATH      # alternate usage.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly usage & cost report")
    parser.add_argument("--month", help="calendar month as YYYY-MM (default: current)")
    parser.add_argument("--json", action="store_true", help="print raw JSON report")
    parser.add_argument("--log", help="usage log path (default: USAGE_LOG_PATH)")
    args = parser.parse_args()

    if args.log:
        os.environ["USAGE_LOG_PATH"] = str(Path(args.log).expanduser().resolve())
    import usage  # after env overrides (module config is read at import time)

    report = usage.monthly_report(args.month)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Usage report for {report['month']}  (log: {report['log_path']})")
    print("=" * 78)
    if not report["models"]:
        print("No usage recorded this month yet.")
    else:
        header = (
            f"{'model':<18}{'provider':<11}{'reqs':>6}{'input':>10}"
            f"{'cached':>9}{'output':>10}{'cost $':>11}"
        )
        print(header)
        print("-" * len(header))
        for row in report["models"]:
            print(
                f"{row['model']:<18}{str(row['provider'] or '-'):<11}"
                f"{row['requests']:>6}{row['prompt_tokens']:>10}"
                f"{row['cached_tokens']:>9}{row['completion_tokens']:>10}"
                f"{row['cost_usd']:>11.4f}"
            )
        print("-" * len(header))
    budget = report["budget_usd"]
    spent = report["spent_usd"]
    pct = report["pct_used"]
    print(f"  Requests        : {report['requests']}")
    if report["requests_without_pricing"]:
        print(
            f"  Without pricing : {report['requests_without_pricing']} "
            "(cost not included in totals)"
        )
    print(f"  Budget          : ${budget:.2f}")
    print(f"  Spent           : ${spent:.2f} ({pct:.1f}% used)" if pct is not None
          else f"  Spent           : ${spent:.2f}")
    remaining = report["remaining_usd"]
    if remaining is not None and remaining < 0:
        print(f"  Remaining       : ${remaining:.2f}  ** OVER BUDGET **")
    else:
        print(f"  Remaining       : ${remaining:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

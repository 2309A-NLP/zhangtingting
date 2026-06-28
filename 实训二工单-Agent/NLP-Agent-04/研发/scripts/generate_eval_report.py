#!/usr/bin/env python3
"""
评测报告生成脚本
从评测结果 JSON 生成可视化报告（HTML + Markdown）
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_eval_results(path: str) -> dict[str, Any]:
    """加载评测结果"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_results(results: list[dict]) -> dict[str, Any]:
    """分析评测结果，生成统计指标"""
    total = len(results)
    if total == 0:
        return {}

    success = sum(1 for r in results if r.get("success", False))
    failed = total - success
    success_rate = success / total * 100

    # 延迟统计
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    # 错误类型分析
    error_types: dict[str, int] = {}
    for r in results:
        if not r.get("success", False):
            error = r.get("error_message") or r.get("error") or "未知错误"
            error_types[error] = error_types.get(error, 0) + 1

    # 分类统计
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0}
        categories[cat]["total"] += 1
        if r.get("success"):
            categories[cat]["success"] += 1

    # 表使用统计
    table_usage: dict[str, int] = {}
    for r in results:
        for t in r.get("tables_used", []):
            table_usage[t] = table_usage.get(t, 0) + 1

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": round(success_rate, 2),
        "latency": {
            "avg": round(avg_latency, 2),
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
            "p99": round(p99_latency, 2),
            "min": round(min(latencies), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        },
        "error_types": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)),
        "categories": categories,
        "table_usage": dict(sorted(table_usage.items(), key=lambda x: x[1], reverse=True)),
    }


def generate_markdown(summary: dict[str, Any], stats: dict[str, Any]) -> str:
    """生成 Markdown 格式报告"""
    md = f"""# NL2SQL 基金问答系统评测报告

> 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 模型：{summary.get("model", "N/A")}

---

## 一、总体指标

| 指标 | 值 |
|------|-----|
| 总题数 | {stats["total"]} |
| 成功数 | {stats["success"]} |
| 失败数 | {stats["failed"]} |
| **成功率** | **{stats["success_rate"]}%** |
| 平均耗时 | {stats["latency"]["avg"]}ms |
| P95 耗时 | {stats["latency"]["p95"]}ms |

---

## 二、性能分布

| 百分位 | 耗时 (ms) |
|--------|-----------|
| P50 | {stats["latency"]["p50"]} |
| P95 | {stats["latency"]["p95"]} |
| P99 | {stats["latency"]["p99"]} |
| 最小 | {stats["latency"]["min"]} |
| 最大 | {stats["latency"]["max"]} |

---

## 三、错误类型分析

"""

    if stats["error_types"]:
        md += "| 错误类型 | 数量 | 占比 |\n|---------|------|------|\n"
        total_errors = sum(stats["error_types"].values())
        for error, count in stats["error_types"].items():
            pct = count / total_errors * 100
            md += f"| {error[:50]} | {count} | {pct:.1f}% |\n"
    else:
        md += "*无错误记录*\n"

    md += "\n---\n\n## 四、按类别统计\n\n"
    md += "| 类别 | 总数 | 成功 | 成功率 |\n|------|------|------|--------|\n"
    for cat, data in stats["categories"].items():
        rate = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
        md += f"| {cat} | {data['total']} | {data['success']} | {rate:.1f}% |\n"

    md += "\n---\n\n## 五、表使用统计\n\n"
    md += "| 数据表 | 使用次数 |\n|--------|----------|\n"
    for table, count in list(stats["table_usage"].items())[:10]:
        md += f"| {table} | {count} |\n"

    md += "\n---\n\n## 六、结论与建议\n\n"
    if stats["success_rate"] >= 90:
        md += "**优秀** - 系统已达到生产可用标准。\n\n建议：\n- 可继续优化少数失败案例\n- 考虑增加更多 Few-shot 案例\n"
    elif stats["success_rate"] >= 70:
        md += "**良好** - 系统基本可用，但仍需优化。\n\n建议：\n- 重点关注高频错误类型\n- 扩充相关领域的 Few-shot 案例\n"
    else:
        md += "**待改进** - 系统成功率较低，需要重点优化。\n\n建议：\n- 检查 Schema 元数据是否完整\n- 增加特定领域的训练案例\n- 考虑优化 Prompt 设计\n"

    return md


def generate_html(summary: dict[str, Any], stats: dict[str, Any]) -> str:
    """生成 HTML 格式报告"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NL2SQL 评测报告</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; font-size: 0.9em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #667eea; margin-bottom: 15px; font-size: 1em; text-transform: uppercase; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .success-rate {{ color: #10b981; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        .table th {{ background: #f8f9fa; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
        .badge-success {{ background: #d1fae5; color: #059669; }}
        .badge-danger {{ background: #fee2e2; color: #dc2626; }}
        .chart-bar {{ height: 24px; background: #e5e7eb; border-radius: 4px; overflow: hidden; margin: 8px 0; }}
        .chart-fill {{ height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px; transition: width 0.3s; }}
        .section {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #333; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        .conclusion {{ padding: 20px; border-radius: 10px; text-align: center; }}
        .conclusion.good {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }}
        .conclusion.warning {{ background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }}
        .conclusion.bad {{ background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NL2SQL 基金问答系统评测报告</h1>
            <div class="meta">
                <div>生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div>模型：{summary.get("model", "N/A")}</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>总题数</h3>
                <div class="stat-value">{stats["total"]}</div>
                <div class="stat-label">道题目</div>
            </div>
            <div class="card">
                <h3>成功数</h3>
                <div class="stat-value success-rate">{stats["success"]}</div>
                <div class="stat-label">正确回答</div>
            </div>
            <div class="card">
                <h3>失败数</h3>
                <div class="stat-value" style="color:#ef4444;">{stats["failed"]}</div>
                <div class="stat-label">执行失败</div>
            </div>
            <div class="card">
                <h3>成功率</h3>
                <div class="stat-value success-rate">{stats["success_rate"]}%</div>
                <div class="stat-label">整体准确率</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>平均耗时</h3>
                <div class="stat-value">{stats["latency"]["avg"]}</div>
                <div class="stat-label">ms / 题</div>
            </div>
            <div class="card">
                <h3>P95 耗时</h3>
                <div class="stat-value">{stats["latency"]["p95"]}</div>
                <div class="stat-label">ms</div>
            </div>
            <div class="card">
                <h3>P99 耗时</h3>
                <div class="stat-value">{stats["latency"]["p99"]}</div>
                <div class="stat-label">ms</div>
            </div>
            <div class="card">
                <h3>最大耗时</h3>
                <div class="stat-value">{stats["latency"]["max"]}</div>
                <div class="stat-label">ms</div>
            </div>
        </div>

        <div class="section">
            <h2>错误类型分析</h2>
"""
    if stats["error_types"]:
        total_errors = sum(stats["error_types"].values())
        for error, count in list(stats["error_types"].items())[:8]:
            pct = count / total_errors * 100
            html += f"""
            <div style="margin-bottom: 15px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span>{error[:60]}</span>
                    <span>{count} ({pct:.1f}%)</span>
                </div>
                <div class="chart-bar">
                    <div class="chart-fill" style="width:{pct}%;"></div>
                </div>
            </div>
"""
    else:
        html += "<p style='color:#10b981;text-align:center;'>无错误记录</p>"

    html += """
        </div>

        <div class="section">
            <h2>按类别统计</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>类别</th>
                        <th>总数</th>
                        <th>成功</th>
                        <th>成功率</th>
                    </tr>
                </thead>
                <tbody>
"""
    for cat, data in stats["categories"].items():
        rate = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
        badge_class = "badge-success" if rate >= 80 else "badge-danger"
        html += f"""
                    <tr>
                        <td>{cat}</td>
                        <td>{data['total']}</td>
                        <td>{data['success']}</td>
                        <td><span class="badge {badge_class}">{rate:.1f}%</span></td>
                    </tr>
"""
    html += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>数据表使用频率 (Top 10)</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>数据表</th>
                        <th>使用次数</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
"""
    max_usage = max(stats["table_usage"].values()) if stats["table_usage"] else 1
    for i, (table, count) in enumerate(list(stats["table_usage"].items())[:10], 1):
        pct = count / max_usage * 100
        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{table}</td>
                        <td>{count}</td>
                        <td>
                            <div class="chart-bar" style="margin:0;">
                                <div class="chart-fill" style="width:{pct}%;"></div>
                            </div>
                        </td>
                    </tr>
"""
    html += """
                </tbody>
            </table>
        </div>
"""

    # 结论
    conclusion_class = "good" if stats["success_rate"] >= 90 else "warning" if stats["success_rate"] >= 70 else "bad"
    conclusion_text = {
        "good": ("优秀", "系统已达到生产可用标准，可继续优化少数失败案例。"),
        "warning": ("良好", "系统基本可用，但仍需优化。重点关注高频错误类型。"),
        "bad": ("待改进", "系统成功率较低，需要重点优化 Schema 和 Prompt 设计。"),
    }
    title, text = conclusion_text[conclusion_class]
    html += f"""
        <div class="conclusion {conclusion_class}">
            <h2 style="border:none;margin-bottom:10px;">评测结论：{title}</h2>
            <p>{text}</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评测报告生成")
    parser.add_argument("input", nargs="?", default="eval_results_latest.json", help="评测结果文件路径")
    parser.add_argument("-o", "--output", default="eval_report", help="输出文件名（不含扩展名）")
    parser.add_argument("--format", choices=["html", "md", "all"], default="all", help="输出格式")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        print("请先运行评测: python scripts/run_eval.py")
        return

    print(f"[INFO] 加载评测结果: {input_path}")
    data = load_eval_results(str(input_path))
    summary = data.get("summary", {})
    results = data.get("results", [])

    print(f"[INFO] 分析 {len(results)} 条结果...")
    stats = analyze_results(results)

    # 生成 Markdown 报告
    if args.format in ("md", "all"):
        md_path = f"{args.output}.md"
        md_content = generate_markdown(summary, stats)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[OK] Markdown 报告: {md_path}")

    # 生成 HTML 报告
    if args.format in ("html", "all"):
        html_path = f"{args.output}.html"
        html_content = generate_html(summary, stats)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[OK] HTML 报告: {html_path}")

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"评测摘要")
    print(f"{'='*50}")
    print(f"  总题数:    {stats['total']}")
    print(f"  成功率:    {stats['success_rate']}%")
    print(f"  平均耗时:   {stats['latency']['avg']}ms")
    print(f"  P95耗时:   {stats['latency']['p95']}ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

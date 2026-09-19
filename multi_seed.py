"""多种子稳定性实验：换随机种子重训 DQN，检验结论稳不稳。

做三件事：

1. 训练：每个种子用完全相同的配置（默认 5000 局、step_cost = 0.1）训练一个 DQN 卖方，
   模型存 models/dqn_seed{N}.pt，训练曲线存 logs/multiseed/train_seed{N}.csv；
2. 评估：每个种子的模型都在同一批评估场景（seed 0~499）上跑 6 种买方。
   Tit-for-Tat / PID / PID-离散 与种子无关，只跑一次作为固定参照；
3. 出表出图：
   results/multi_seed.csv          每个 (种子, 买方, 卖方) 一行，原始数据
   results/multi_seed_summary.csv  跨种子的均值 ± 标准差 ± 95%CI + 基线对照
   figures/multi_seed.png          DQN 的跨种子误差棒 vs 三条基线
   figures/training_band.png       各种子训练曲线的 均值 ± 标准差 带

为什么需要这个：单一种子的训练曲线只能说明"这一次训得还行"，不能排除运气。
跨种子的分布才能支撑"泛化""打平"这类结论——6.4 节里"PID-离散 与 DQN 在未见对手上
打平"的说法，必须在这个实验里检查是否跨种子稳定。

用法：
    python multi_seed.py                    # 默认 5 个种子，全流程
    python multi_seed.py --seeds 0 1 2      # 指定种子
    python multi_seed.py --skip-training    # 模型已存在时跳过训练，只评估
"""
import argparse
import contextlib
import csv
import io
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存图片
import matplotlib.pyplot as plt  # noqa: E402

from metrics import summarize  # noqa: E402
from negotiation import (  # noqa: E402
    HELD_OUT_BUYERS,
    TRAINED_BUYERS,
    evaluate,
)
from pid_seller import make_pid  # noqa: E402
from plot_results import setup_font  # noqa: E402  复用同一套中文字体探测
from strategies import TitForTatSeller  # noqa: E402

HAS_CJK = setup_font()

MODEL_DIR = Path("models")
LOG_DIR = Path("logs/multiseed")
RESULTS_PATH = Path("results/multi_seed.csv")
SUMMARY_PATH = Path("results/multi_seed_summary.csv")

SEEDS = [0, 1, 2, 3, 4]
EPISODES = 5000
STEP_COST = 0.1
GAMES = 500
TIT_FOR_TAT_RESPONSIVENESS = 0.7

ALL_BUYERS = TRAINED_BUYERS + HELD_OUT_BUYERS
BASELINE_NAMES = ("Tit-for-Tat", "PID", "PID-离散")
# 95% 置信区间的 t 临界值（双侧）；种子数很少时不能用 1.96
T_CRITICAL = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def model_path(seed):
    return MODEL_DIR / f"dqn_seed{seed}.pt"


def log_path(seed):
    return LOG_DIR / f"train_seed{seed}.csv"


def _buyer_groups():
    for name, _factory in TRAINED_BUYERS:
        yield name, "训练时见过"
    for name, _factory in HELD_OUT_BUYERS:
        yield name, "没见过的对手（泛化测试）"


def _key(label):
    """卖方名 -> CSV 列名前缀（用 ASCII，方便后续脚本处理）。"""
    return {"Tit-for-Tat": "TFT", "PID-离散": "PID_discrete"}.get(label, label)


# ---------- 1. 训练 ----------
def train_seed(seed, episodes, step_cost):
    """用与主实验完全一致的配置训练一个种子，返回耗时（秒）。"""
    from train_dqn import main as train_main

    out = model_path(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = log_path(seed)
    os.makedirs(log.parent, exist_ok=True)

    start = time.time()
    # 训练脚本每 50 局打印一行，这里吞掉，由本脚本统一汇报进度
    with contextlib.redirect_stdout(io.StringIO()):
        train_main([
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--step-cost", str(step_cost),
            "--out", str(out),
            "--log", str(log),
        ])
    return time.time() - start


# ---------- 2. 评估 ----------
def make_baselines():
    """三条与种子无关的基线卖方。"""
    def tit_for_tat():
        seller = TitForTatSeller()
        seller.responsiveness = TIT_FOR_TAT_RESPONSIVENESS
        return seller

    return [
        ("Tit-for-Tat", tit_for_tat),
        ("PID", lambda: make_pid()),
        ("PID-离散", lambda: make_pid(discrete=True)),
    ]


def evaluate_dqn_model(path, games):
    """某个种子的模型在全部买方上的配对评估结果。"""
    from evaluate_dqn import load_dqn

    seller = load_dqn(path)
    results = {}
    for buyer_name, buyer_factory in ALL_BUYERS:
        results[buyer_name] = evaluate(
            lambda: seller, buyer_factory, games, seed_base=0
        )
    return results


def evaluate_baselines(games):
    results = {}
    for label, factory in make_baselines():
        results[label] = {}
        for buyer_name, buyer_factory in ALL_BUYERS:
            results[label][buyer_name] = evaluate(
                factory, buyer_factory, games, seed_base=0
            )
    return results


# ---------- 3. 汇总 ----------
def stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def ci_half_width(values):
    """跨种子的均值 95% 置信区间半宽（t 分布）。"""
    n = len(values)
    if n < 2:
        return 0.0
    critical = T_CRITICAL.get(n, 1.96)
    return critical * stdev(values) / math.sqrt(n)


def _row(seed, buyer_name, group, seller, stats):
    return {
        "seed": seed,
        "buyer": buyer_name,
        "group": group,
        "seller": seller,
        "n": stats["n"],
        "deal_rate": f"{stats['deal_rate']:.4f}",
        "deal_low": f"{stats['deal_low']:.4f}",
        "deal_high": f"{stats['deal_high']:.4f}",
        "feasible_deal_rate": f"{stats['feasible_deal_rate']:.4f}",
        "seller_profit": f"{stats['seller_profit']:.4f}",
        "buyer_profit": f"{stats['buyer_profit']:.4f}",
        "joint_profit": f"{stats['joint_profit']:.4f}",
        "surplus_share": f"{stats['surplus_share']:.4f}",
        "avg_price": f"{stats['avg_price']:.4f}",
        "avg_rounds": f"{stats['avg_rounds']:.4f}",
    }


def write_rows(rows, path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"原始结果已写入 {path}")


def build_summary_rows(per_seed, baselines):
    """每个买方一行：DQN 跨种子的均值/标准差/CI + 三条基线的点估计。"""
    rows = []
    seeds = list(per_seed)
    for buyer_name, group in _buyer_groups():
        deal = [per_seed[seed][buyer_name]["deal_rate"] for seed in seeds]
        profit = [per_seed[seed][buyer_name]["seller_profit"] for seed in seeds]
        feasible = [
            per_seed[seed][buyer_name]["feasible_deal_rate"] for seed in seeds
        ]
        deal_mean = sum(deal) / len(deal)
        profit_mean = sum(profit) / len(profit)
        deal_ci = ci_half_width(deal)
        profit_ci = ci_half_width(profit)

        row = {
            "buyer": buyer_name,
            "group": group,
            "n_seeds": len(seeds),
            "dqn_deal_mean": f"{deal_mean:.4f}",
            "dqn_deal_std": f"{stdev(deal):.4f}",
            "dqn_deal_ci95": f"{deal_ci:.4f}",
            "dqn_deal_min": f"{min(deal):.4f}",
            "dqn_deal_max": f"{max(deal):.4f}",
            "dqn_profit_mean": f"{profit_mean:.4f}",
            "dqn_profit_std": f"{stdev(profit):.4f}",
            "dqn_profit_ci95": f"{profit_ci:.4f}",
            "dqn_profit_min": f"{min(profit):.4f}",
            "dqn_profit_max": f"{max(profit):.4f}",
            "dqn_feasible_mean": f"{sum(feasible) / len(feasible):.4f}",
        }
        for label in BASELINE_NAMES:
            name = _key(label)
            stats = summarize(baselines[label][buyer_name])
            row[f"{name}_deal"] = f"{stats['deal_rate']:.4f}"
            row[f"{name}_profit"] = f"{stats['seller_profit']:.4f}"
            row[f"covers_{name}_deal"] = int(
                abs(stats["deal_rate"] - deal_mean) <= deal_ci
            )
            row[f"covers_{name}_profit"] = int(
                abs(stats["seller_profit"] - profit_mean) <= profit_ci
            )
        rows.append(row)
    return rows


def write_summary(rows, path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"跨种子汇总已写入 {path}")


def print_summary(rows):
    print("\n=== 跨种子汇总（DQN 为 均值 ± 标准差；✓ 表示该基线落在 DQN 的 95%CI 内）===")
    for row in rows:
        print(
            f"{row['buyer']:<12s} DQN       成交率 "
            f"{float(row['dqn_deal_mean']):6.1%} ± {float(row['dqn_deal_std']):4.1%}"
            f"   卖方收益 {float(row['dqn_profit_mean']):6.2f} ± "
            f"{float(row['dqn_profit_std']):4.2f}"
        )
        for label in BASELINE_NAMES:
            name = _key(label)
            mark_deal = "✓" if int(row[f"covers_{name}_deal"]) else "×"
            mark_profit = "✓" if int(row[f"covers_{name}_profit"]) else "×"
            print(
                f"{'':<12s} {label:<10s}成交率 "
                f"{float(row[f'{name}_deal']):6.1%} {mark_deal}"
                f"   卖方收益 {float(row[f'{name}_profit']):6.2f} {mark_profit}"
            )
        print()


# ---------- 4. 画图 ----------
def plot_multi_seed(rows, outdir=Path("figures")):
    buyers = [row["buyer"] for row in rows]
    positions = list(range(len(buyers)))
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6))
    marks = {"Tit-for-Tat": "s", "PID": "^", "PID-离散": "D"}
    colours = {"Tit-for-Tat": "#c46a3f", "PID": "#3f8f6a", "PID-离散": "#b5a13c"}

    for axis, key, ylabel, title in (
        (left, "deal", "成交率（%）", "成交率：DQN 跨种子 均值 ± 标准差"),
        (right, "profit", "卖方平均收益", "卖方收益：DQN 跨种子 均值 ± 标准差"),
    ):
        scale = 100 if key == "deal" else 1
        means = [float(row[f"dqn_{key}_mean"]) * scale for row in rows]
        errors = [float(row[f"dqn_{key}_std"]) * scale for row in rows]
        axis.bar(
            positions, means, 0.5, yerr=errors, capsize=4,
            color="#3b6ea5", label="DQN（5 个种子）", alpha=0.85,
        )
        for label in BASELINE_NAMES:
            name = _key(label)
            values = [float(row[f"{name}_{key}"]) * scale for row in rows]
            axis.plot(
                positions, values, linestyle="none", marker=marks[label],
                markersize=7, color=colours[label], label=label, zorder=5,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(buyers, rotation=20, ha="right", fontsize=9)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)

    figure.tight_layout()
    out = Path(outdir) / "multi_seed.png"
    os.makedirs(out.parent, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"已生成 {out}")


def read_log(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "episode": [int(row["episode"]) for row in rows],
        "reward": [float(row["reward"]) for row in rows],
        "deal_rate": [float(row["deal_rate"]) for row in rows],
    }


def plot_training_band(seeds, outdir=Path("figures")):
    logs = [read_log(log_path(seed)) for seed in seeds if log_path(seed).is_file()]
    if not logs:
        print("没找到多种子的训练日志，跳过训练曲线误差带")
        return

    episodes = logs[0]["episode"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, key, ylabel, title in (
        (left, "reward", "滑动平均收益（50 局）", "训练收益：均值 ± 标准差"),
        (right, "deal_rate", "滑动成交率（50 局）", "训练成交率：均值 ± 标准差"),
    ):
        series = [log[key] for log in logs]
        means, lower, upper = [], [], []
        for index in range(len(episodes)):
            column = [row[index] for row in series]
            mean = sum(column) / len(column)
            half = stdev(column)
            means.append(mean)
            lower.append(mean - half)
            upper.append(mean + half)
        for row in series:
            axis.plot(episodes, row, color="#3b6ea5", alpha=0.25, linewidth=1)
        axis.plot(episodes, means, color="#1f3f61", linewidth=2,
                  label=f"{len(series)} 个种子的均值")
        axis.fill_between(episodes, lower, upper, color="#3b6ea5", alpha=0.25,
                          label="± 1 个标准差")
        axis.set_xlabel("训练局数")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.tight_layout()
    out = Path(outdir) / "training_band.png"
    os.makedirs(out.parent, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    print(f"已生成 {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQN 多种子稳定性实验")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS,
                        help="随机种子列表")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="每个种子的训练局数")
    parser.add_argument("--step-cost", type=float, default=STEP_COST,
                        help="每轮时间成本，需与主实验一致")
    parser.add_argument("--games", type=int, default=GAMES,
                        help="每种买方在评估集上跑多少局")
    parser.add_argument("--skip-training", action="store_true",
                        help="跳过训练，直接用 models/dqn_seed{N}.pt 评估")
    parser.add_argument("--only-summary", action="store_true",
                        help="只按已有模型重新评估并出表出图（等价于 --skip-training）")
    args = parser.parse_args(argv)
    if args.only_summary:
        args.skip_training = True

    seeds = list(args.seeds)
    if not HAS_CJK:
        print("警告：没找到中文字体，图里的中文会显示成方框（数据不受影响）")
    print(
        f"种子：{seeds}；每个种子训练 {args.episodes} 局，step_cost={args.step_cost}；"
        f"评估：每种买方 {args.games} 局（评估集固定 seed 0~{args.games - 1}）\n"
    )

    # 1. 训练
    for seed in seeds:
        if args.skip_training and model_path(seed).is_file():
            print(f"[seed {seed}] 已有模型 {model_path(seed)}，跳过训练")
            continue
        print(f"[seed {seed}] 训练中……", end="", flush=True)
        elapsed = train_seed(seed, args.episodes, args.step_cost)
        print(f"完成，用时 {elapsed:.1f} 秒 -> {model_path(seed)}")

    # 2. 评估
    print("\n评估各基线卖方……")
    baselines = evaluate_baselines(args.games)
    per_seed = {}
    rows = []
    for seed in seeds:
        print(f"[seed {seed}] 评估 DQN……", end="", flush=True)
        per_seed[seed] = {}
        for buyer_name, results in evaluate_dqn_model(
            model_path(seed), args.games
        ).items():
            stats = summarize(results)
            per_seed[seed][buyer_name] = stats
            group = dict(_buyer_groups())[buyer_name]
            rows.append(_row(seed, buyer_name, group, "DQN", stats))
        print("完成")

    for label in BASELINE_NAMES:
        for buyer_name, results in baselines[label].items():
            stats = summarize(results)
            group = dict(_buyer_groups())[buyer_name]
            rows.append(_row("baseline", buyer_name, group, label, stats))

    write_rows(rows, RESULTS_PATH)

    # 3. 汇总与出图
    summary = build_summary_rows(per_seed, baselines)
    write_summary(summary, SUMMARY_PATH)
    print_summary(summary)
    plot_multi_seed(summary)
    plot_training_band(seeds)


if __name__ == "__main__":
    main()

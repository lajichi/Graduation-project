"""PID 卖方系数整定：先标定 P，再补 I，最后补 D（分步整定）。

为什么不用教科书式的 Ziegler-Nichols：
本实验的被控对象是**单调**的（卖方报价只降不升），不存在经典意义下的
"临界持续振荡"，对应的失稳模式是**过冲**——单轮降幅太大，一步就砸到接近底价，
成交价低、利润塌陷。所以第一步改为扫出 P 增益的可用区间，再在其中取工作点。

目标函数：
    J = 可成交场景成交率 × 卖方平均收益
    只用收益 -> 最优解退化成"永不降价"（没成交，但也没让利）；
    只用成交率 -> 最优解退化成"立刻砸到底价"。
    取乘积让两者互相约束。

整定集与评估集严格分离：
    整定（本脚本）用 seed 10000 起；
    评估（evaluate_dqn.py）用 seed 0~499。
如果两者混用，PID 就等于见过测试集，三方对比失效。

用法：
    python tune_pid.py
    python tune_pid.py --games 200 --out results/pid_coeffs.json
"""
import argparse
import csv
import json
import os
from pathlib import Path

from metrics import summarize
from negotiation import TRAINED_BUYERS, evaluate
from pid_seller import COEFF_PATH, DISCRETE_COEFF_PATH, PIDSeller

TUNE_SEED_BASE = 10000
TUNE_GAMES = 200
GRID_KP = [0.05, 0.08, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
GRID_KI = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
GRID_KD = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
TUNE_LOG = Path("results/pid_tune.csv")
DISCRETE_TUNE_LOG = Path("results/pid_tune_discrete.csv")


def score(coefficients, n_games, seed_base, discrete=False):
    """在整定集上跑全部训练对手，返回 (汇总指标, 目标函数值)。"""
    results = []
    for _name, buyer_factory in TRAINED_BUYERS:
        results.extend(
            evaluate(
                lambda: PIDSeller(discrete=discrete, **coefficients),
                buyer_factory,
                n_games,
                seed_base,
            )
        )
    stats = summarize(results)
    objective = stats["feasible_deal_rate"] * stats["seller_profit"]
    return stats, objective


def print_rows(title, rows, key_name):
    header = (
        f"{key_name:>6s}   成交率  可成交成交率   卖方收益    均价   均轮   目标函数"
    )
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        stats = row["stats"]
        print(
            f"{row['value']:6.3f}"
            f"  {stats['deal_rate']:6.1%}"
            f"  {stats['feasible_deal_rate']:10.1%}"
            f"  {stats['seller_profit']:9.2f}"
            f"  {stats['avg_price']:7.2f}"
            f"  {stats['avg_rounds']:4.1f}"
            f"  {row['objective']:8.3f}"
        )


def sweep_stage(base, value_name, values, n_games, seed_base, stage, discrete=False):
    """固定其余两项、扫一项；返回 (最优取值, 该阶段全部结果行)。"""
    rows = []
    for value in values:
        coefficients = dict(base)
        coefficients[value_name] = value
        stats, objective = score(coefficients, n_games, seed_base, discrete)
        rows.append({
            "stage": stage,
            "value": value,
            "coefficients": coefficients,
            "stats": stats,
            "objective": objective,
        })
    best = max(rows, key=lambda row: row["objective"])
    return best["value"], rows


def overshoot_start(rows, best_row):
    """过冲起点：P 增益继续增大后，利润跌破峰值 90% 的最小增益。"""
    peak_profit = best_row["stats"]["seller_profit"]
    for row in rows:
        if row["value"] <= best_row["value"]:
            continue
        if row["stats"]["seller_profit"] < 0.9 * peak_profit:
            return row["value"]
    return None


def write_csv(rows, path):
    path = Path(path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    fieldnames = [
        "stage", "kp", "ki", "kd", "deal_rate", "feasible_deal_rate",
        "seller_profit", "avg_price", "avg_rounds", "objective",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stats = row["stats"]
            coefficients = row["coefficients"]
            writer.writerow({
                "stage": row["stage"],
                "kp": f"{coefficients['kp']:.4f}",
                "ki": f"{coefficients['ki']:.4f}",
                "kd": f"{coefficients['kd']:.4f}",
                "deal_rate": f"{stats['deal_rate']:.4f}",
                "feasible_deal_rate": f"{stats['feasible_deal_rate']:.4f}",
                "seller_profit": f"{stats['seller_profit']:.4f}",
                "avg_price": f"{stats['avg_price']:.4f}",
                "avg_rounds": f"{stats['avg_rounds']:.4f}",
                "objective": f"{row['objective']:.4f}",
            })
    print(f"整定日志已写入 {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="整定 PID 卖方系数")
    parser.add_argument("--games", type=int, default=TUNE_GAMES,
                        help="整定集里每种买方跑多少局")
    parser.add_argument("--seed-base", type=int, default=TUNE_SEED_BASE,
                        help="整定集起始种子（必须与评估集错开）")
    parser.add_argument("--discrete", action="store_true",
                        help="整定离散档位版（降幅吸附到 [0,2,5,10,15]）")
    parser.add_argument("--out", default=None, help="系数输出路径")
    parser.add_argument("--log", default=None, help="整定日志 CSV 路径")
    args = parser.parse_args(argv)

    if args.out is None:
        args.out = str(DISCRETE_COEFF_PATH if args.discrete else COEFF_PATH)
    if args.log is None:
        args.log = str(DISCRETE_TUNE_LOG if args.discrete else TUNE_LOG)
    variant = "离散档位版" if args.discrete else "连续输出版"

    print(
        f"整定对象：{variant}；整定集：seed {args.seed_base} 起，"
        f"每种买方 {args.games} 局，"
        f"{len(TRAINED_BUYERS)} 种买方；"
        f"目标函数 J = 可成交场景成交率 × 卖方平均收益"
    )

    stage_rows = []
    coefficients = {"kp": 0.0, "ki": 0.0, "kd": 0.0}

    # 第一步：标定 P。k_i = k_d = 0，扫 k_p
    coefficients["kp"], rows = sweep_stage(
        coefficients, "kp", GRID_KP, args.games, args.seed_base, "P",
        args.discrete,
    )
    stage_rows.extend(rows)
    print_rows("第一步：标定比例增益 k_p（k_i = k_d = 0）", rows, "k_p")
    best_p_row = max(rows, key=lambda row: row["objective"])
    ceiling = overshoot_start(rows, best_p_row)
    print(
        f"-> 取 k_p = {coefficients['kp']:.3f}"
        f"（J = {best_p_row['objective']:.3f}）"
        + (
            f"；继续增大会过冲，利润跌破峰值 90% 的起点 k_p = {ceiling:.2f}"
            if ceiling else "；可扫描范围内未出现过冲"
        )
    )

    # 第二步：补积分项。固定最优 k_p，扫 k_i
    coefficients["ki"], rows = sweep_stage(
        coefficients, "ki", GRID_KI, args.games, args.seed_base, "PI",
        args.discrete,
    )
    stage_rows.extend(rows)
    print_rows(
        f"第二步：补积分项 k_i（k_p = {coefficients['kp']:.3f}，k_d = 0）",
        rows, "k_i",
    )
    print(f"-> 取 k_i = {coefficients['ki']:.4f}")

    # 第三步：补微分项。固定 k_p、k_i，扫 k_d
    coefficients["kd"], rows = sweep_stage(
        coefficients, "kd", GRID_KD, args.games, args.seed_base, "PID",
        args.discrete,
    )
    stage_rows.extend(rows)
    print_rows(
        f"第三步：补微分项 k_d（k_p = {coefficients['kp']:.3f}，"
        f"k_i = {coefficients['ki']:.4f}）",
        rows, "k_d",
    )
    print(f"-> 取 k_d = {coefficients['kd']:.3f}")

    final_stats, final_objective = score(
        coefficients, args.games, args.seed_base, args.discrete
    )
    print(
        f"\n整定完成：k_p = {coefficients['kp']:.3f}, "
        f"k_i = {coefficients['ki']:.4f}, k_d = {coefficients['kd']:.3f}\n"
        f"整定集表现：成交率 {final_stats['deal_rate']:.1%}，"
        f"可成交场景成交率 {final_stats['feasible_deal_rate']:.1%}，"
        f"卖方收益 {final_stats['seller_profit']:.2f}，J = {final_objective:.3f}"
    )

    payload = {
        "kp": round(coefficients["kp"], 4),
        "ki": round(coefficients["ki"], 4),
        "kd": round(coefficients["kd"], 4),
        "discrete": args.discrete,
        "objective": round(final_objective, 4),
        "tuned_on_seed_base": args.seed_base,
        "tuned_games_per_buyer": args.games,
        "tuned_buyers": [name for name, _factory in TRAINED_BUYERS],
        "objective_definition": "feasible_deal_rate * seller_profit",
    }
    out_path = Path(args.out)
    if out_path.parent:
        os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"系数已写入 {out_path}")

    if args.log:
        write_csv(stage_rows, args.log)


if __name__ == "__main__":
    main()

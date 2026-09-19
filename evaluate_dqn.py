"""评估脚本：DQN / Tit-for-Tat / PID 三个卖方在完全相同的随机场景上逐局对比。

卖方池（为什么是这四行，见 README "控制论基线" 一节）：
    DQN            学习型控制律，状态含价格历史与轮次
    Tit-for-Tat    手写纯 P 反馈（误差 = 买方加价），对照组
    PID            缺口型 P + I + D，连续输出降幅
    PID-离散       同上，但降幅吸附到 DQN 的动作档位 [0,2,5,10,15]
                   （用来排除"连续动作带来的不公平优势"）

输出：
  1. 每种买方策略下各卖方的成交率（带 95% 置信区间）、均价、双方收益、议价效率；
  2. 相对对照组 Tit-for-Tat 的配对显著性检验（成交率 McNemar / 收益配对 t），
     以及 DQN 与 PID 的直接对比；
  3. results/eval_results.csv（每个卖方一行）与 results/eval_pairwise.csv（两两配对）。

用法：
    python evaluate_dqn.py
    python evaluate_dqn.py --model models/dqn_stepcost_01.pt --games 500
"""
import argparse
import csv
import itertools
import os
import random
from pathlib import Path

import torch

from dqn_seller import DQNSeller
from metrics import (
    mcnemar_test,
    paired_ttest,
    significance_star,
    summarize,
)
from negotiation import (
    HELD_OUT_BUYERS,
    TRAINED_BUYERS,
    evaluate,
    latest_price,  # 供 plot_results.py 继续从这里导入
)
from pid_seller import DISCRETE_COEFF_PATH, load_coefficients, make_pid
from simulate import MAX_ROUNDS
from strategies import TitForTatSeller


MODEL_PATH = Path("dqn_seller.pt")
RESULT_PATH = Path("results/eval_results.csv")
PAIRWISE_PATH = Path("results/eval_pairwise.csv")
N_GAMES = 500
TIT_FOR_TAT_RESPONSIVENESS = 0.7
CONTROL = "Tit-for-Tat"
PID_COEFFS = load_coefficients()
PID_DISCRETE_COEFFS = load_coefficients(DISCRETE_COEFF_PATH)


def load_dqn(path=MODEL_PATH):
    """加载训练好的 DQN 卖家，并切换到"只推理不学习"的评估模式。"""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"未找到模型文件：{path}。请先运行 train_dqn.py 训练。"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    seller = DQNSeller(max_rounds=MAX_ROUNDS, learning=False)
    seller.q.load_state_dict(checkpoint["q"])
    seller.target_q.load_state_dict(seller.q.state_dict())
    seller.q.eval()
    seller.eps = 0.0  # 评估时不做探索
    return seller


def make_tit_for_tat():
    seller = TitForTatSeller()
    seller.responsiveness = TIT_FOR_TAT_RESPONSIVENESS
    return seller


def seller_pool(dqn_seller):
    """返回 [(卖方名, 工厂函数)]；工厂每局调用一次，手写策略就是新建实例。"""
    coefficients = dict(PID_COEFFS)
    return [
        ("DQN", lambda: dqn_seller),
        (CONTROL, make_tit_for_tat),
        ("PID", lambda: make_pid(**coefficients)),
        ("PID-离散", lambda: make_pid(discrete=True)),
    ]


def compare(results_a, results_b):
    """两组配对结果的差异与显著性检验。"""
    only_a, only_b, _chi2, deal_p = mcnemar_test(results_a, results_b)
    stats_a = summarize(results_a)
    stats_b = summarize(results_b)
    profit_diff, _t, profit_p = paired_ttest(
        [r.seller_profit for r in results_a],
        [r.seller_profit for r in results_b],
    )
    return {
        "deal_rate_diff": stats_a["deal_rate"] - stats_b["deal_rate"],
        "profit_diff": profit_diff,
        "deal_p": deal_p,
        "profit_p": profit_p,
        "only_a": only_a,
        "only_b": only_b,
    }


def print_block(buyer_name, per_seller, control_name=CONTROL):
    """打印一个买方策略下各卖方的汇总指标。"""
    print(f"=== {buyer_name}（{per_seller[control_name]['n']} 局）===")
    print("卖方            成交率[95%CI]               可成交场景成交率    均价    均轮"
          "   卖方收益  买方收益  联合收益  卖方分成")
    for label, stats in per_seller.items():
        print(
            f"{label:<14s} {stats['deal_rate']:6.1%} "
            f"[{stats['deal_low']:5.1%},{stats['deal_high']:5.1%}]"
            f"   {stats['feasible_deal_rate']:13.1%}"
            f"  {stats['avg_price']:6.2f}"
            f"  {stats['avg_rounds']:5.1f}"
            f"   {stats['seller_profit']:7.2f}"
            f"  {stats['buyer_profit']:7.2f}"
            f"  {stats['joint_profit']:7.2f}"
            f"  {stats['surplus_share']:7.2f}"
        )
    print()


def print_comparisons(rows, control_name=CONTROL):
    """rows: [(卖方 A, 卖方 B, 比较结果)]"""
    print(f"配对检验（{control_name} 为对照组）")
    for label_a, label_b, diff in rows:
        print(
            f"  {label_a:<9s} vs {label_b:<11s}"
            f"成交率 {diff['deal_rate_diff'] * 100:+5.1f} 个百分点"
            f"（仅左成交 {diff['only_a']} 局 / 仅右成交 {diff['only_b']} 局，"
            f"McNemar p={diff['deal_p']:.4f} {significance_star(diff['deal_p'])}）"
        )
        print(
            f"  {'':<9s}    {'':<11s}"
            f"卖方收益 {diff['profit_diff']:+6.2f}"
            f"（配对 t 检验 p={diff['profit_p']:.4f} "
            f"{significance_star(diff['profit_p'])}）"
        )
    print()


def write_results(rows, path):
    path = Path(path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    fieldnames = [
        "buyer", "group", "seller", "n", "deal_rate", "deal_low", "deal_high",
        "feasible_deal_rate", "avg_price", "avg_rounds", "seller_profit",
        "buyer_profit", "joint_profit", "surplus_share",
        "vs_tft_deal_p", "vs_tft_profit_diff", "vs_tft_profit_p",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果表已写入 {path}")


def write_pairwise(rows, path):
    path = Path(path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    fieldnames = [
        "buyer", "group", "seller_a", "seller_b", "deal_rate_diff", "deal_p",
        "deal_significance", "profit_diff", "profit_p", "profit_significance",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"两两配对表已写入 {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估 DQN / Tit-for-Tat / PID 卖方")
    parser.add_argument("--model", default=str(MODEL_PATH), help="DQN 模型路径")
    parser.add_argument("--games", type=int, default=N_GAMES, help="每种买方跑多少局")
    parser.add_argument("--out", default=str(RESULT_PATH), help="结果 CSV 路径")
    parser.add_argument("--pairwise", default=str(PAIRWISE_PATH),
                        help="两两配对检验 CSV 路径")
    args = parser.parse_args(argv)

    random.seed(0)
    dqn_seller = load_dqn(args.model)
    sellers = seller_pool(dqn_seller)
    print(
        f"模型：{args.model}\n"
        f"PID 系数：k_p={PID_COEFFS['kp']}, k_i={PID_COEFFS['ki']}, "
        f"k_d={PID_COEFFS['kd']}\n"
        f"PID-离散 系数：k_p={PID_DISCRETE_COEFFS['kp']}, "
        f"k_i={PID_DISCRETE_COEFFS['ki']}, k_d={PID_DISCRETE_COEFFS['kd']}"
        f"（两套系数都由 tune_pid.py 在 seed 10000+ 的整定集上得到，"
        f"与下面的评估集完全隔离）\n"
        f"每种买方 {args.games} 局；买方保留价 50~100；卖方底价 40~90；"
        f"最大 {MAX_ROUNDS} 轮；Tit-for-Tat 响应系数 {TIT_FOR_TAT_RESPONSIVENESS}\n"
    )

    result_rows = []
    pairwise_rows = []

    for group, buyers in (
        ("训练时见过", TRAINED_BUYERS),
        ("没见过的对手（泛化测试）", HELD_OUT_BUYERS),
    ):
        for buyer_name, buyer_factory in buyers:
            raw = {}
            for seller_label, factory in sellers:
                raw[seller_label] = evaluate(factory, buyer_factory, args.games)
            per_seller = {label: summarize(res) for label, res in raw.items()}

            print_block(f"{buyer_name}｜{group}", per_seller)
            comparisons = []
            for seller_label, _factory in sellers:
                if seller_label == CONTROL:
                    continue
                comparisons.append(
                    (seller_label, CONTROL, compare(raw[seller_label], raw[CONTROL]))
                )
            # 关键对比：学习型控制律 vs 人工整定的控制律
            comparisons.append(("DQN", "PID", compare(raw["DQN"], raw["PID"])))
            print_comparisons(comparisons)

            for seller_label, stats in per_seller.items():
                vs_control = compare(raw[seller_label], raw[CONTROL])
                result_rows.append({
                    "buyer": buyer_name,
                    "group": group,
                    "seller": seller_label,
                    "n": stats["n"],
                    "deal_rate": f"{stats['deal_rate']:.4f}",
                    "deal_low": f"{stats['deal_low']:.4f}",
                    "deal_high": f"{stats['deal_high']:.4f}",
                    "feasible_deal_rate": f"{stats['feasible_deal_rate']:.4f}",
                    "avg_price": f"{stats['avg_price']:.3f}",
                    "avg_rounds": f"{stats['avg_rounds']:.3f}",
                    "seller_profit": f"{stats['seller_profit']:.3f}",
                    "buyer_profit": f"{stats['buyer_profit']:.3f}",
                    "joint_profit": f"{stats['joint_profit']:.3f}",
                    "surplus_share": f"{stats['surplus_share']:.3f}",
                    "vs_tft_deal_p": f"{vs_control['deal_p']:.6f}",
                    "vs_tft_profit_diff": f"{vs_control['profit_diff']:.3f}",
                    "vs_tft_profit_p": f"{vs_control['profit_p']:.6f}",
                })

            labels = [seller_label for seller_label, _factory in sellers]
            for label_a, label_b in itertools.combinations(labels, 2):
                diff = compare(raw[label_a], raw[label_b])
                pairwise_rows.append({
                    "buyer": buyer_name,
                    "group": group,
                    "seller_a": label_a,
                    "seller_b": label_b,
                    "deal_rate_diff": f"{diff['deal_rate_diff']:.4f}",
                    "deal_p": f"{diff['deal_p']:.6f}",
                    "deal_significance": significance_star(diff["deal_p"]),
                    "profit_diff": f"{diff['profit_diff']:.3f}",
                    "profit_p": f"{diff['profit_p']:.6f}",
                    "profit_significance": significance_star(diff["profit_p"]),
                })

    write_results(result_rows, args.out)
    write_pairwise(pairwise_rows, args.pairwise)


if __name__ == "__main__":
    main()

"""评估 DQN 卖家：和 Tit-for-Tat 在完全相同的随机场景上逐局对比。

输出三件事：
  1. 每种买方策略下，两个卖方的成交率（带 95% 置信区间）、均价、双方收益、议价效率；
  2. DQN 相对 Tit-for-Tat 的配对显著性检验（成交率用 McNemar，收益用配对 t 检验）；
  3. 一张 CSV（results/eval_results.csv），给 plot_results.py 画图用。

用法：
    python evaluate_dqn.py
    python evaluate_dqn.py --model models/dqn_stepcost.pt --games 500
"""
import argparse
import csv
import os
import random
from pathlib import Path

import torch

from dqn_seller import DQNSeller
from metrics import (
    Result,
    mcnemar_test,
    paired_ttest,
    significance_star,
    summarize,
)
from simulate import (
    BUYER_TOP,
    MAX_ROUNDS,
    START_OFFER,
    sample_scenario,
    set_seed,
)
from strategies import (
    ConcederBuyer,
    GrandetBuyer,
    MirrorBuyer,
    RandomBuyer,
    ReactiveBuyer,
    TimeDependentBuyer,
    TitForTatSeller,
)


MODEL_PATH = Path("dqn_seller.pt")
RESULT_PATH = Path("results/eval_results.csv")
N_GAMES = 500
TIT_FOR_TAT_RESPONSIVENESS = 0.7


# ---- 对手池：前 5 个是训练时见过的，最后 1 个是没见过的（泛化测试）----
TRAINED_BUYERS = [
    ("让步型买方", lambda r: ConcederBuyer(
        BUYER_TOP, start_offer=START_OFFER, step=6, reservation_price=r)),
    ("时间依赖型买方", lambda r: TimeDependentBuyer(
        BUYER_TOP, MAX_ROUNDS, start_offer=START_OFFER, reservation_price=r)),
    ("固定报价买方", lambda r: GrandetBuyer(
        BUYER_TOP, start_offer=START_OFFER, reservation_price=r)),
    ("反应型买方", lambda r: ReactiveBuyer(
        BUYER_TOP, start_offer=START_OFFER, match_rate=0.7,
        min_seller_drop=2.0, reservation_price=r)),
    ("随机加价买方", lambda r: RandomBuyer(
        BUYER_TOP, start_offer=START_OFFER, max_step=10, reservation_price=r)),
]

HELD_OUT_BUYERS = [
    ("以牙还牙型买方", lambda r: MirrorBuyer(
        BUYER_TOP, start_offer=START_OFFER, match_rate=0.6,
        min_step=1.5, reservation_price=r)),
]


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


def latest_price(seller):
    """两个卖方类的"当前报价"字段名不一样，这里统一取一下。"""
    if isinstance(seller, DQNSeller):
        return seller.prev_price
    return seller.prev_seller_price


def play_game(seller, buyer, seller_start, seller_bottom):
    """跑一局，返回 metrics.Result。买家先出价，卖方后还价。"""
    feasible = buyer.reservation_price >= seller_bottom
    seller.reset(seller_start, MAX_ROUNDS)
    buyer.reset()

    for round_index in range(1, MAX_ROUNDS + 1):
        buyer_action, _buyer_value = buyer.decide(latest_price(seller))
        if buyer_action == "accept":
            # 买家接受了卖方的报价，成交价就是卖方当前的报价
            price = latest_price(seller)
            if isinstance(seller, DQNSeller):
                seller.finalize_terminal(price)
            return Result.build(
                True, price, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )

        seller_action, seller_value = seller.decide(_buyer_value, seller_bottom)
        if seller_action == "accept":
            # 卖方接受了买方的出价，成交价就是买方的出价
            return Result.build(
                True, seller_value, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )

    if isinstance(seller, DQNSeller):
        seller.finalize_crash()
    return Result.build(
        False, 0.0, MAX_ROUNDS, feasible, seller_bottom, buyer.reservation_price
    )


def evaluate(seller_factory, buyer_factory, n_games=N_GAMES):
    """跑 n_games 局。每局先固定种子再抽场景，两个卖方保证面对同样的问题。"""
    results = []
    for seed in range(n_games):
        set_seed(seed)
        seller_start, seller_bottom, reservation_price = sample_scenario()
        buyer = buyer_factory(reservation_price)
        results.append(
            play_game(seller_factory(), buyer, seller_start, seller_bottom)
        )
    return results


def print_block(buyer_name, dqn_results, tft_results):
    """打印一个买方策略下的对比结果。"""
    dqn_stats = summarize(dqn_results)
    tft_stats = summarize(tft_results)

    print(f"=== {buyer_name}（{len(dqn_results)} 局）===")
    print("卖方            成交率[95%CI]              可成交场景成交率   均价   均轮"
          "  卖方收益  买方收益  联合收益  卖方分成")
    for label, stats in (("DQN 卖方", dqn_stats), ("Tit-for-Tat", tft_stats)):
        print(
            f"{label:<12s}  {stats['deal_rate']:6.1%} "
            f"[{stats['deal_low']:5.1%},{stats['deal_high']:5.1%}]"
            f"   {stats['feasible_deal_rate']:8.1%}"
            f"   {stats['avg_price']:5.2f}  {stats['avg_rounds']:4.1f}"
            f"   {stats['seller_profit']:7.2f}  {stats['buyer_profit']:7.2f}"
            f"   {stats['joint_profit']:7.2f}  {stats['surplus_share']:7.2f}"
        )

    only_dqn, only_tft, _chi2, deal_p = mcnemar_test(dqn_results, tft_results)
    profit_diff, _t, profit_p = paired_ttest(
        [r.seller_profit for r in dqn_results],
        [r.seller_profit for r in tft_results],
    )
    print(
        f"对比：成交率差 {(dqn_stats['deal_rate'] - tft_stats['deal_rate']) * 100:+.1f} 个百分点"
        f"（仅 DQN 成交 {only_dqn} 局 / 仅 Tit-for-Tat 成交 {only_tft} 局，"
        f"McNemar p={deal_p:.4f} {significance_star(deal_p)}）"
    )
    print(
        f"      卖方收益差 {profit_diff:+.2f}"
        f"（配对 t 检验 p={profit_p:.4f} {significance_star(profit_p)}）"
    )
    print()
    return dqn_stats, tft_stats, deal_p, profit_p


def write_csv(rows, path=RESULT_PATH):
    path = Path(path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    fieldnames = [
        "buyer", "group", "seller", "n", "deal_rate", "deal_low", "deal_high",
        "feasible_deal_rate", "avg_price", "avg_rounds", "seller_profit",
        "buyer_profit", "joint_profit", "surplus_share", "deal_p",
        "profit_diff", "profit_p",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果表已写入 {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估 DQN 卖家")
    parser.add_argument("--model", default=str(MODEL_PATH), help="模型路径")
    parser.add_argument("--games", type=int, default=N_GAMES, help="每种买方跑多少局")
    parser.add_argument("--out", default=str(RESULT_PATH), help="结果 CSV 路径")
    args = parser.parse_args(argv)

    random.seed(0)
    dqn_seller = load_dqn(args.model)
    print(
        f"模型：{args.model}\n"
        f"每种买方 {args.games} 局；买方保留价 50~100；卖方底价 40~90；"
        f"最大 {MAX_ROUNDS} 轮；Tit-for-Tat 响应系数 {TIT_FOR_TAT_RESPONSIVENESS}\n"
    )

    rows = []
    for group, buyers in (
        ("训练时见过", TRAINED_BUYERS),
        ("没见过的对手（泛化测试）", HELD_OUT_BUYERS),
    ):
        for buyer_name, buyer_factory in buyers:
            dqn_results = evaluate(lambda: dqn_seller, buyer_factory, args.games)
            tft_results = evaluate(make_tit_for_tat, buyer_factory, args.games)
            dqn_stats, tft_stats, deal_p, profit_p = print_block(
                f"{buyer_name}｜{group}", dqn_results, tft_results
            )

            profit_diff, _t, _p = paired_ttest(
                [r.seller_profit for r in dqn_results],
                [r.seller_profit for r in tft_results],
            )
            for seller_name, stats in (
                ("DQN", dqn_stats),
                ("TitForTat", tft_stats),
            ):
                rows.append({
                    "buyer": buyer_name,
                    "group": group,
                    "seller": seller_name,
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
                    "deal_p": f"{deal_p:.6f}",
                    "profit_diff": f"{profit_diff:.3f}",
                    "profit_p": f"{profit_p:.6f}",
                })

    write_csv(rows, args.out)


if __name__ == "__main__":
    main()

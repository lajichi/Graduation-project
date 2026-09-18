"""谈判小游戏：你来当买家，和 AI 卖家砍价（答辩现场演示用）。

用法：
    python play.py                 # 对手是手写 Tit-for-Tat 卖家
    python play.py --seller dqn    # 对手是训练好的 DQN 卖家

规则：你的预算上限 100，最多 10 轮，谈不成就崩（双方收益都是 0）。
输入数字出价；也可以直接输入"接受"，接受卖方的当前报价。
"""
import argparse
import random

from simulate import BUYER_TOP, MAX_ROUNDS, sample_scenario
from strategies import TitForTatSeller


def latest_price(seller):
    """卖方当前的报价：两个卖方类字段名不一样，这里统一取。"""
    if hasattr(seller, "prev_price"):
        return seller.prev_price
    return seller.prev_seller_price


def build_seller(kind, model_path):
    if kind == "dqn":
        from evaluate_dqn import load_dqn  # 放在函数里，玩 TFT 时不用加载 torch

        return load_dqn(model_path), "DQN 卖家（训练了 5000 局）"
    return TitForTatSeller(), "Tit-for-Tat 卖家"


def play(seller, seller_name, seller_start, seller_bottom):
    seller.reset(seller_start, MAX_ROUNDS)

    print("=" * 46)
    print("  谈判开始！你是买家")
    print(f"  你的预算上限：{BUYER_TOP}（这是你愿意出的最高价）")
    print(f"  最多 {MAX_ROUNDS} 轮，谈不成就崩")
    print(f"  卖方底线多少？不知道，只知道他开价 {latest_price(seller)}")
    print(f"  卖方策略：{seller_name}")
    print("=" * 46)

    for round_index in range(1, MAX_ROUNDS + 1):
        print(f"\n--- 第 {round_index} 轮 ---")
        print(f"卖方报价：{latest_price(seller)}")
        try:
            raw = input("你出价（数字，或输入 接受）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出游戏。")
            return

        if raw in {"接受", "accept", "a"}:
            price = latest_price(seller)
            print(f"\n你接受了卖方的报价，成交价 {price}")
            print(f"你的收益：{BUYER_TOP - price:.1f}")
            return

        try:
            buyer_offer = float(raw)
        except ValueError:
            print("没看懂，这轮不算，重新出价")
            continue

        if not 0 <= buyer_offer <= BUYER_TOP:
            print(f"出价要在 0~{BUYER_TOP} 之间，这轮不算")
            continue

        action, price = seller.decide(buyer_offer, seller_bottom)
        if action == "accept":
            print(f"\n卖方接受了！成交价 {price}")
            print(f"你的收益：{BUYER_TOP - price:.1f}（预算 {BUYER_TOP} - 成交价）")
            return
        print(f"卖方还价：{price}")

    print(f"\n{MAX_ROUNDS} 轮谈完，没成交。双方都没收益。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="和 AI 卖家谈判")
    parser.add_argument("--seller", choices=["tft", "dqn"], default="tft",
                        help="对手类型：tft=手写策略，dqn=训练好的模型")
    parser.add_argument("--model", default="dqn_seller.pt", help="DQN 模型路径")
    args = parser.parse_args(argv)

    random.seed()  # 每局都不一样
    seller_start, seller_bottom, _ = sample_scenario()
    seller, seller_name = build_seller(args.seller, args.model)
    play(seller, seller_name, seller_start, seller_bottom)


if __name__ == "__main__":
    main()

"""批量模拟：手写买方策略 vs 手写 Tit-for-Tat 卖方，作为论文的基线实验。

用法：
    python simulate.py
"""
import random

from metrics import Result, format_summary, summarize
from strategies import (
    ConcederBuyer,
    GrandetBuyer,
    RandomBuyer,
    ReactiveBuyer,
    TimeDependentBuyer,
    TitForTatSeller,
)


# ---- 统一实验设置：训练、评估、消融实验都从这里取，保证数字可比 ----
BUYER_TOP = 100
START_OFFER = 40
RESERVATION_RANGE = (50, 100)
SELLER_BOTTOM_RANGE = (40, 90)
SELLER_START_MIN = 60
SELLER_START_MAX = 100
MAX_ROUNDS = 10
N_GAMES = 500


def set_seed(seed):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)


def sample_reservation_price():
    """买方保留价：他最多愿意出多少钱。"""
    return round(random.uniform(*RESERVATION_RANGE), 1)


def sample_scenario():
    """随机抽一个谈判场景：(卖方起始报价, 卖方底价, 买方保留价)。"""
    seller_bottom = random.randint(*SELLER_BOTTOM_RANGE)
    seller_start = random.randint(max(SELLER_START_MIN, seller_bottom), SELLER_START_MAX)
    reservation_price = sample_reservation_price()
    return seller_start, seller_bottom, reservation_price


def one_game(buyer, seller_start, seller_bottom, max_rounds, responsiveness):
    """跑一局（买家先出价，卖方再还价），返回 metrics.Result。"""
    seller = TitForTatSeller()
    seller.responsiveness = responsiveness  # 覆盖默认值，reset 不会动它
    seller.reset(seller_start, max_rounds)

    feasible = buyer.reservation_price >= seller_bottom
    for round_index in range(1, max_rounds + 1):
        action, value = buyer.decide(seller.prev_seller_price)
        if action == "accept":
            return Result.build(
                True, value, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )
        action, value = seller.decide(value, seller_bottom)
        if action == "accept":
            return Result.build(
                True, value, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )
    return Result.build(
        False, 0.0, max_rounds, feasible, seller_bottom, buyer.reservation_price
    )


def run_games(buyer_cls, buyer_kwargs, n_games, responsiveness=0.7, seed_offset=0):
    """同一种买方跑 n 局；每局用固定种子重新抽场景，结果可复现。"""
    results = []
    for seed in range(n_games):
        random.seed(seed + seed_offset)
        seller_start, seller_bottom, reservation_price = sample_scenario()
        buyer = buyer_cls(
            BUYER_TOP,
            start_offer=START_OFFER,
            reservation_price=reservation_price,
            **buyer_kwargs,
        )
        results.append(
            one_game(buyer, seller_start, seller_bottom, MAX_ROUNDS, responsiveness)
        )
    return results


# 对手池：名字, 类, 构造参数
BUYERS = [
    ("ConcederBuyer", ConcederBuyer, {"step": 6}),
    ("TimeDependentBuyer", TimeDependentBuyer, {"max_rounds": MAX_ROUNDS}),
    ("GrandetBuyer", GrandetBuyer, {}),
    ("ReactiveBuyer", ReactiveBuyer, {"match_rate": 0.7, "min_seller_drop": 2.0}),
    ("RandomBuyer", RandomBuyer, {"max_step": 10}),
]


if __name__ == "__main__":
    print(
        f"每种买方策略 {N_GAMES} 局（每局固定种子，结果可复现）\n"
        f"买方保留价 {RESERVATION_RANGE}；卖方底价 {SELLER_BOTTOM_RANGE}；"
        f"最大 {MAX_ROUNDS} 轮；卖方为 Tit-for-Tat（响应系数 0.7）\n"
    )
    for name, buyer_cls, buyer_kwargs in BUYERS:
        results = run_games(buyer_cls, buyer_kwargs, N_GAMES)
        print(format_summary(name, summarize(results)))

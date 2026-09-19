"""谈判对局循环与对手池：买方先出价、卖方后还价，跑满 10 轮没成交就算崩。

单独抽成模块的理由：整定 PID、评估 DQN、画图都要跑对局，
如果各写一份循环，口径迟早会漂移。这里保持唯一实现，供所有脚本共用。
本模块刻意不依赖 torch，方便只做控制论基线的人不装深度学习栈也能跑。

用法：
    from negotiation import TRAINED_BUYERS, evaluate
    results = evaluate(make_pid, TRAINED_BUYERS[0][1], n_games=500)
"""
from __future__ import annotations

from metrics import Result
from simulate import (
    BUYER_TOP,
    MAX_ROUNDS,
    N_GAMES,
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
)


def latest_price(seller):
    """卖方当前报价：几种卖方类的字段名不一样，这里统一取。"""
    if hasattr(seller, "prev_price"):
        return seller.prev_price
    return seller.prev_seller_price


def play_game(seller, buyer, seller_start, seller_bottom, max_rounds=MAX_ROUNDS):
    """跑一局，返回 metrics.Result。

    接受规则对所有卖方完全一致：买方出价 >= 卖方新报价就成交，
    成交价取买方那一方的出价（与 TitForTat 的口径相同）。
    这样 DQN / TFT / PID 的差异只来自"让步律"，不来自成交判据。
    """
    feasible = buyer.reservation_price >= seller_bottom
    seller.reset(seller_start, max_rounds)
    buyer.reset()

    for round_index in range(1, max_rounds + 1):
        buyer_action, buyer_value = buyer.decide(latest_price(seller))
        if buyer_action == "accept":
            # 买方接受卖方上轮报价，成交价就是卖方当前报价
            price = latest_price(seller)
            # 学习型卖方要收尾这一步转移（手写策略没这个方法）
            if hasattr(seller, "finalize_terminal"):
                seller.finalize_terminal(price)
            return Result.build(
                True, price, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )

        seller_action, seller_value = seller.decide(buyer_value, seller_bottom)
        if seller_action == "accept":
            # 卖方接受买方出价，成交价就是买方出价
            return Result.build(
                True, seller_value, round_index, feasible, seller_bottom,
                buyer.reservation_price,
            )

    if hasattr(seller, "finalize_crash"):
        seller.finalize_crash()
    return Result.build(
        False, 0.0, max_rounds, feasible, seller_bottom, buyer.reservation_price
    )


# ---- 对手池 ----
# 前 5 个是训练时见过的，MirrorBuyer 只用于泛化测试，任何卖方都不许训练它。
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


def evaluate(seller_factory, buyer_factory, n_games=N_GAMES, seed_base=0):
    """跑 n_games 局。

    每局先用固定种子抽场景，所以不同卖方面对的是同一批场景和同一个保留价，
    可以做配对检验（McNemar / 配对 t）。

    seed_base 是"整定集 / 评估集"分离的抓手：整定用 10000 起，评估用 0 起，
    两批场景完全独立，PID 的系数不可能是在测试集上调出来的。
    """
    results = []
    for seed in range(n_games):
        set_seed(seed + seed_base)
        seller_start, seller_bottom, reservation_price = sample_scenario()
        buyer = buyer_factory(reservation_price)
        results.append(play_game(seller_factory(), buyer, seller_start, seller_bottom))
    return results

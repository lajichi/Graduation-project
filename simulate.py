"""批量模拟：每种买家策略 vs TitForTat 卖家，跑 N 局后统计成交率/均价/均轮数

用法：python simulate.py
"""
import random
from strategies import (TitForTatSeller, ConcederBuyer, TimeDependentBuyer,
                         GrandetBuyer, RandomBuyer)

BUYER_TOP = 100       # 买家预算上限
MAX_ROUNDS = 15      # 每局最多轮数
N_GAMES = 500         # 每种策略跑多少局


def one_game(buyer, seller_start, seller_bottom, max_rounds, responsiveness):
    """跑一局，返回 (是否成交, 成交价, 轮数)"""
    seller = TitForTatSeller()
    seller.responsiveness = responsiveness   # 覆盖默认 0.5（reset 不会动它）
    seller.reset(seller_start, max_rounds)
    for r in range(max_rounds):
        act, val = buyer.decide(seller.prev_seller_price)   # 买家先动
        if act == "accept":
            return True, val, r + 1
        act, val = seller.decide(val, seller_bottom)         # 卖家回应
        if act == "accept":
            return True, val, r + 1
    return False, None, max_rounds


def run_games(buyer_cls, buyer_kwargs, n_games, responsiveness=0.7):
    """同一种买家跑 n 局，每局重新随机卖家场景"""
    results = []
    for i in range(n_games):
        random.seed(i)                       # 每局一个种子，结果可复现
        seller_start = random.randint(60, 90)
        seller_bottom = random.randint(1, 50)
        buyer = buyer_cls(BUYER_TOP, **buyer_kwargs)
        deal, price, rounds = one_game(buyer, seller_start, seller_bottom, MAX_ROUNDS, responsiveness)
        results.append((deal, price, rounds))
    return results


def summarize(name, results):
    """打印一行的统计"""
    n = len(results)
    deals = [r for r in results if r[0]]
    deal_rate = len(deals) / n
    avg_price = sum(r[1] for r in deals) / len(deals) if deals else 0.0
    avg_rounds = sum(r[2] for r in deals) / len(deals) if deals else 0.0
    print(f"{name:20s} 成交率 {deal_rate:6.1%}  均价 {avg_price:6.2f}  均轮 {avg_rounds:5.1f}  崩局 {n - len(deals)}")


# 对手池：名字, 类, 构造参数
BUYERS = [
    ("ConcederBuyer",      ConcederBuyer,      {}),
    ("TimeDependentBuyer", TimeDependentBuyer, {"max_rounds": MAX_ROUNDS}),
    ("GrandetBuyer",       GrandetBuyer,       {}),
    ("RandomBuyer",        RandomBuyer,        {}),
]


if __name__ == "__main__":
    print(f"每种策略 {N_GAMES} 局，买家预算 {BUYER_TOP}，最多 {MAX_ROUNDS} 轮\n")
    for name, cls, kw in BUYERS:
        results = run_games(cls, kw, N_GAMES)
        summarize(name, results)
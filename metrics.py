"""统一的评价指标与统计检验。

论文里所有表格都用这里的口径，避免不同脚本各算一套数字：
  - 成交率 / 可成交场景成交率（带 Wilson 95% 置信区间）
  - 卖方收益、买方收益、联合收益
  - 议价效率（卖方在共同增益里的分成比例）
  - DQN 与 Tit-for-Tat 的配对显著性检验（McNemar + 配对 t 检验）

用法:
    from metrics import Result, summarize, mcnemar_test
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

try:  # scipy 只用于配对 t 检验，环境里没有也能跑
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None


class Result(NamedTuple):
    """一局谈判的结果。未成交时 price 与收益都记 0.0。"""

    deal: bool  # 是否成交
    price: float  # 成交价（未成交 0.0）
    rounds: int  # 实际用了多少轮
    feasible: bool  # 场景本身有没有共同增益区间（买方保留价 >= 卖方底价）
    seller_profit: float  # 成交价 - 卖方底价（未成交 0.0）
    buyer_profit: float  # 买方保留价 - 成交价（未成交 0.0）
    surplus_share: float  # 卖方分成 = 卖方收益 / 共同增益，不能成交时 nan

    @staticmethod
    def build(deal, price, rounds, feasible, seller_bottom, reservation_price):
        """把一局的原始信息换算成 Result。"""
        if deal:
            seller_profit = price - seller_bottom
            buyer_profit = reservation_price - price
        else:
            seller_profit = 0.0
            buyer_profit = 0.0

        joint_surplus = reservation_price - seller_bottom
        if deal and feasible and joint_surplus > 0:
            surplus_share = seller_profit / joint_surplus
        else:
            surplus_share = float("nan")

        return Result(
            deal=bool(deal),
            price=float(price) if deal else 0.0,
            rounds=int(rounds),
            feasible=bool(feasible),
            seller_profit=seller_profit,
            buyer_profit=buyer_profit,
            surplus_share=surplus_share,
        )


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Wilson 区间：小样本/接近 0% 或 100% 时也比正态近似稳。"""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def mean_interval(values: Sequence[float], z: float = 1.96):
    """均值的正态近似置信区间，返回 (均值, 半宽)。"""
    values = list(values)
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = z * math.sqrt(variance / len(values))
    return mean, half_width


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(results: Sequence[Result]):
    """把一批对局汇总成论文表格需要的指标。"""
    results = list(results)
    n = len(results)
    deals = [r for r in results if r.deal]
    feasible = [r for r in results if r.feasible]
    feasible_deals = [r for r in feasible if r.deal]

    deal_low, deal_high = wilson_interval(len(deals), n)
    feasible_deal_low, feasible_deal_high = wilson_interval(
        len(feasible_deals), len(feasible)
    )

    seller_profit, seller_profit_half = mean_interval(
        [r.seller_profit for r in results]
    )
    buyer_profit, _ = mean_interval([r.buyer_profit for r in results])
    joint_profit = seller_profit + buyer_profit
    surplus_shares = [r.surplus_share for r in results if r.surplus_share == r.surplus_share]

    return {
        "n": n,
        "deal_rate": len(deals) / n if n else 0.0,
        "deal_low": deal_low,
        "deal_high": deal_high,
        "feasible_rate": len(feasible) / n if n else 0.0,
        "feasible_deal_rate": len(feasible_deals) / len(feasible) if feasible else 0.0,
        "feasible_deal_low": feasible_deal_low,
        "feasible_deal_high": feasible_deal_high,
        "stalemates": n - len(deals),
        "avg_price": _mean([r.price for r in deals]),
        "avg_rounds": _mean([r.rounds for r in deals]),
        "seller_profit": seller_profit,
        "seller_profit_half": seller_profit_half,
        "buyer_profit": buyer_profit,
        "joint_profit": joint_profit,
        "surplus_share": _mean(surplus_shares),
    }


def mcnemar_test(results_a: Sequence[Result], results_b: Sequence[Result]):
    """配对检验成交与否。a=新方法(DQN)，b=对照(Tit-for-Tat)。

    返回 (b_only, c_only, 统计量, p 值)：b_only 是"只有 A 成交"的场景数。
    """
    if len(results_a) != len(results_b):
        raise ValueError("McNemar 检验要求两组结果一一配对，长度必须相同")
    only_a = sum(1 for x, y in zip(results_a, results_b) if x.deal and not y.deal)
    only_b = sum(1 for x, y in zip(results_a, results_b) if y.deal and not x.deal)
    if only_a + only_b == 0:
        return only_a, only_b, 0.0, 1.0
    statistic = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    # 自由度 1 的卡方分布：p = erfc(sqrt(chi2 / 2))
    p_value = math.erfc(math.sqrt(statistic / 2))
    return only_a, only_b, statistic, p_value


def paired_ttest(values_a: Sequence[float], values_b: Sequence[float]):
    """配对 t 检验（用于比较每组卖方收益）。返回 (均值差, t, p)。"""
    if len(values_a) != len(values_b):
        raise ValueError("配对 t 检验要求两组数据一一配对")
    differences = [a - b for a, b in zip(values_a, values_b)]
    mean_diff = _mean(differences)
    if len(differences) < 2:
        return mean_diff, 0.0, 1.0
    variance = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
    if variance == 0:
        return mean_diff, 0.0, 1.0
    t_statistic = mean_diff / math.sqrt(variance / len(differences))
    if _scipy_stats is not None:
        p_value = float(2 * _scipy_stats.t.sf(abs(t_statistic), len(differences) - 1))
    else:  # 没有 scipy 时退化成正态近似
        p_value = math.erfc(abs(t_statistic) / math.sqrt(2))
    return mean_diff, t_statistic, p_value


def significance_star(p_value: float) -> str:
    """论文表格里的显著性标记。"""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def format_summary(name: str, stats: dict, width: int = 16) -> str:
    """一行式的汇总，给 simulate.py 这种快速对比用。"""
    return (
        f"{name:<{width}s} 成交率 {stats['deal_rate']:6.1%}"
        f" [{stats['deal_low']:5.1%},{stats['deal_high']:5.1%}]"
        f"  可成交场景成交率 {stats['feasible_deal_rate']:6.1%}"
        f"  均价 {stats['avg_price']:6.2f}"
        f"  均轮 {stats['avg_rounds']:5.1f}"
        f"  卖方收益 {stats['seller_profit']:5.2f}"
        f"  僵局 {stats['stalemates']}"
    )

"""Sensitivity analysis for Tit-for-Tat seller responsiveness."""
from simulate import BUYERS, N_GAMES, run_games


RESP_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


def collect():
    """Return overall rate, feasible-only rate, and mean deal price."""
    table = {}
    for name, buyer_cls, buyer_kwargs in BUYERS:
        table[name] = {}
        for responsiveness in RESP_VALUES:
            results = run_games(
                buyer_cls, buyer_kwargs, N_GAMES, responsiveness=responsiveness
            )
            deals = [result for result in results if result[0]]
            feasible = [result for result in results if result[3]]
            feasible_deals = [result for result in feasible if result[0]]
            deal_rate = len(deals) / len(results)
            feasible_deal_rate = (
                len(feasible_deals) / len(feasible) if feasible else 0.0
            )
            avg_price = (
                sum(result[1] for result in deals) / len(deals) if deals else 0.0
            )
            table[name][responsiveness] = (deal_rate, feasible_deal_rate, avg_price)
    return table


def print_table(title, metric_index, table):
    header = f"{'买方策略':20s} " + " ".join(
        f"{value:>6.2f}" for value in RESP_VALUES
    )
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for name, metrics in table.items():
        cells = [metrics[value][metric_index] for value in RESP_VALUES]
        if metric_index in (0, 1):
            row = " ".join(f"{cell * 100:5.0f}%" for cell in cells)
        else:
            row = " ".join(f"{cell:6.2f}" for cell in cells)
        print(f"{name:20s} {row}")


if __name__ == "__main__":
    print(
        f"每种组合 {N_GAMES} 局；扫描响应系数 "
        f"{RESP_VALUES[0]}~{RESP_VALUES[-1]}"
    )
    table = collect()
    print_table("总体成交率", 0, table)
    print_table("可成交场景成交率", 1, table)
    print_table("平均成交价", 2, table)

"""参数扫描：扫 TitForTatSeller.responsiveness 0.2~1.0

输出两张表：成交率、平均成交价。每行一个买家，每列一个 responsiveness 值。
用法：python sweep.py
"""
from simulate import run_games, BUYERS, N_GAMES

RESP_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


def collect():
    """跑完所有 (买家 × responsiveness) 组合，返回嵌套 dict"""
    table = {}  # {buyer_name: {responsiveness: (deal_rate, avg_price)}}
    for name, cls, kw in BUYERS:
        table[name] = {}
        for r in RESP_VALUES:
            results = run_games(cls, kw, N_GAMES, responsiveness=r)
            deals = [x for x in results if x[0]]
            deal_rate = len(deals) / N_GAMES
            avg_price = sum(x[1] for x in deals) / len(deals) if deals else 0.0
            table[name][r] = (deal_rate, avg_price)
    return table


def print_table(title, key_idx, table):
    """key_idx=0 打印成交率，=1 打印均价"""
    header = f"{'买家':20s} " + " ".join(f"{r:>6.2f}" for r in RESP_VALUES)
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for name in table:
        cells = [table[name][r][key_idx] for r in RESP_VALUES]
        if key_idx == 0:
            row = " ".join(f"{c*100:5.0f}%" for c in cells)
        else:
            row = " ".join(f"{c:6.2f}" for c in cells)
        print(f"{name:20s} {row}")


if __name__ == "__main__":
    print(f"每种组合 {N_GAMES} 局，扫 responsiveness {RESP_VALUES[0]}~{RESP_VALUES[-1]}")
    table = collect()
    print_table("成交率", 0, table)
    print_table("平均成交价", 1, table)

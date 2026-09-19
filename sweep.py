"""敏感性分析：手写基线的响应系数，以及 PID 三个系数各自的敏感度。

两种模式：
    python sweep.py                       # Tit-for-Tat 响应系数 0.2~1.0 扫描
    python sweep.py --pid kp               # 固定 k_i / k_d，单独扫 k_p
    python sweep.py --pid ki --discrete    # 离散档位版的 k_i 敏感度

PID 的敏感度跑在**整定集**（seed 10000 起）上：那是"开发数据"，
评估集（seed 0~499）只用来出最终对比表，不参与任何系数选择。
"""
import argparse

from metrics import summarize
from negotiation import (
    HELD_OUT_BUYERS,
    TRAINED_BUYERS,
    evaluate as evaluate_sellers,
)
from pid_seller import (
    COEFF_PATH,
    DISCRETE_COEFF_PATH,
    PIDSeller,
    load_coefficients,
)
from simulate import BUYERS, N_GAMES, run_games

RESP_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
PID_SEED_BASE = 10000
PID_VALUES = {
    "kp": [0.05, 0.08, 0.12, 0.2, 0.3, 0.5],
    "ki": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
    "kd": [0.0, 0.1, 0.3, 0.5, 0.8, 1.2],
}
ALL_BUYERS = TRAINED_BUYERS + HELD_OUT_BUYERS


# ---- 模式一：Tit-for-Tat 响应系数 ----
def collect_tft():
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


def print_tft_table(title, metric_index, table):
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


# ---- 模式二：PID 单个系数的敏感度 ----
def collect_pid(coefficient, values, games, seed_base, discrete):
    """固定其余两项扫一项，返回 {取值: {买方: 汇总指标}}。"""
    path = DISCRETE_COEFF_PATH if discrete else COEFF_PATH
    base = load_coefficients(path)
    table = {}
    for value in values:
        coefficients = dict(base)
        coefficients[coefficient] = value
        table[value] = {}
        for buyer_name, factory in ALL_BUYERS:
            results = evaluate_sellers(
                lambda: PIDSeller(discrete=discrete, **coefficients),
                factory,
                games,
                seed_base,
            )
            table[value][buyer_name] = summarize(results)
    return table


def print_pid_table(title, metric, table):
    values = list(table)
    header = f"{'买方策略':20s} " + " ".join(f"{value:>7.3f}" for value in values)
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for buyer_name, _factory in ALL_BUYERS:
        cells = [table[value][buyer_name][metric] for value in values]
        if metric in ("deal_rate", "feasible_deal_rate"):
            row = " ".join(f"{cell * 100:6.0f}%" for cell in cells)
        else:
            row = " ".join(f"{cell:7.2f}" for cell in cells)
        print(f"{buyer_name:20s} {row}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="敏感性分析")
    parser.add_argument("--pid", choices=sorted(PID_VALUES), default=None,
                        help="扫 PID 的哪个系数；不填就扫 Tit-for-Tat 响应系数")
    parser.add_argument("--values", nargs="*", type=float, default=None,
                        help="自定义扫描取值")
    parser.add_argument("--games", type=int, default=N_GAMES,
                        help="PID 敏感度里每种买方跑多少局")
    parser.add_argument("--seed-base", type=int, default=PID_SEED_BASE,
                        help="PID 敏感度用的起始种子（默认整定集）")
    parser.add_argument("--discrete", action="store_true",
                        help="扫离散档位版的系数")
    args = parser.parse_args(argv)

    if args.pid is None:
        print(
            f"每种组合 {N_GAMES} 局；扫描 Tit-for-Tat 响应系数 "
            f"{RESP_VALUES[0]}~{RESP_VALUES[-1]}"
        )
        table = collect_tft()
        print_tft_table("总体成交率", 0, table)
        print_tft_table("可成交场景成交率", 1, table)
        print_tft_table("平均成交价", 2, table)
        return

    values = args.values or PID_VALUES[args.pid]
    variant = "离散档位版" if args.discrete else "连续输出版"
    path = DISCRETE_COEFF_PATH if args.discrete else COEFF_PATH
    base = load_coefficients(path)
    print(
        f"{variant} PID：扫 {args.pid}，其余系数固定为 "
        + ", ".join(f"{name}={value}" for name, value in base.items() if name != args.pid)
        + f"；每种买方 {args.games} 局，整定集 seed {args.seed_base} 起"
    )
    table = collect_pid(args.pid, values, args.games, args.seed_base, args.discrete)
    print_pid_table(f"成交率随 {args.pid} 变化", "deal_rate", table)
    print_pid_table(f"可成交场景成交率随 {args.pid} 变化", "feasible_deal_rate", table)
    print_pid_table(f"卖方收益随 {args.pid} 变化", "seller_profit", table)


if __name__ == "__main__":
    main()

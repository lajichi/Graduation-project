"""把训练曲线和评估结果画成论文用的图。

用法：
    python plot_results.py                       # 自动找 logs/train_*.csv
    python plot_results.py --logs logs/train_stepcost_0.csv logs/train_stepcost_01.csv

输出（默认在 figures/ 目录）：
    training_curve.png     训练过程：滑动平均收益 & 成交率 & epsilon
    deal_rate.png          各买方策略下的成交率（DQN vs Tit-for-Tat，带 95% 置信区间）
    seller_profit.png      各买方策略下的卖方平均收益
    concession_example.png 一局示例的还价轨迹对比
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存图片
import matplotlib.pyplot as plt  # noqa: E402

from evaluate_dqn import (  # noqa: E402
    load_dqn,
    make_tit_for_tat,
    latest_price,
)
from simulate import (  # noqa: E402
    BUYER_TOP,
    MAX_ROUNDS,
    START_OFFER,
    set_seed,
)
from strategies import ConcederBuyer  # noqa: E402


DEFAULT_FIGDIR = Path("figures")
RESULTS_PATH = Path("results/eval_results.csv")

# 常见中文字体，找到哪个用哪个；都没有就退回英文标签
CJK_FONTS = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Source Han Sans SC", "PingFang SC", "Arial Unicode MS",
]


def setup_font():
    """返回当前环境能不能正常显示中文。"""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_CJK = setup_font()


def L(chinese, english):
    """有中文字体就用中文标签，否则用英文。"""
    return chinese if HAS_CJK else english


# ---------- 1. 训练曲线 ----------
def read_log(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "episode": [int(row["episode"]) for row in rows],
        "reward": [float(row["reward"]) for row in rows],
        "deal_rate": [float(row["deal_rate"]) for row in rows],
        "epsilon": [float(row["epsilon"]) for row in rows],
    }


def plot_training_curves(log_paths, outdir):
    if not log_paths:
        print("没有找到训练日志（logs/train_*.csv），跳过训练曲线")
        return None

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for path in log_paths:
        data = read_log(path)
        label = Path(path).stem
        ax1.plot(data["episode"], data["reward"], label=label, linewidth=1.5)
        ax2.plot(data["episode"], data["deal_rate"], label=label, linewidth=1.5)

    ax1.set_xlabel(L("训练局数", "Episodes"))
    ax1.set_ylabel(L("滑动平均收益（50 局）", "Rolling mean reward (50 eps)"))
    ax1.set_title(L("训练过程中的收益", "Reward during training"))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(L("训练局数", "Episodes"))
    ax2.set_ylabel(L("滑动成交率（50 局）", "Rolling deal rate (50 eps)"))
    ax2.set_title(L("训练过程中的成交率", "Deal rate during training"))
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    figure.tight_layout()
    out = Path(outdir) / "training_curve.png"
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


# ---------- 2 / 3. 各买方策略下的柱状对比 ----------
def read_results(path=RESULTS_PATH):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _grouped_bars(rows, outdir, key, low_key, high_key, ylabel, title, filename,
                  percent=False):
    buyers = []
    for row in rows:
        if row["buyer"] not in buyers:
            buyers.append(row["buyer"])

    dqn = {row["buyer"]: row for row in rows if row["seller"] == "DQN"}
    tft = {row["buyer"]: row for row in rows if row["seller"] == "TitForTat"}

    positions = range(len(buyers))
    width = 0.38
    scale = 100 if percent else 1

    figure, axis = plt.subplots(figsize=(1.6 * len(buyers) + 3, 4.2))
    for offset, table, label, colour in (
        (-width / 2, dqn, "DQN", "#3b6ea5"),
        (width / 2, tft, "Tit-for-Tat", "#c46a3f"),
    ):
        values = [float(table[buyer][key]) * scale for buyer in buyers]
        errors = None
        if low_key:
            errors = [
                [
                    (float(table[buyer][key]) - float(table[buyer][low_key])) * scale
                    for buyer in buyers
                ],
                [
                    (float(table[buyer][high_key]) - float(table[buyer][key])) * scale
                    for buyer in buyers
                ],
            ]
        axis.bar(
            [p + offset for p in positions], values, width,
            label=label, color=colour, yerr=errors, capsize=3,
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(buyers, rotation=20, ha="right", fontsize=9)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    out = Path(outdir) / filename
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


# ---------- 4. 一局示例的还价轨迹 ----------
def record_game(seller, buyer, seller_start, seller_bottom):
    """记录一局里买方的出价和卖方的报价轨迹。"""
    seller.reset(seller_start, MAX_ROUNDS)
    buyer.reset()
    buyer_points = []
    seller_points = [(0, seller_start)]

    for round_index in range(1, MAX_ROUNDS + 1):
        buyer_action, buyer_value = buyer.decide(latest_price(seller))
        buyer_points.append((round_index, buyer_value))
        if buyer_action == "accept":
            return buyer_points, seller_points, latest_price(seller)
        seller_action, seller_value = seller.decide(buyer_value, seller_bottom)
        seller_points.append((round_index, seller_value))
        if seller_action == "accept":
            return buyer_points, seller_points, seller_value
    return buyer_points, seller_points, None


def plot_concession_example(outdir, model_path, seller_bottom=50.0,
                            seller_start=85.0, reservation_price=80.0):
    """固定一个场景，画 DQN 与 Tit-for-Tat 的还价轨迹差异。"""
    set_seed(7)
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    dealt = {}

    for label, factory, colour in (
        ("DQN", lambda: load_dqn(model_path), "#3b6ea5"),
        ("Tit-for-Tat", make_tit_for_tat, "#c46a3f"),
    ):
        set_seed(7)  # 两边看到完全相同的对手行为
        buyer = ConcederBuyer(
            BUYER_TOP, start_offer=START_OFFER, step=6,
            reservation_price=reservation_price,
        )
        seller = factory()
        buyer_points, seller_points, deal_price = record_game(
            seller, buyer, seller_start, seller_bottom
        )
        dealt[label] = deal_price
        axis.step(
            [r for r, _ in buyer_points], [p for _, p in buyer_points],
            where="post", linestyle="--", color="grey", linewidth=1,
            label=L("买方报价", "Buyer offer") if label == "DQN" else None,
        )
        axis.step(
            [r for r, _ in seller_points], [p for _, p in seller_points],
            where="post", color=colour, linewidth=2, label=label,
        )

    axis.axhline(seller_bottom, color="red", linestyle=":", linewidth=1)
    axis.text(0.1, seller_bottom + 1, L("卖方底价", "Seller bottom"),
              color="red", fontsize=8)
    axis.axhline(reservation_price, color="green", linestyle=":", linewidth=1)
    axis.text(0.1, reservation_price + 1, L("买方保留价", "Buyer reservation"),
              color="green", fontsize=8)

    axis.set_xlabel(L("轮次", "Round"))
    axis.set_ylabel(L("价格", "Price"))
    axis.set_title(L(
        f"一局示例的还价轨迹（买方={reservation_price:.0f}，底价={seller_bottom:.0f}）",
        f"Concession paths (buyer={reservation_price:.0f}, bottom={seller_bottom:.0f})",
    ))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    out = Path(outdir) / "concession_example.png"
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out, dealt


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成论文用的图")
    parser.add_argument("--logs", nargs="*", default=None,
                        help="训练日志 CSV；不填就自动找 logs/train_*.csv")
    parser.add_argument("--results", default=str(RESULTS_PATH),
                        help="评估结果 CSV")
    parser.add_argument("--model", default="dqn_seller.pt", help="DQN 模型路径")
    parser.add_argument("--outdir", default=str(DEFAULT_FIGDIR), help="图片输出目录")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    log_paths = args.logs or sorted(glob.glob("logs/train_*.csv"))
    print(f"中文字体：{'可用' if HAS_CJK else '不可用，图中改用英文标签'}")

    outputs = [plot_training_curves(log_paths, args.outdir)]
    if Path(args.results).is_file():
        rows = read_results(args.results)
        outputs.append(_grouped_bars(
            rows, args.outdir, "deal_rate", "deal_low", "deal_high",
            L("成交率（%）", "Deal rate (%)"),
            L("各买方策略下的成交率（误差棒为 95% 置信区间）",
              "Deal rate by buyer strategy (95% CI)"),
            "deal_rate.png", percent=True,
        ))
        outputs.append(_grouped_bars(
            rows, args.outdir, "seller_profit", None, None,
            L("卖方平均收益", "Mean seller profit"),
            L("各买方策略下的卖方平均收益", "Seller profit by buyer strategy"),
            "seller_profit.png",
        ))
    else:
        print(f"没找到 {args.results}，跳过评估图（先运行 evaluate_dqn.py）")

    if Path(args.model).is_file():
        example, dealt = plot_concession_example(args.outdir, args.model)
        print(f"示例成交价：{dealt}")
        outputs.append(example)

    for path in outputs:
        if path:
            print(f"已生成 {path}")


if __name__ == "__main__":
    main()

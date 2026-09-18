"""训练 DQN 卖家：对手池轮换，记录训练曲线并保存模型。

用法：
    python train_dqn.py                                   # 默认配置
    python train_dqn.py --episodes 5000 --seed 0          # 指定局数和随机种子
    python train_dqn.py --step-cost 0.1 --out models/dqn_stepcost.pt
"""
import argparse
import csv
import os
import random

import torch

from dqn_seller import DQNSeller
from simulate import (
    BUYER_TOP,
    MAX_ROUNDS,
    START_OFFER,
    sample_reservation_price,
    sample_scenario,
    set_seed,
)
from strategies import (
    ConcederBuyer,
    GrandetBuyer,
    RandomBuyer,
    ReactiveBuyer,
    TimeDependentBuyer,
)


N_EPISODES = 5000
SAVE_PATH = "dqn_seller.pt"
LOG_PATH = "logs/train_log.csv"
LOG_EVERY = 50
ROLLING_WINDOW = 50


def make_buyers():
    """训练对手池，和 simulate.py 里的基线买方保持一致。"""
    return [
        lambda: ConcederBuyer(
            BUYER_TOP, start_offer=START_OFFER, step=6,
            reservation_price=sample_reservation_price(),
        ),
        lambda: TimeDependentBuyer(
            BUYER_TOP, MAX_ROUNDS, start_offer=START_OFFER,
            reservation_price=sample_reservation_price(),
        ),
        lambda: GrandetBuyer(
            BUYER_TOP, start_offer=START_OFFER,
            reservation_price=sample_reservation_price(),
        ),
        lambda: ReactiveBuyer(
            BUYER_TOP, start_offer=START_OFFER, match_rate=0.7,
            min_seller_drop=2.0, reservation_price=sample_reservation_price(),
        ),
        lambda: RandomBuyer(
            BUYER_TOP, start_offer=START_OFFER, max_step=10,
            reservation_price=sample_reservation_price(),
        ),
    ]


def run_episode(seller, buyer, seller_start, seller_bottom):
    """跑一局，返回 (卖方总收益, 是否成交, 成交价, 轮数)。

    总收益直接累加 DQNSeller 每一步记下的 last_reward，
    这样打印出来的曲线和真正进 replay buffer 的奖励是同一个口径。
    """
    seller.reset(seller_start, MAX_ROUNDS)
    buyer.reset()
    total_reward = 0.0

    for round_index in range(1, MAX_ROUNDS + 1):
        # 买家先动
        buyer_action, buyer_value = buyer.decide(seller.prev_price)
        if buyer_action == "accept":
            # 买家直接接受卖家上轮报价，本局结束
            seller.finalize_terminal(seller.prev_price)
            total_reward += seller.last_reward
            return total_reward, True, seller.prev_price, round_index

        # 卖家回应
        seller_action, seller_value = seller.decide(buyer_value, seller_bottom)
        total_reward += seller.last_reward
        if seller_action == "accept":
            return total_reward, True, seller_value, round_index

    seller.finalize_crash()
    total_reward += seller.last_reward
    return total_reward, False, None, MAX_ROUNDS


def main(argv=None):
    parser = argparse.ArgumentParser(description="训练 DQN 卖家")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="训练局数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子，保证训练可复现")
    parser.add_argument("--step-cost", type=float, default=0.0,
                        help="每轮的时间成本，>0 时卖家会更愿意早点成交")
    parser.add_argument("--out", default=SAVE_PATH, help="模型保存路径")
    parser.add_argument("--log", default=LOG_PATH, help="训练曲线 CSV 路径")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    seller = DQNSeller(max_rounds=MAX_ROUNDS, step_cost=args.step_cost)
    buyers = make_buyers()

    rewards = []
    deal_flags = []
    log_rows = []

    for episode in range(1, args.episodes + 1):
        # 每局重新随机抽卖家场景，和 simulate.py 的范围一致，保证公平对比
        seller_start, seller_bottom, _ = sample_scenario()
        buyer = random.choice(buyers)()
        reward, deal, _price, _rounds = run_episode(
            seller, buyer, seller_start, seller_bottom
        )
        rewards.append(reward)
        deal_flags.append(1.0 if deal else 0.0)

        if episode % LOG_EVERY == 0 or episode == args.episodes:
            recent_rewards = rewards[-ROLLING_WINDOW:]
            recent_deals = deal_flags[-ROLLING_WINDOW:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            deal_rate = sum(recent_deals) / len(recent_deals)
            log_rows.append({
                "episode": episode,
                "reward": f"{avg_reward:.4f}",
                "deal_rate": f"{deal_rate:.4f}",
                "epsilon": f"{seller.eps:.4f}",
            })
            print(
                f"ep {episode:5d}  eps={seller.eps:.3f}  "
                f"近{ROLLING_WINDOW}局 均reward={avg_reward:6.2f}  "
                f"成交率={deal_rate:5.1%}  buffer={len(seller.buffer)}"
            )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "q": seller.q.state_dict(),
        "eps": seller.eps,
        "step_done": seller.step_done,
        "step_cost": args.step_cost,
        "episodes": args.episodes,
        "seed": args.seed,
    }, args.out)

    if args.log:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["episode", "reward", "deal_rate", "epsilon"]
            )
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"训练曲线已写入 {args.log}")

    print(f"\n训练完成，模型存到 {args.out}")


if __name__ == "__main__":
    main()

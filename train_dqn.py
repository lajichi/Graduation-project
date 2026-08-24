"""训练 DQN 卖家：对手池轮换，跑 N 局，打印 reward 曲线，存模型

用法：python train_dqn.py
"""
import random

from strategies import ConcederBuyer, TimeDependentBuyer, GrandetBuyer, RandomBuyer
from dqn_seller import DQNSeller

BUYER_TOP = 95
MAX_ROUNDS = 10
N_EPISODES = 5000         # 训练局数，gamma 提到 0.99 后多训一些
SAVE_PATH = "dqn_seller.pt"

# 对手池（和 simulate.py 一致，但买家是 instance，每局 reset）
def make_buyers():
    return [
        ConcederBuyer(BUYER_TOP),
        TimeDependentBuyer(BUYER_TOP, MAX_ROUNDS),
        GrandetBuyer(BUYER_TOP),
        RandomBuyer(BUYER_TOP),
    ]


def run_episode(seller, buyer, seller_start, seller_bottom):
    """跑一局，返回 (seller总reward, 是否成交, 成交价, 轮数)"""
    seller.reset(seller_start, MAX_ROUNDS)
    buyer.reset()
    total_reward = 0.0

    for r in range(MAX_ROUNDS):
        # 买家先动
        b_act, b_val = buyer.decide(seller.prev_price)
        if b_act == "accept":
            # 买家直接接受卖家上轮报价
            seller.finalize_terminal(seller.prev_price)
            total_reward += seller.prev_price - seller.bottom
            return total_reward, True, seller.prev_price, r + 1

        # 卖家动
        s_act, s_val = seller.decide(b_val, seller_bottom)
        if s_act == "accept":
            total_reward += s_val - seller.bottom
            return total_reward, True, s_val, r + 1

        total_reward += -0.1  # 中间轮惩罚（也已进 buffer）

    seller.finalize_crash()
    return total_reward - 1.0, False, None, MAX_ROUNDS  # 谈崩大惩罚


def main():
    seller = DQNSeller(max_rounds=MAX_ROUNDS)
    buyers = make_buyers()

    rewards = []            # 每局 reward
    deal_rates = []         # 滑动成交率
    log_every = 50

    for ep in range(1, N_EPISODES + 1):
        # 每局随机卖家场景（范围和 simulate.py 一致，公平对比）
        seller_bottom = random.randint(1, 50)
        seller_start = random.randint(60, 90)
        buyer = random.choice(buyers)

        reward, deal, price, rounds = run_episode(seller, buyer, seller_start, seller_bottom)
        rewards.append(reward)
        deal_rates.append(1.0 if deal else 0.0)

        if ep % log_every == 0:
            recent_r = rewards[-log_every:]
            recent_d = deal_rates[-log_every:]
            avg_r = sum(recent_r) / len(recent_r)
            avg_d = sum(recent_d) / len(recent_d)
            print(f"ep {ep:4d}  eps={seller.eps:.3f}  "
                  f"近{log_every}局 均reward={avg_r:6.2f}  成交率={avg_d:5.1%}  "
                  f"buffer={len(seller.buffer)}")

    # 存模型
    import torch
    torch.save({
        "q": seller.q.state_dict(),
        "eps": seller.eps,
        "step_done": seller.step_done,
    }, SAVE_PATH)
    print(f"\n训练完成，模型存到 {SAVE_PATH}")


if __name__ == "__main__":
    main()

"""DQN 卖家：用 RL 学让步策略，替代手写 TitForTatSeller

四件套：Q 网络（MLP）、epsilon-greedy、replay buffer、target network。
接口和 TitForTatSeller 对齐：reset(seller_start, max_rounds)、decide(buyer_offer, seller_bottom)
→ 返回 ("accept"/"offer", price)。训练时训练循环额外调 store/finalize 填 transition。
"""
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Q 网络 ----------
class QNet(nn.Module):
    """输入 state 向量(6维)，输出每个动作的 Q 值(5维)"""
    def __init__(self, state_dim=6, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s):
        return self.net(s)


# ---------- 2. Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch):
        s, a, r, s_next, done = zip(*random.sample(self.buf, batch))

        # 终止步没有"下一状态"（push 时存的 None），用零向量占位
        clean = []
        for x in s_next:
            if x is None:
                clean.append(torch.zeros(6))
            else:
                clean.append(x)

        return (
            torch.stack(s),  # (batch, 6)
            torch.tensor(a, dtype=torch.long).unsqueeze(1),  # (batch, 1)  给 gather 用
            torch.tensor(r, dtype=torch.float),  # (batch,)
            torch.stack(clean),  # (batch, 6)
            torch.tensor(done, dtype=torch.float),  # (batch,)
        )

    def __len__(self):
        return len(self.buf)


# ---------- 3. DQN Seller ----------
# 动作档位：降幅 0/2/5/10/15
ACTIONS = [0, 2, 5, 10, 15]


class DQNSeller:
    def __init__(self, seller_bottom=55, max_rounds=10,
                 lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=500,
                 batch=32, buffer_size=10000, target_update=100,
                 step_cost=0.0, learning=True):
        self.bottom = seller_bottom
        self.max_rounds = max_rounds

        self.gamma = gamma
        self.batch = batch
        self.target_update = target_update
        # 每轮的时间成本：>0 时卖家"拖轮次"要付出代价，会更愿意早点成交
        self.step_cost = step_cost
        # False = 只推理不学习（评估模式），避免再靠 batch 大小去做 hack
        self.learning = learning
        self.last_reward = 0.0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.eps = eps_start
        self.step_done = 0

        self.q = QNet()
        self.target_q = QNet()
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()  # target 网络只推理，不训
        self.opt = torch.optim.Adam(self.q.parameters(), lr)

        self.buffer = ReplayBuffer(buffer_size)

        # 局内状态
        self.prev_price = None
        self.prev_buyer_offer = None
        self.extalk = 0
        self.rounds = 0
        self.pending = None  # (s, a, r)：上一非终止步，等下一状态的 s' 来收尾

    # ---- 与 TitForTatSeller 对齐的接口 ----
    def reset(self, seller_start, all_round):
        self.max_rounds = all_round
        self.prev_price = seller_start
        self.prev_buyer_offer = None
        self.extalk = 0
        self.rounds = 0
        self.pending = None
        self.last_reward = 0.0

    def _state(self, buyer_offer):
        if self.prev_buyer_offer is not None:
            delta = (buyer_offer - self.prev_buyer_offer)
        else :
            delta = 0.0
        return torch.tensor([
            buyer_offer / 100.0,
            delta / 20.0,
            self.prev_price / 100.0,
            self.bottom / 100.0,
            (self.max_rounds - self.rounds) / self.max_rounds,
            self.extalk / self.max_rounds,
        ], dtype=torch.float)

    def _select_action(self, s):
        if random.random() < self.eps:
            return random.randrange(len(ACTIONS))
        with torch.no_grad():
            return self.q(s).argmax().item()

    # ---- 训练用的主入口（也兼 eval，eval 时 eps 设 0）----
    def decide(self, buyer_offer, seller_bottom):
        self.bottom = seller_bottom
        # Preserve this value before episode state is updated. It is used for
        # the non-terminal shaping reward below.
        buyer_delta = (
            buyer_offer - self.prev_buyer_offer
            if self.prev_buyer_offer is not None
            else 0.0
        )
        # 买家重复出价计数（Grandet 会触发，是"买家在死扛"的信号）
        if self.prev_buyer_offer is not None and abs(buyer_delta) < 0.05:
            self.extalk += 1
        s = self._state(buyer_offer)

        # 收尾上一非终止步：现在拿到的 buyer_offer 让我们能算 s' 了
        if self.pending is not None:
            ps, pa, pr = self.pending
            self.buffer.push(ps, pa, pr, s, False)
            self._learn()
            self.pending = None

        a = self._select_action(s)
        drop = ACTIONS[a]
        new_price = max(self.bottom, round(self.prev_price - drop, 1))
        accept = buyer_offer >= new_price

        # 更新局内状态（供下次 _state 用）
        self.prev_buyer_offer = buyer_offer
        self.prev_price = new_price
        self.rounds += 1

        if accept:
            # A buyer offer is the transaction price throughout the simulator.
            # This matches TitForTatSeller's payoff and return-value semantics.
            reward = buyer_offer - self.bottom
            self.last_reward = reward
            self.buffer.push(s, a, reward, torch.zeros(6), True)  # 终止，s_next 占位
            self._learn()
        else:
            # shaping：买家这轮加价了多少，按系数给正奖励，引导"等买家加价"
            # Grandet 重复出价 delta=0，shaping=0，不会误导它去磨
            # step_cost：每多磨一轮扣一点，避免一直拖着等买家让步
            reward = 0.3 * buyer_delta - self.step_cost
            self.last_reward = reward
            self.pending = (s, a, reward)  # 等下一轮 s' 收尾

        return "accept" if accept else "offer", buyer_offer if accept else new_price

    # ---- 买家先一步接受时，卖家的上一动作算终止并结算 ----
    def finalize_terminal(self, deal_price):
        """买家直接接受卖家上轮报价，episode 在此终止"""
        if self.pending is None:
            self.last_reward = 0.0  # 没有待结算的转移，避免重复计奖励
            return
        ps, pa, _pr = self.pending
        reward = deal_price - self.bottom
        self.last_reward = reward
        self.buffer.push(ps, pa, reward, torch.zeros(6), True)
        self._learn()
        self.pending = None

    def finalize_crash(self):
        """max_rounds 走完仍没成交"""
        if self.pending is None:
            self.last_reward = -1.0
            return
        ps, pa, pr = self.pending
        self.last_reward = -1.0
        self.buffer.push(ps, pa, -1.0, torch.zeros(6), True)  # 谈崩大惩罚
        self._learn()
        self.pending = None

    # ---- 学习一步 ----
    def _learn(self):
        if not self.learning:
            return
        if len(self.buffer) < self.batch:
            return
        s, a, r, s_next, done = self.buffer.sample(self.batch)
        q_sa = self.q(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target_q(s_next).max(1).values
            target = r + self.gamma * q_next * (1 - done)
        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        # 梯度裁剪，防爆炸
        for p in self.q.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1, 1)
        self.opt.step()

        self.step_done += 1
        if self.step_done % self.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        # epsilon 指数衰减
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
                   math.exp(-self.step_done / self.eps_decay)

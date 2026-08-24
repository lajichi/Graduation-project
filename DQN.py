import random
import torch
import torch.nn as nn
from collections import deque


class QNet (nn.Module):
    def __init__(self,state_dim=6, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),nn.ReLU(),
            nn.Linear(hidden, hidden),nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, state):
        return self.net(state)


class ReplayBuffer :
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch):
        s, a, r, s_next, done = zip(*random.sample(self.buf, batch))

        clean = []
        for x in s_next:
            if x is None:
                clean.append(torch.zeros(6))
            else:
                clean.append(x)

        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long).unsqueeze(1),
            torch.tensor(r, dtype=torch.float),
            torch.stack(clean),
            torch.tensor(done, dtype=torch.float),
        )
    def __len__(self):
        return len(self.buf)


ACTIONS = [0, 2, 5, 10, 15]

class DQNSeller:
    def __init__(self,seller_bottom=55, max_rounds=10,gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=500,
                 batch=32, buffer_size=10000, target_update=100):
        self.gamma = gamma
        self.batch = batch
        self.target_update = target_update
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.eps = eps_start
        self.step_done = 0

        self.q = QNet()
        self.target_q = QNet()
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()  # target 网络只推理，不训
        self.opt = torch.optim.Adam(self.q.parameters(), lr)

        self.buffer = ReplayBuffer(buffer_size)

        self.prev_price = None
        self.prev_buyer_offer = None
        self.extalk = 0
        self.round = 0
        self.pending = None
        self.bottom = seller_bottom
        self.max_rounds = max_rounds
        self.pending = None


    def reset(self, seller_start, all_round):
        self.max_rounds = all_round
        self.prev_price = seller_start
        self.prev_buyer_offer = None
        self.extalk = 0
        self.round = 0
        self.pending = None

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
            (self.max_rounds - self.round) / self.max_rounds,
            self.extalk / self.max_rounds,
        ], dtype=torch.float)


    def _select_action(self, s):
        if random.random() < self.eps:
            return random.randrange(len(ACTIONS))
        with torch.no_grad():
            return self.q(s).argmax().item()
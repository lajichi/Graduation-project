"""PID 卖方：把"怎么让步"写成一个反馈控制律。

与手写基线的关系：

- Conceder / TimeDependent 买方是**开环时间控制**：只看轮次，不看对手；
- TitForTatSeller 是**纯比例（P）控制**，但它的误差取"买方本轮加价 Δb"，
  属于互惠型误差——买家的加价一旦归零，控制器就失去了输入信号。
  这正是 simulate.py 里"反应型买方把 TFT 拖死"的成因：不是某条规则写错，
  而是纯 P 控制在稳态误差下失去调节能力；
- 本文件把误差换成**缺口型** e = 卖方报价 - 买方报价，并补上积分、微分两项：
    P 项：缺口越大，让步越快；
    I 项：累积未闭合的缺口，专门修"对手不加价 -> 僵局"的稳态误差（带抗积分饱和）；
    D 项：缺口的闭合趋势（即买家的加价速度），买家让得快，卖方就少让。

控制量是本轮降幅，上界与 DQN 的动作档位一致；discrete=True 时把降幅吸附到同一组
档位 [0, 2, 5, 10, 15]，用于排除"连续动作带来的不公平优势"。

接口与 TitForTatSeller 完全一致：
    reset(seller_start, all_round)
    decide(buyer_offer, seller_bottom) -> ("accept" / "offer", price)

用法：
    from pid_seller import load_coefficients, make_pid
    seller = make_pid()
"""
from __future__ import annotations

import json
from pathlib import Path

# 与 dqn_seller.ACTIONS 保持一致；单独写一份是为了让本模块不依赖 torch
DISCRETE_DROPS = (0.0, 2.0, 5.0, 10.0, 15.0)
DISCRETE_POSITIVE = DISCRETE_DROPS[1:]

# 设计初值由量纲推导得到，不是随机起点（推导过程见 README "控制论基线" 一节）：
#   e 的典型量级 0~40，单轮降幅希望落在 3~6 元            -> kp ≈ 0.15
#   积分项在第 10 轮的贡献约 2~3 元（此时 Σe ≈ 200）        -> ki ≈ 0.01
#   D 项贡献 1~2 元，Δb 典型约 5                            -> kd ≈ 0.3
DESIGN_DEFAULTS = {"kp": 0.15, "ki": 0.01, "kd": 0.3}
COEFF_PATH = Path("results/pid_coeffs.json")
# 离散档位版会被吸附到 [0,2,5,10,15]，有效增益和连续版不一样，
# 所以单独整定一套系数（tune_pid.py --discrete），两边都取各自最优点再比。
DISCRETE_COEFF_PATH = Path("results/pid_coeffs_discrete.json")


def _clip(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


class PIDSeller:
    """缺口型 PID 卖方。误差 e = 卖方当前报价 - 买方本轮报价。"""

    def __init__(
        self,
        kp=DESIGN_DEFAULTS["kp"],
        ki=DESIGN_DEFAULTS["ki"],
        kd=DESIGN_DEFAULTS["kd"],
        d_max=15.0,
        i_max=None,
        i_clip=5.0,
        deadband=0.2,
        discrete=False,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.d_max = float(d_max)
        # 积分限幅：把积分项的"贡献"限制在 i_clip 元以内，避免它压过利润项。
        # 折算成积分本身的上下界就是 i_clip / ki。
        self.i_clip = float(i_clip)
        self.i_max = (
            float(i_max) if i_max is not None
            else (self.i_clip / self.ki if self.ki > 0 else 0.0)
        )
        self.deadband = float(deadband)
        self.discrete = bool(discrete)

        # 局内状态
        self.prev_price = None
        self.prev_buyer_offer = None
        self.e_prev = None
        self.integral = 0.0
        self.rounds = 0
        self.last_drop = 0.0

    # ---- 与 TitForTatSeller / DQNSeller 对齐的接口 ----
    def reset(self, seller_start, all_round):
        self.prev_price = seller_start
        self.prev_buyer_offer = None
        self.e_prev = None
        self.integral = 0.0
        self.rounds = 0
        self.last_drop = 0.0

    @property
    def prev_seller_price(self):
        """字段别名：老的仿真循环按这个名字读卖方的当前报价。"""
        return self.prev_price

    def _control(self, error, delta_error):
        drop = self.kp * error + self.ki * self.integral + self.kd * delta_error
        # 截断到 [0, d_max]：卖方不主动涨价（"顶回去"这类行为交给 TFT 表现），
        # 所以 D 项为负时最多少降，而不是反向加价。
        drop = _clip(drop, 0.0, self.d_max)
        if self.discrete:
            # 离散化要小心一个陷阱：如果按"就近取整"到含 0 的档位表，
            # 需要的降幅 0.6 会被吸附成 0，报价原地不动，控制器就退化成
            # TFT 那种"稳态误差"了。所以只要需求超出死区，就至少走最小档位。
            if drop <= self.deadband:
                drop = 0.0
            else:
                drop = min(
                    DISCRETE_POSITIVE,
                    key=lambda value: (abs(value - drop), value),
                )
        return drop

    def decide(self, buyer_offer, seller_bottom):
        error = self.prev_price - buyer_offer
        if abs(error) < self.deadband:
            error = 0.0
        delta_error = 0.0 if self.e_prev is None else error - self.e_prev

        # 抗积分饱和：已经贴着底价时不再累积积分（否则饱和后回不来），
        # 也不在"缺口已在闭合"的负方向上无限累积。
        if self.prev_price > seller_bottom:
            self.integral = _clip(self.integral + error, -self.i_max, self.i_max)

        old_price = self.prev_price
        drop = self._control(error, delta_error)
        new_price = max(seller_bottom, round(old_price - drop, 1))

        self.e_prev = error
        self.prev_buyer_offer = buyer_offer
        self.prev_price = new_price
        self.rounds += 1
        self.last_drop = round(old_price - new_price, 1)  # 实际降幅（含底价截断）

        if buyer_offer >= new_price:
            return "accept", buyer_offer
        return "offer", new_price

    def coefficients(self):
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}


def load_coefficients(path=COEFF_PATH):
    """读取 tune_pid.py 整定出的系数；没有结果文件时退回设计初值。"""
    path = Path(path)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return {name: float(data[name]) for name in ("kp", "ki", "kd")}
    return dict(DESIGN_DEFAULTS)


def make_pid(discrete=False, path=None, **overrides):
    """按整定出的系数造一个 PID 卖方，overrides 可临时覆盖某一项。

    离散档位版默认读自己的整定文件；没有该文件时退回连续版系数。
    """
    if path is None:
        path = DISCRETE_COEFF_PATH if discrete else COEFF_PATH
        if discrete and not Path(path).is_file():
            path = COEFF_PATH
    coefficients = load_coefficients(path)
    coefficients.update(overrides)
    return PIDSeller(discrete=discrete, **coefficients)

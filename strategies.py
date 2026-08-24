import random


class TitForTatSeller:
    def __init__(self):
        self.responsiveness = 0.5  # 响应系数：买家加1块，卖方降0.5块
        self.aggression = None  # 负响应系数
        self.initial_drop = None # 第一轮固定降幅（没有历史数据可参考）

        self.prev_buyer_offer = None
        self.prev_seller_price = None
        self.personality = None
        self.extalk = None

    def reset(self, seller_start, all_round):
        self.prev_buyer_offer = None
        self.prev_seller_price = seller_start
        self.initial_drop = random.uniform(0.02, 0.10)
        self.aggression = random.uniform(0.1, 0.5) * self.responsiveness
        self.personality = random.randint(1,all_round-1)
        self.extalk = 0

    def decide(self, buyer_offer,seller_bottom):

        if self.prev_buyer_offer is None:
            drop = round(self.prev_seller_price * self.initial_drop,1)
        else:
            buyer_delta = buyer_offer - self.prev_buyer_offer
            if buyer_delta > 0:
                    # 买家加价：卖方降价
                drop = round(buyer_delta * self.responsiveness,1)
            elif buyer_delta == 0:
                if self.extalk < self.personality:
                    drop = round(max(0.5,self.extalk * self.responsiveness),1)
                else:
                    drop = -1*round(max(0.5, self.extalk * self.responsiveness) ,1)
                self.extalk = self.extalk + 1
            else:
                    # 买家降价：卖方涨价（但幅度更小）
                drop = round(buyer_delta * self.aggression,1)  # 负数，等于涨价

            # 计算新报价，不低于底线
        new_price = max(seller_bottom, self.prev_seller_price - drop)

            # 更新
        self.prev_buyer_offer = buyer_offer
        self.prev_seller_price = new_price

            # 买家出价 >= 当前报价  成交
        if buyer_offer is not None and buyer_offer >= new_price:
            return "accept", buyer_offer
        return "offer", new_price

class Buyer:

    def __init__(self, buyer_top, start_offer):
        self.buyer_top = buyer_top    # 预算上限
        self.start_offer = start_offer
        self.reset()

    def reset(self):
        """每局开始时重置公共状态"""
        self.prev_offer = self.start_offer
        self.epoch = 0

    def next_offer(self):
        """子类必须实现：返回这轮的新出价"""
        raise NotImplementedError("子类无加价逻辑")

    def decide(self, seller_price):
        """统一入口：看卖家报价，决定接受还是还价"""
        # 卖家报价 <= 我上次出的价 → 我当然接受（本来就想付这么多）
        if seller_price <= self.prev_offer:
            return "accept", min(seller_price, self.prev_offer)

        # 又要出一轮新价，轮次 +1
        self.epoch += 1

        # 算新出价，不能超过预算
        new_offer = round(min(self.next_offer(), self.buyer_top), 1)
        self.prev_offer = new_offer
        return "offer", new_offer


class ConcederBuyer(Buyer):
    """让步型：急着成交，每轮大步加价"""

    def __init__(self, buyer_top, start_offer=60, step=8):
        super().__init__(buyer_top, start_offer)  # 先让基类初始化
        self.step = step  # 每轮加价量

    def next_offer(self):
        return self.prev_offer + self.step



class TimeDependentBuyer(Buyer):
    """时间依赖型：前期慢慢磨，越到 deadline 加价越猛"""

    def __init__(self, buyer_top, max_rounds, start_offer=60, base_step=10):
        super().__init__(buyer_top, start_offer)
        self.max_rounds = max_rounds  # 总轮数（由 simulate.py 传进来）
        self.base_step = base_step    # 最后关头的最大步长

    def next_offer(self):
        # 进度 = 当前轮次 / 总轮数，前几轮接近 0，最后一轮接近 1
        progress = self.epoch / self.max_rounds
        # 平方曲线：前期涨得慢（0.1^2=0.01），后期涨得快（0.9^2=0.81）
        step = self.base_step * progress ** 3
        return self.prev_offer + step



class GrandetBuyer(Buyer):
    """一口价"""
    def __init__(self, buyer_top, start_offer=60,jitter=15):
        super().__init__(buyer_top, start_offer)
        self.cost = round(start_offer + random.uniform(0, jitter), 1)

    def next_offer(self):
        return self.cost


class RandomBuyer(Buyer):
    """随机型：每轮加价量随机，无规律（鲁棒性测试用）"""
    def __init__(self, buyer_top, start_offer=60,max_step=15):
        super().__init__(buyer_top, start_offer)
        self.max_step = max_step

    def next_offer(self):
        return self.prev_offer + random.uniform(0, self.max_step)
import random


class TitForTatSeller:
    def __init__(self):
        self.responsiveness = 0.5
        self.aggression = None
        self.initial_drop = None
        self.prev_buyer_offer = None
        self.prev_seller_price = None
        self.personality = None
        self.extalk = None

    def reset(self, seller_start, all_round):
        self.prev_buyer_offer = None
        self.prev_seller_price = seller_start
        self.initial_drop = random.uniform(0.02, 0.10)
        self.aggression = random.uniform(0.1, 0.5) * self.responsiveness
        self.personality = random.randint(1, all_round - 1)
        self.extalk = 0

    def decide(self, buyer_offer, seller_bottom):
        if self.prev_buyer_offer is None:
            drop = round(self.prev_seller_price * self.initial_drop, 1)
        else:
            buyer_delta = buyer_offer - self.prev_buyer_offer
            if buyer_delta > 0:
                drop = round(buyer_delta * self.responsiveness, 1)
            elif buyer_delta == 0:
                if self.extalk < self.personality:
                    drop = round(max(0.5, self.extalk * self.responsiveness), 1)
                else:
                    drop = -round(max(0.5, self.extalk * self.responsiveness), 1)
                self.extalk += 1
            else:
                drop = round(buyer_delta * self.aggression, 1)

        new_price = max(seller_bottom, self.prev_seller_price - drop)
        self.prev_buyer_offer = buyer_offer
        self.prev_seller_price = new_price
        if buyer_offer >= new_price:
            return "accept", buyer_offer
        return "offer", new_price


class Buyer:
    """Base buyer with a private reservation price for one negotiation game."""

    def __init__(self, buyer_top, start_offer=60, reservation_price=None):
        self.buyer_top = buyer_top
        self.start_offer = start_offer
        self.reservation_price = (
            buyer_top if reservation_price is None else round(reservation_price, 1)
        )
        if not 0 <= self.start_offer <= self.reservation_price <= self.buyer_top:
            raise ValueError(
                "Expected 0 <= start_offer <= reservation_price <= buyer_top"
            )
        self.reset()

    def reset(self):
        self.prev_offer = self.start_offer
        self.prev_seller_price = None
        self.epoch = 0

    def next_offer(self, seller_price):
        raise NotImplementedError

    def decide(self, seller_price):
        if seller_price <= self.prev_offer:
            return "accept", seller_price

        self.epoch += 1
        raw_offer = self.next_offer(seller_price)
        new_offer = min(
            self.reservation_price,
            max(self.prev_offer, round(raw_offer, 1)),
        )
        self.prev_seller_price = seller_price
        self.prev_offer = new_offer
        return "offer", new_offer


class ConcederBuyer(Buyer):
    """Raises its offer by a fixed step until its reservation price."""

    def __init__(self, buyer_top, start_offer=60, step=8, reservation_price=None):
        super().__init__(buyer_top, start_offer, reservation_price)
        self.step = step

    def next_offer(self, seller_price):
        return self.prev_offer + self.step


class TimeDependentBuyer(Buyer):
    """Boulware-style buyer: it concedes slowly until the deadline."""

    def __init__(
        self,
        buyer_top,
        max_rounds,
        start_offer=60,
        exponent=3,
        reservation_price=None,
        base_step=None,
    ):
        super().__init__(buyer_top, start_offer, reservation_price)
        self.max_rounds = max_rounds
        self.exponent = exponent
        self.base_step = base_step

    def next_offer(self, seller_price):
        progress = min(self.epoch / self.max_rounds, 1.0)
        concession = progress ** self.exponent
        return self.start_offer + (self.reservation_price - self.start_offer) * concession


class GrandetBuyer(Buyer):
    """Makes one fixed offer equal to its reservation price."""

    def __init__(self, buyer_top, start_offer=60, jitter=15, reservation_price=None):
        if reservation_price is None:
            reservation_price = min(
                buyer_top, round(start_offer + random.uniform(0, jitter), 1)
            )
        super().__init__(buyer_top, start_offer, reservation_price)
        self.cost = self.reservation_price

    def next_offer(self, seller_price):
        return self.reservation_price


class ReactiveBuyer(Buyer):
    """Only concedes after a sufficiently large seller concession."""

    def __init__(
        self,
        buyer_top,
        start_offer=60,
        match_rate=0.7,
        min_seller_drop=2.0,
        reservation_price=None,
    ):
        super().__init__(buyer_top, start_offer, reservation_price)
        self.match_rate = match_rate
        self.min_seller_drop = min_seller_drop

    def next_offer(self, seller_price):
        if self.prev_seller_price is None:
            return self.prev_offer
        seller_drop = self.prev_seller_price - seller_price
        if seller_drop < self.min_seller_drop:
            return self.prev_offer
        return self.prev_offer + seller_drop * self.match_rate


class RandomBuyer(Buyer):
    """Monotonically raises a random amount; use as a noise baseline only."""

    def __init__(self, buyer_top, start_offer=60, max_step=15, reservation_price=None):
        super().__init__(buyer_top, start_offer, reservation_price)
        self.max_step = max_step

    def next_offer(self, seller_price):
        return self.prev_offer + random.uniform(0, self.max_step)


class MirrorBuyer(Buyer):
    """以牙还牙型买方：卖方降多少，我就按比例加多少，另加一点保底加价。

    训练时不出场，只在评估阶段用来检验 DQN 卖家对"没见过的对手"是否也有效。
    """

    def __init__(
        self,
        buyer_top,
        start_offer=60,
        match_rate=0.6,
        min_step=1.5,
        reservation_price=None,
    ):
        super().__init__(buyer_top, start_offer, reservation_price)
        self.match_rate = match_rate
        self.min_step = min_step

    def next_offer(self, seller_price):
        if self.prev_seller_price is None:
            return self.prev_offer + self.min_step
        seller_drop = self.prev_seller_price - seller_price
        return self.prev_offer + max(self.min_step, seller_drop * self.match_rate)

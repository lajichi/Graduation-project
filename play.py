"""谈判小游戏 —— 你来当买家，和 AI 卖家砍价

卖方策略：Tit-for-Tat 响应式
- 买家加价多 → 卖方少降（吊着，因为买家急）
- 买家加价少 → 卖方多降（示好，推动成交）
- 响应系数目前固定，后期交给 RL 学习
"""
import random
from strategies import TitForTatSeller

SELLER_BOTTOM = random.randint(1, 50)
BUYER_TOP = 100
SELLER_START = random.randint(SELLER_BOTTOM, BUYER_TOP-1)
MAX_ROUNDS = 10

def play():
    # 每局重置状态
    seller = TitForTatSeller()
    seller.reset(SELLER_START,MAX_ROUNDS)

    print("=" * 40)
    print("  谈判开始！你是买家")
    print(f"  你的预算上限: {BUYER_TOP}")
    print(f"  最多 {MAX_ROUNDS} 轮，谈不成就崩")
    print(f"  卖方策略: Tit-for-Tat (响应系数={seller.responsiveness})")
    print("=" * 40)

    for r in range(1, MAX_ROUNDS + 1):
        print(f"\n--- 第 {r} 轮 ---")

        raw = input("你出价（输入数字）: ").strip()
        try:
            buyer_offer = float(raw)
        except ValueError:
            print("不是数字，算你弃权这轮")
            continue

        if buyer_offer > BUYER_TOP:
            print(f"你最多只能出 {BUYER_TOP}！")
            continue

        action, price = seller.decide( buyer_offer,SELLER_BOTTOM)

        if action == "accept":
            profit = BUYER_TOP - buyer_offer
            print(f"\n卖方接受了！成交价: {buyer_offer}")
            print(f"你的收益: {profit}（预算 {BUYER_TOP} - 成交 {buyer_offer}）")
            return
        else:
            print(f"卖方还价: {price}")

    print(f"\n{MAX_ROUNDS} 轮谈完，没成交。双方都没收益。")


if __name__ == "__main__":
    play()

class Solution:
    def maxProfit(prices):
        buyprice = prices[0]
        profit = 0
        for i in range(1,len(prices)):
            if buyprice > prices[i]:
                buyprice = prices[i]
            elif prices[i] - buyprice > profit:
                profit = prices[i] - buyprice

        return profit

#max_profit1 = Solution.maxProfit(prices = [7,1,5,3,6,4]) #5
#max_profit1 = Solution.maxProfit(prices = [7,6,4,3,1]) #0
max_profit1 = Solution.maxProfit(prices = [1,2]) #1

print(max_profit1)
""" time limit exceeded code:
class Solution:
    def maxProfit(prices):
        buyprice = 0
        profit = []
        for i in range(len(prices)):
            #print(prices[i])
            buyprice = prices[i]
            for j in range(len(prices)):
                if j>i:
                    if prices[j] > buyprice:
                        sellprice = prices[j]
                    else:
                        continue #sellprice = 0
                    if sellprice:
                        subtract = sellprice - buyprice
                        if subtract > 0:
                            #print(profit)
                            profit.append(subtract)
                    else:
                        continue
        if profit:
            max_profit = max(profit)
        else:
            return 0

        return max_profit
"""




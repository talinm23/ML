



class Solution:
    def maxSubArray(nums):
        res = nums[0]
        total = 0
        for i in nums:
            if total < 0:
                total = 0
            total+=i
            res = max(res, total)
        return res

#nums = [-2,1,-3,4,-1,2,1,-5,4] # 6
nums = [5,4,-1,7,8] #23
output = Solution.maxSubArray(nums)
print(output)

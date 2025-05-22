class Solution:
    def singleNumber(nums):
        if len(nums) == 1:
            return nums[0]

        a = []
        for i in nums:
            if i in a:
                a.remove(i)
            else:
                a.append(i)
        return a.pop()

#print(Solution.singleNumber([4,1,2,1,2])) #4
print(Solution.singleNumber([2,2,1])) #1


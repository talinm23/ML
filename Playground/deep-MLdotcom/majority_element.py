class Solution:
    def majorityElement(nums):
        element = 0
        count = 0
        for i in nums:
            if count == 0:
                element = i
            if i == element:
                count += 1
            else:
                count = 1
        return element

#print(Solution.majorityElement([2,2,1,1,1,2,2]))
#print(Solution.majorityElement([3,2,3]))
print(Solution.majorityElement([5,6,3,7,6,3,4,6,7]))







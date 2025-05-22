class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        new_s = ''.join(e for e in s if e.isalnum())
        if new_s == new_s[::-1]:
            return True
        else:
            return False
        print(new_s)


#s = "yes4sey" #True
#s ="A man, a plan, a canal: Panama" #True
s= "race a car" #false
#s=" " #true


print(Solution().isPalindrome(s))


# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:36:27 2022

@author: user
"""

"while 예제1"
num1 = int(input('숫자1를 입력하시오: '))
i=1

while i <= num1:
    print(' ',num1)
    i=i+1

"while 예제2"    
num2 = int(input('숫자2를 입력하시오: '))
i=1

while i <= num2:
    print(i,' ',i*i)
    i=i+1
 
    
print('\n')    


"while 예제3"
num3 = 100
i=1

while i <= 10:
    height = round(num3*3/5,4)
    print(i,' ', height)
    i=i+1
    num3=height
    
print('\n')  


"while 예제4"    
number = 358

rem = rev = 0
while number >= 1:
    rem = number % 10
    rev = rev * 10 + rem
    number = number // 10

print(rev)
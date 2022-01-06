# -*- coding: utf-8 -*-
"""
위키도스 연습문제
"""

print('변수-파일크기 계산 연습문제1\n')
speed = 0.8
duration = 110
file_size = speed * duration
print(file_size)

print('\n')  

print('직각삼각형 그리기')
leg = int(input('변의 길이: '))

for i in range(leg):
    print('* ' * (i + 1))

area = (leg ** 2) / 2
print('넓이:', area)

print('\n')  

print('제곱값 구하기\n')
x = int(input('숫자를 입력하시오: '))
result = x*x
print(x,'의 제곱값:', result)

print('\n')  

print('while 연습 문제1: 입력받은 숫자만큼 반복하기\n')
num1 = int(input('숫자1를 입력하시오: '))
i=1

while i <= num1:
    print(' ',num1)
    i=i+1

print('\n')  
    
print('while 연습 문제2: 제곱표\n')
num2 = int(input('숫자2를 입력하시오: '))
i=1

while i <= num2:
    print(i,' ',i*i)
    i=i+1
 
print('\n')    
    
print('while 연습 문제3: 얌체공\n')
num3 = 100
i=1

while i <= 10:
    height = round(num3*3/5,4)
    print(i,' ', height)
    i=i+1
    num3=height
    
print('\n')  

print('while 연습 문제4: 코드를 보고 실행 결과 맞히기\n')
number = 358

rem = rev = 0
while number >= 1:
    rem = number % 10
    rev = rev * 10 + rem
    number = number // 10

print(rev)

print('\n')  

print('If 연습 문제1: 숫자 읽기(1~3)\n')
num1 = int(input('숫자1를 입력하시오: '))
if num1 == 1:
        print('일')
elif num1 == 2:
        print('이')
elif num1 == 3:
        print('삼')
        
print('\n')      

print('If 연습 문제2: 양수만 덧셈하기\n')
sum = 0

while True:
    num2 = int(input('숫자2를 입력하시오: '))
    if num2 > 0:
        sum += num2
    elif num2 < 0:
        print(sum)
        break
        
 
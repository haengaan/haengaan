# -*- coding: utf-8 -*-=============================================================================
"""
위키도스 연습문제
"""
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

# while 문 ==========================================================================================
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

# If문 ==========================================================================================
print('If 연습 문제2: 양수만 덧셈하기\n')
sum = 0

while True:
    num2 = int(input('숫자2를 입력하시오: '))
    if num2 > 0:
        sum += num2
    elif num2 < 0:
        print(sum)
        break
    
print('\n')      


# For문 ==========================================================================================

print('for 연습 문제2: 제곱표(for)\n')
num1 = int(input('숫자를 입력하시오: '))
for i in range(1,num1+1):
    print(i,' ',i*i)
 
print('\n') 

\
print('for 연습 문제3: 화학 실험실\n')
min_tmp, max_tmp = map(int,input('최소, 최대 온도를 입력하시오: ').split())

temps = input('장치 온도를 입력하시오. ').split()

for i in range(len(temps)):
    if (int(temps[i]) >= min_tmp and int(temps[i]) <= max_tmp):
        print('Nothing to report')
    else:
        print('Alert!')
        
        
print('\n')         

print('함수 연습 문제: 구구단\n')
def gugu(a):
    for i in range(1,10):
        print(a,'x',i,'=',a*i)
        
for j in range(2,10):
    gugu(j)

print('\n')  


print('반환문 연습 문제1: 숫자 읽기 함수(1~5)\n')
def korean_number(a):
    if a == 1:
        print('일')
    elif a == 2:
        print('이')
    elif a == 3:
        print('삼')
    elif a == 4:
        print('사')
    elif a == 5:
        print('오')
        
korean_number(3)

print('\n')


print('반환문 연습 문제2: 함수 정의하기\n')
def triple(x):
    return x * 3
 
print(triple(2))
print(triple('x'))

print('\n')
 

print('반환문 연습 문제2: 날짜 객체\n') 
    
def korean_age(birth_year):
    from datetime import datetime
    today = datetime.today()
    return today.year - birth_year + 1    

print(korean_age(1993))
print('\n')
 
""" 
print('함수 연습 문제1: 단리 이자\n') 
def simple_interest(p,r,t):
    interest = p*r*t
    return interest

print(simple_interest(1100000, 0.05, 5/12))

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

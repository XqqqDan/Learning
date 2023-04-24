import random


def func(counter):
    a = 0
    b = 0
    while 1:
        counter = counter + 1
        a = b
        b = random.randint(1, 6)
        if (
            (a == 6 and b == 1)
            or (a == 1 and b == 6)
            or (a == 1 and b == 1)
            or (a == 6 and b == 6)
        ):
            break
    return counter


s = 0
n = 10000
for i in range(n):
    s = s + func(0)

print(s / n)

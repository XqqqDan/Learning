#include <iostream>

using pf_type = void (*)(int);
void myFunc(pf_type pf, int i)
{
    pf(i);
}
void test(int i)
{
    std::cout << i << std::endl;
    std::cout << "hello world" << std::endl;
}

using pf_type2 = int (*)(int, int);
int myFunc2(pf_type2 pf, int i, int j)
{
    return pf(i, j);
}
int test2(int i, int j)
{
    int sum = i + j;
    return sum;
}

int main()
{
    myFunc(test, 100);
    std::cout << myFunc2(test2, 100, 200) << std::endl;
    return 0;
}

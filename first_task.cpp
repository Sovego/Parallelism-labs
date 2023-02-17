#include <iostream>
#include <math.h>
#include <chrono>
#define n 10000000
int main()
{
    auto begin = std::chrono::steady_clock::now();
    double sum = 0;
    double *array;
    array = new double [n];
    double pi = acos(-1);
    #pragma acc enter data create(array[0:n],sum) copyin(pi)

    #pragma acc parallel loop present(array[0:n],pi)
    for (int i = 0; i < n; ++i)
    {
        array[i] = sin(2*pi/n*i);
    }

    #pragma acc parallel loop present(array[0:n],sum) reduction(+:sum)
    for (int i = 0; i < n; ++i)
    {
        sum += array[i];
    }

    #pragma acc exit data delete(array[0:n]) copyout(sum)

    std::cout << sum << std::endl;
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    return 0;
}
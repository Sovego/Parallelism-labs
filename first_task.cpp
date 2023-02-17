#include <iostream>
#include <math.h>
#define n 10000000
int main()
{
    double sum = 0;
    double *array;
    array = new double [n];
    double pi = acos(-1);
    #pragma acc enter data create(array[0:n],sum) copyin(pi)

    #pragma acc parallel loop present(array[0:n],pi)
    for (int i = 0; i < n; ++i)
    {
        array[i] = sin(2*pi/i*n);
    }

    #pragma acc parallel loop present(array[0:n],sum) reduction(+:sum)
    for (int i = 0; i < n; ++i)
    {
        sum += array[i];
    }

    #pragma acc exit data delete(array[0:n]) copyout(sum)

    std::cout << sum << std::endl;

    return 0;
}
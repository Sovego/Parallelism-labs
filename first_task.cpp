#include <iostream>
#include <math.h>
int main()
{
    double sum = 0;
    double *array;
    array = new double [100];

    #pragma acc enter data create(array[0:100],sum)

    #pragma acc parallel loop present(array[0:100])
    for (int i = 0; i < 100; ++i)
    {
        array[i] = sin(35*3.14/180);
    }

    #pragma acc parallel loop present(array[0:100],sum) reduction(+:sum)
    for (int i = 0; i < 100; ++i)
    {
        sum += array[i];
    }

    #pragma acc exit data delete(array[0:100]) copyout(sum)

    std::cout << sum << std::endl;

    return 0;
}
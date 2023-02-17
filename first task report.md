# Название компилятора
    pgc++
# Время работы циклов на видеокарте double

|Type           |Time(%)|Time   |Calls|Avg     |Min     |Max     |Name            |
|---            |---    |---    |---  |---     |---     |---     |---             |
|GPU activities:|35.80%|131.71us| 1   |131.71us|131.71us|131.71us|main_16_gpu     |
|               |35.68%|131.26us| 1   |131.26us|131.26us|131.26us|main_9_gpu      |
|               |27.24%|100.19us| 1   |100.19us|100.19us|100.19us|main_16_gpu__red|

# Время работы циклов на видеокарте float

|Type           |Time(%)|Time   |Calls|Avg     |Min     |Max     |Name            |
|---            |---    |---    |---  |---     |---     |---     |---             |
|GPU activities:|35.80%|120.54us| 1   |120.54us|120.54us|120.54us|main_16_gpu     |
|               |35.68%|116.99us| 1   |116.99us|116.99us|116.99us|main_9_gpu      |
|               |27.24%|88.543us| 1   |88.54us|88.54us|88.54us|main_16_gpu__red|
# Общее время работы на видеокарте
    Time: 3287 ms
# Общее работы на процессоре double один поток
    Time: 250 ms
# Общее работы на процессоре float один поток
    Time: 250 ms
# Время работы циклов на процессоре в многопотоке double
|Type           |Time(%)|Time   |Calls|Avg     |Min     |Max     |Name            |
|---            |---    |---    |---  |---     |---     |---     |---             |
|OpenACC (excl):|97.38%|21.785ms| 1   |21.785ms|21.785ms|21.785ms|acc_compute_construct@iostream:11|
|               |2.62%|585.87us| 1   |585.87us|585.87us|585.87us|acc_compute_construct@iostream:18 |
# Время работы на процессоре double многопоток
    Time: 20 ms
# Время работы на процессоре float многопоток
    Time: 20 ms
# Точность на видеокарте double 
    -3.12639e-12
# Точность на видеокарте float 
    -0.0277023
# Точность на процессоре float 
    0.349212
# Точность на процессоре double
    -6.76916e-10
# Диаграмма сравнения общего времени работы программы на видеокарте и выполнения циклов
![alt text](https://i.imgur.com/Dg4z7CX.png "Диаграмма видеокарта")
# Диаграмма сравнения общего времени работы программы на процессоре (многопоток) и выполнения циклов
![alt text](https://i.imgur.com/ZAyKIhx.png "Диаграмма процессор")
# Код программы для CPU
```
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
```
# Код программы для GPU
```
#include <iostream>
#include <math.h>
#include <chrono>
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
        array[i] = sin(2*pi/n*i);
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
```
# Выводы
    Из результатов работы можно сделать вывод что для решения данной задачи оптимальнее использовать CPU так как передача данных на GPU занимает
    слишком много времени, по сравнению с которым время работы циклов практически не влияет на время работы программы в целом
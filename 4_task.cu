#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_runtime.h"
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#define at(arr, x, y) (arr[(x) * (n) + (y)])

// Values
constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
constexpr int THREADS_PER_BLOCK_REDUCE = 256;

// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

// Function definitions
void initArrays(double *mainArr, double *main_D, double *sub_D, int n, int m){

size_t size = n * m * sizeof(double);
at(mainArr, 0, 0) = LEFT_UP;
at(mainArr, 0, m - 1) = RIGHT_UP;
at(mainArr, n - 1, 0) = LEFT_DOWN;
at(mainArr, n - 1, m - 1) = RIGHT_DOWN;

for (int i = 1; i < n - 1; i++) {
    at(mainArr, 0, i) = (at(mainArr, 0, m - 1) - at(mainArr, 0, 0)) / (m - 1) * i + at(mainArr, 0, 0);
    at(mainArr, i, 0) = (at(mainArr, n - 1, 0) - at(mainArr, 0, 0)) / (n - 1) * i + at(mainArr, 0, 0);
    at(mainArr, n - 1, i) = (at(mainArr, n - 1, m - 1) - at(mainArr, n - 1, 0)) / (m - 1) * i + at(mainArr, n - 1, 0);
    at(mainArr, i, m - 1) = (at(mainArr, n - 1, m - 1) - at(mainArr, 0, m - 1)) / (m - 1) * i + at(mainArr, 0, m - 1);
}
cudaMemcpy(main_D, mainArr, size, cudaMemcpyHostToDevice);
cudaMemcpy(sub_D, mainArr, size, cudaMemcpyHostToDevice);
}
// Шаг алгоритма
__global__ void iterate(double* F, double* Fnew, double* subs, int* n_d){

int j = blockIdx.x * blockDim.x + threadIdx.x; // вычисляем ячейку
int i = blockIdx.y * blockDim.y + threadIdx.y;

if(j == 0 || i == 0 || i == *n_d-1 || j == *n_d-1) return; // Dont update borders
// сам алгоритм
int n = *n_d;
at(Fnew, i, j) = 0.25 * (at(F, i+1, j) + at(F, i-1, j) + at(F, i, j+1) + at(F, i, j-1));
at(subs, i, j) = fabs(at(Fnew, i, j) - at(F, i, j));
}
// Блок редукции и магии
__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
    typedef cub::BlockReduce<double, 256> BlockReduce; // создаем блок
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double max_diff = 0;
    // проходим по массивам и находим макс разницу
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
    double diff = abs(in1[i] - in2[i]);
    max_diff = fmax(diff, max_diff);
    }
    // засовываем максимальную разницу в блок редукции
    double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

    if (threadIdx.x == 0)
    {
    out[blockIdx.x] = block_max_diff; // обновляем значение
    }
}

// Главная функция решения
void solve(double* F, double* Fnew, double* subs, int n, double* error, int itermax,int iter,double tol){
    int* n_d;
    cudaMalloc(&n_d, sizeof(int));
    cudaMemcpy(n_d, &n, sizeof(int), cudaMemcpyHostToDevice);
    size_t size = n * n;

    dim3 threadPerBlock = dim3(32, 32); //dim3((n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK, (n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK); // определяем количество потоков на блок
    dim3 blocksPerGrid =  dim3((n + 31) / 32, (n+31)/32);//dim3(((n + (n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK) - 1) / ((n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK),
            // (n + ((n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK) - 1) / ((n + MAXIMUM_THREADS_PER_BLOCK - 1) / MAXIMUM_THREADS_PER_BLOCK)); // определяяем количество блоков на сетку

    int num_blocks_reduce = (size + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE; // количество блоков редукции
    // выделяем память под ошибку блочной редукции
    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, subs, error, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    double* tmp_err = (double*)malloc(sizeof(double)); 
    do {
        iterate<<<blocksPerGrid, threadPerBlock>>>(F, Fnew, subs, n_d); // проходим алгоритмом
        iterate<<<blocksPerGrid, threadPerBlock>>>(Fnew, F, subs, n_d); // проходим алгоритмом
        if (iter%(400)==0)
        {
            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F, Fnew, size, error_reduction); // по блочно проходим редукцией
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error, num_blocks_reduce); // проходим редукцией по всем блокам 
            cudaMemcpy(tmp_err, error, sizeof(double), cudaMemcpyDeviceToHost);
        }
        

        (iter)+=2;
    } while(*tmp_err > tol && itermax > iter);
    cudaFree(n_d);

}

bool isUint(const std::string& s){
    return s.find_first_not_of("0123456789") == std::string::npos;
}
/// @brief Сommand line help
void commandHelp()
{
    std::cout << "-a <Accuracy> -s <Grid size> -m <Max iteration count>" << std::endl;
}
int main(int argc, char *argv[]){
    auto begin = std::chrono::steady_clock::now(); // start time
    cudaSetDevice(3); // selecting free GPU device
// Arguments check Begin
    if (argv[1]=="-h") // check help argument
    {
        commandHelp();
        return 0;
    }
    if (argc-1!=6) // check count of arguments
    {
        throw std::runtime_error("Argument count not enough. Use -h for help"); 
    }
    if (!isUint(argv[4]) || !isUint(argv[6])) // check type of arguments
    {
        throw std::runtime_error("Argument invalid. Use -h for help");
    }
    int n {std::stoi(argv[4])},m {std::stoi(argv[4])}; // n m - grid size
    double error {1},tol {std::stod(argv[2])}; // error - error value. tol - accuracy
    int iter {0},iter_max {std::stoi(argv[6])}; // iter - iterator. iter_max - max iteration count
    double *F_H;
    double* F_D, *Fnew_D, *substractions;
    size_t size = n * m * sizeof(double);
    std::cout << "-----------------------------" << std::endl;
    std::cout << "- Accuracy: " << tol << std::endl << "- Max iteration count: "<< iter_max << std::endl << "- Grid size: " << n << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    int iterationsElapsed = 0;

    cudaMalloc(&F_D, size);
    cudaMalloc(&Fnew_D, size);
    cudaMalloc(&substractions, size);
    F_H = (double *)calloc(sizeof(double), size);
    initArrays(F_H, F_D, Fnew_D,n,m);
    nvtxRangePush("MainCycle");
    {
        // объявляем переменные на видеокарте
        double* error_d;
        int* iterations_d;
        int* iter_max_d;
        double* tol_d;
        cudaMalloc(&error_d, sizeof(double));
        cudaMalloc(&iterations_d, sizeof(int));
        cudaMalloc(&iter_max_d,sizeof(int));
        cudaMalloc(&tol_d, sizeof(double));
        cudaMemcpy(iter_max_d, &iter_max, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(tol_d, &tol, sizeof(double), cudaMemcpyHostToDevice);
        // запускаем решение
        solve(F_D, Fnew_D, substractions, n, error_d,iter_max,iter,tol);
        // копирууем данные на хост
        cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&iterationsElapsed, iterations_d, sizeof(int), cudaMemcpyDeviceToHost);
        // чистим память видеокарты
        cudaFree(error_d);
        cudaFree(iterations_d);
        cudaFree(iter_max_d);
        cudaFree(tol_d);
    }
    nvtxRangePop();
    std::cout << "Iterations: " << iterationsElapsed << std::endl;
    std::cout << "Error: " << error << std::endl;

    cudaFree(F_D);
    cudaFree(Fnew_D);
    free(F_H);
    auto end = std::chrono::steady_clock::now(); // Code end time
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); // Calculate execution time
    std::cout << "The time: " << elapsed_ms.count() << " ms\n"; // Output execution time
    return 0;
}








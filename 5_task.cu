
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include "cuda_runtime.h"
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif



constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
#define THREAD_PER_BLOCK_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, max_thread) ((arr_lenght_dim_1+max_thread-1)/max_thread), ((arr_lenght_dim_2+max_thread-1)/max_thread)
#define BLOCK_COUNT_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, threads_count) ((arr_lenght_dim_1 + threads_count.x - 1) / threads_count.x), ((arr_lenght_dim_2 + threads_count.y - 1) / threads_count.y)



// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int THREADS_PER_BLOCK_REDUCE = 256;

constexpr int ITERS_BETWEEN_UPDATE = 400;

typedef struct cmdArgs{
    bool showResultArr;
    bool initUsingMean;
    double eps;
    int iterations;
    int n; // Count of rows
    int m; // Count of collumns
} cmdArgs;

template<typename T>
T extractArgument(char* arr){
    std::stringstream stream;
    stream << arr;
    T result;
    if (!(stream >> result)){
        throw std::invalid_argument("Wrong argument type");
    }
    return result;
}

void processArgs(int argc, char *argv[], cmdArgs* args){
    for(int arg = 0; arg < argc; arg++){
        if(std::strcmp(argv[arg], "-eps") == 0){
            args->eps = extractArgument<double>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-i") == 0){
            args->iterations = extractArgument<int>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-s") == 0){
            args->n = extractArgument<int>(argv[arg+1]);
            args->m = args->n;
            arg++;
        }
    }
}


void printSettings(cmdArgs* args){
    std::cout << "------------------------------" << std::endl;
    std::cout << "\tAccur: " << args->eps << std::endl;
    std::cout << "\tMax iteration: " << (double)(args->iterations) << std::endl;
    std::cout << "\tSize: " << args->n << 'x' << args->m << std::endl;
}



#define at(arr, x, y) (arr[(x) * (n) + (y)])
void transfer_data(const int rank, const int ranks_count, double* F_from, double* F_to, cmdArgs& local_args, cudaStream_t stream = 0){

    if(rank != 0){
        cudaMemcpyAsync(F_from + local_args.m, F_to + local_args.m, local_args.m * sizeof(double), cudaMemcpyDeviceToHost, stream);
        MPI_Request rq;
        // отправляем указатель на вторую строку процессу, работующему сверху
        MPI_Isend(
            F_from + local_args.m,
            local_args.m,
            MPI_DOUBLE,
            rank-1,
            rank-1,
            MPI_COMM_WORLD,
            &rq
        );
    }

    // отправляем свою вторую строку вниз
    if(rank != ranks_count-1){
        MPI_Request rq;
        cudaMemcpyAsync(F_from + local_args.m*(local_args.n-2), F_to + local_args.m*(local_args.n-2), local_args.m * sizeof(double), cudaMemcpyDeviceToHost, stream);
        MPI_Isend(
            F_from + local_args.m*(local_args.n-2),
            local_args.m,
            MPI_DOUBLE,
            rank+1,
            rank+1,
            MPI_COMM_WORLD,
            &rq
        );
    }

    // принимаем строку от верхнего
    if(rank != 0){
        MPI_Status status;
        MPI_Recv(F_from, local_args.m, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(F_to, F_from, local_args.m * sizeof(double), cudaMemcpyHostToDevice, stream);

    }
    // принимаем строку от нижнего
    if(rank != ranks_count - 1){
        MPI_Status status;
        MPI_Recv(F_from+(local_args.m * (local_args.n-1)), local_args.m, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(F_to+(local_args.m * (local_args.n-1)), F_from+(local_args.m * (local_args.n-1)), local_args.m * sizeof(double), cudaMemcpyHostToDevice, stream);
    }
}
__global__ void iterate(double *F, double *Fnew, const cmdArgs *args){

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0 || i == 0 || i >= args->n - 1 || j >= args->m - 1) return; // Don't update borders

    int n = args->m;
    at(Fnew, i, j) = 0.25 * (at(F, i + 1, j) + at(F, i - 1, j) + at(F, i, j + 1) + at(F, i, j - 1));
}


__global__ void block_reduce(const double *in1, const double *in2, const int n, double *out){
    typedef cub::BlockReduce<double, THREADS_PER_BLOCK_REDUCE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double max_diff = 0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double diff = fabs(in1[i] - in2[i]);
    max_diff = fmax(diff, max_diff);

    double block_max_diff = BlockReduce(temp_storage).Reduce(max_diff, cub::Max());

    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = block_max_diff;
    }
}
int main(int argc, char *argv[]) {

// ------------------
// Подготовка к работе
// ------------------

    // ====== Инициализируем MPI ======
    int rank, ranks_count;
    MPI_Init(&argc, &argv);

    // ====== Определяем сколько процессов внутри глоабльного коммуникатора ======
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);

    // ====== Каждый процесс выбирает свою видеокарту ======
    cudaSetDevice(rank);

    // ====== Парсинг аргументов командной строки ======
    cmdArgs global_args = cmdArgs{1E-6, (int)1E6, 16, 16};
    processArgs(argc, argv, &global_args);

    if(rank == 0){
        printSettings(&global_args);
    }

    // ====== Расчет элементов для каждого процесса ======
    int TOTAL_GRID_SIZE = global_args.m * global_args.n;

    cmdArgs local_args{global_args};
    local_args.n =  TOTAL_GRID_SIZE / ranks_count / global_args.m + 2 * (rank != ranks_count - 1);

    int ELEMENTS_BY_PROCESS = local_args.n * local_args.m;

    // ====== Создание указателей на массивы ======
    double *F_H_full = nullptr; // Указатель для хранения всего массива (используется в rank = 0)
    double *error_array = nullptr; // Указатель для хранения массива ошибок полученных с остальных процессов на нулевой (используется в rank 0)
    double *F_H;
    double *F_D, *Fnew_D;
    size_t array_size_bytes = ELEMENTS_BY_PROCESS * sizeof(double);

    // ====== Выделяем память для GPU/CPU ======
    cudaMalloc(&F_D, array_size_bytes);
    cudaMalloc(&Fnew_D, array_size_bytes);

    cudaMallocHost(&F_H, array_size_bytes);

    if(rank == 0){
        error_array = (double*)malloc(sizeof(double) * ranks_count);
    }

// ------------------
// Иницилизируем массив в 0 процессе и отправляем всем остальным процессам их части
// Каждый процесс обрабатывает local_args.n - строк
// Это значение зависит от того какой участок обрабатывает наш процесс
// В итоге получаем что каждый процесс обрабатывает global_args.n / 4 + 2 строк
// Кроме последнего, он обрабатывает только global_args.n / 4 строк
// Это происходит из-за того, что ему нет необходимости поддерживать граничные значения с нижним блоком (он является самым нижним)
// ------------------
{
    int n = global_args.n;
    int m = global_args.m;

    // Заполняем полный массив в 0 процессе
    if(rank == 0){

        F_H_full = (double*)calloc(n*m, sizeof(double));

        for (int i = 0; i < n * m && global_args.initUsingMean; i++){
            F_H_full[i] = (LEFT_UP + LEFT_DOWN + RIGHT_UP + RIGHT_DOWN) / 4;
        }

        at(F_H_full, 0, 0) = LEFT_UP;
        at(F_H_full, 0, m - 1) = RIGHT_UP;
        at(F_H_full, n - 1, 0) = LEFT_DOWN;
        at(F_H_full, n - 1, m - 1) = RIGHT_DOWN;

        for (int i = 1; i < n - 1; i++) {
            at(F_H_full, 0, i) = (at(F_H_full, 0, m - 1) - at(F_H_full, 0, 0)) / (m - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, i, 0) = (at(F_H_full, n - 1, 0) - at(F_H_full, 0, 0)) / (n - 1) * i + at(F_H_full, 0, 0);
            at(F_H_full, n - 1, i) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, n - 1, 0)) / (m - 1) * i + at(F_H_full, n - 1, 0);
            at(F_H_full, i, m - 1) = (at(F_H_full, n - 1, m - 1) - at(F_H_full, 0, m - 1)) / (m - 1) * i + at(F_H_full, 0, m - 1);
        }

        int data_start = 0;
        int data_lenght = 0;

    // ------------------
    // Отправляем необходимые части всем процессам, включая самого себя
    // ------------------
        for(size_t target = 0; target < ranks_count; target++){
            MPI_Request req;
            data_lenght = ELEMENTS_BY_PROCESS - 2 * local_args.m * (target == (ranks_count - 1) && ranks_count != 1);
            MPI_Isend(
                F_H_full + data_start,
                data_lenght,
                MPI_DOUBLE,
                target,
                0,
                MPI_COMM_WORLD,
                &req
            );

            data_start += data_lenght - local_args.m * 2;

        }

    }

// ------------------
// Ждём получения обрабатываемой части от 0 процесса
// ------------------
    MPI_Status status;
    MPI_Recv(
        F_H,
        ELEMENTS_BY_PROCESS,
        MPI_DOUBLE,
        0,
        0,
        MPI_COMM_WORLD,
        &status
    );

    cudaMemcpy(F_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Fnew_D, F_H, array_size_bytes, cudaMemcpyHostToDevice);

}

    double error = 1;
    int iterationsElapsed = 0;


// ------------------
// Основной цикл работы программы
// Иницилизация ->
//  =========== ЦИКЛ ============
//      Проход по своему участку
//          Обмен граничными условиями
//      Проход по своему участку
//          Обмен граничными условиями
//      Расчет ошибки
//          Сбор ошибок с каждого процесса и вычисление общей
//          Отправка общей ошибки каждому процессу (Маркер того что основной процесс обработал все их отправленные данные)
// ------------------
{

    cmdArgs *args_d;
    cudaMalloc(&args_d, sizeof(cmdArgs));
    cudaMemcpy(args_d, &local_args, sizeof(cmdArgs), cudaMemcpyHostToDevice);

    int num_blocks_reduce = (ELEMENTS_BY_PROCESS + THREADS_PER_BLOCK_REDUCE - 1) / THREADS_PER_BLOCK_REDUCE;

    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);
    double *error_d;
    cudaMalloc(&error_d, sizeof(double));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadPerBlock {THREAD_PER_BLOCK_DEFINED(local_args.n, local_args.m, MAXIMUM_THREADS_PER_BLOCK)}; // ПЕРЕОСМЫСЛИТЬ
    if(threadPerBlock.x > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.x = MAXIMUM_THREADS_PER_BLOCK;
    }
    if(threadPerBlock.y > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.y = MAXIMUM_THREADS_PER_BLOCK;
    }
    dim3 blocksPerGrid {BLOCK_COUNT_DEFINED(local_args.n, local_args.m, threadPerBlock)}; // ПЕРЕОСМЫСЛИТЬ


    MPI_Barrier(MPI_COMM_WORLD);

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif
    do {

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(F_D, Fnew_D, args_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data(rank, ranks_count, F_H, F_D, local_args, stream);

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(Fnew_D, F_D, args_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data(rank, ranks_count, F_H, Fnew_D, local_args, stream);

        iterationsElapsed += 2;
        if(iterationsElapsed % ITERS_BETWEEN_UPDATE == 0){
            block_reduce<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE, 0, stream>>>(F_D, Fnew_D, ELEMENTS_BY_PROCESS, error_reduction);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error_d, num_blocks_reduce, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(&error, error_d, sizeof(double), cudaMemcpyDeviceToHost);

// ------------------
// Сборка ошибок с каждого процесса и обработка их на 0 потоке (Procces reduction)
// ------------------
            {
                MPI_Gather(&error, 1, MPI_DOUBLE, error_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if(rank == 0){
                    error = 0;
                    for(int err_id = 0; err_id < ranks_count; err_id++){
                        error = max(error, error_array[err_id]);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    } while(error > global_args.eps && iterationsElapsed < global_args.iterations);
#ifdef NVPROF_
    nvtxRangePop();
#endif
    cudaStreamDestroy(stream);
}


    if(global_args.showResultArr){
// ------------------
// Отправка финального массива на нулевой процесс
// ------------------
        cudaMemcpy(F_H, F_D, array_size_bytes, cudaMemcpyDeviceToHost);
        MPI_Request req;
        MPI_Isend(F_H + local_args.m, ELEMENTS_BY_PROCESS - (local_args.m * 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);

// ------------------
// Отображение финального массива с нулевого процесса
// ------------------
        if(rank == 0){
            int array_offset = local_args.m;
            for(int target = 0; target < ranks_count; target++){
                MPI_Status status;
                int recive_size = ELEMENTS_BY_PROCESS - 2 * local_args.m - 2 * local_args.m * (target == (ranks_count - 1));
                MPI_Recv(F_H_full + array_offset, recive_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &status);
                array_offset += recive_size;
            }

            std::cout << rank << " ---\n";
            for (int x = 0; x < global_args.n; x++) {
                    int n = global_args.m;
                    for (int y = 0; y < global_args.m; y++) {
                        std::cout << at(F_H_full, x, y) << ' ';
                    }
                    std::cout << std::endl;
                }
            std::cout << std::endl;
        }
    }

    if(rank == 0){
        std::cout << "Iterations: " << iterationsElapsed << std::endl;
        std::cout << "Error: " << error << std::endl;
    }

    if(F_H_full) free(F_H_full);
    if(error_array) free(error_array);
    cudaFree(F_D);
    cudaFree(Fnew_D);
    cudaFree(F_H);

    MPI_Finalize();
    return 0;
}
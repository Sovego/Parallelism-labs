
#include <mpi.h>

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#ifdef DEBUG
#define DEBUG_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG_PRINTF(line, a...) 0
#endif

#ifdef DEBUG1
#define DEBUG1_PRINTF(line, a...) printf(line, ## a)
#else
#define DEBUG1_PRINTF(line, a...) 0
#endif

#define at(arr, x, y) (arr[(x) * (n) + (y)])
/*
* Режимы распределения потоков по блокам
*/
#ifdef THREAD_BIG_GRID
constexpr int MAXIMUM_THREADS_PER_BLOCK = 1024;
#define THREAD_PER_BLOCK_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, max_thread) (max_thread)
#define BLOCK_COUNT_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, threads_count) (arr_lenght_dim_2/threads_count.x?arr_lenght_dim_2/threads_count.x:1), arr_lenght_dim_1
#else
constexpr int MAXIMUM_THREADS_PER_BLOCK = 32;
#define THREAD_PER_BLOCK_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, max_thread) ((arr_lenght_dim_1+max_thread-1)/max_thread), ((arr_lenght_dim_2+max_thread-1)/max_thread)
#define BLOCK_COUNT_DEFINED(arr_lenght_dim_1, arr_lenght_dim_2, threads_count) ((arr_lenght_dim_1 + threads_count.x - 1) / threads_count.x), ((arr_lenght_dim_2 + threads_count.y - 1) / threads_count.y)
#endif


// Cornerns
constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

constexpr int THREADS_PER_BLOCK_REDUCE = 256;

constexpr int ITERS_BETWEEN_UPDATE = 400;

void transfer_data(const int rank, const int ranks_count, double* F_from, double* F_to, int m,int n, cudaStream_t stream = 0){

    if(rank != 0){
        cudaMemcpyAsync(F_from + m, F_to + m, m * sizeof(double), cudaMemcpyDeviceToHost, stream);
        MPI_Request rq;
        // отправляем указатель на вторую строку процессу, работующему сверху
        MPI_Isend(
            F_from + m,
            m,
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
        cudaMemcpyAsync(F_from + m*(n-2), F_to + m*(n-2), m * sizeof(double), cudaMemcpyDeviceToHost, stream);
        MPI_Isend(
            F_from + m*(n-2),
            m,
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
        MPI_Recv(F_from, m, MPI_DOUBLE, rank-1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(F_to, F_from, m * sizeof(double), cudaMemcpyHostToDevice, stream);

    }
    // принимаем строку от нижнего
    if(rank != ranks_count - 1){
        MPI_Status status;
        MPI_Recv(F_from+(m * (n-1)), m, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(F_to+(m * (n-1)), F_from+(m * (n-1)), m * sizeof(double), cudaMemcpyHostToDevice, stream);
    }
}


__global__ void iterate(const double *F, double *Fnew,const int* m){

#ifdef THREAD_BIG_GRID
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
#else
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    if (j == 0 || i == 0 || i >= *m - 1 || j >= *m - 1) return; // Don't update borders

    int n = *m;
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
    //cmdArgs global_args = cmdArgs{false, false, 1E-6, (int)1E6, 16, 16};
    //processArgs(argc, argv, &global_args);
    int n_g {std::stoi(argv[4])},m_g {std::stoi(argv[4])}; // n m - grid size
    double error_g {1},tol_g {std::stod(argv[2])}; // error - error value. tol - accuracy
    int iter_g {0},iter_max_g {std::stoi(argv[6])}; // iter - iterator. iter_max - max iteration count
    int n_l,m_l;
    if(rank == 0){
        std::cout << "-----------------------------" << std::endl;
    std::cout << "- Accuracy: " << tol_g << std::endl << "- Max iteration count: "<< iter_max_g << std::endl << "- Grid size: " << n_g << std::endl;
    std::cout << "-----------------------------" << std::endl;
    }

    // ====== Расчет элементов для каждого процесса ======
    int TOTAL_GRID_SIZE = m_g * n_g;

    //cmdArgs local_args{global_args};
    n_l =  TOTAL_GRID_SIZE / ranks_count / m_g + 2 * (rank != ranks_count - 1);
    m_l=m_g;
    int ELEMENTS_BY_PROCESS = n_l * m_l;

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
    int n = n_g;
    int m = m_g;

    // Заполняем полный массив в 0 процессе 
    if(rank == 0){
        
        F_H_full = (double*)calloc(n*m, sizeof(double));
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
            data_lenght = ELEMENTS_BY_PROCESS - 2 * m_l * (target == (ranks_count - 1) && ranks_count != 1);
            DEBUG_PRINTF("Sended to %d elems: %d from: %d\n", target, data_lenght, data_start);
            MPI_Isend(
                F_H_full + data_start,
                data_lenght,
                MPI_DOUBLE,
                target,
                0,
                MPI_COMM_WORLD,
                &req
            );
            
            data_start += data_lenght - m_l * 2;
            
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

    //cmdArgs *args_d;
    //double* error_d;
    int* iterations_d;
    int* iter_max_d;
    double* tol_d;
    //cudaMalloc(&error_d, sizeof(double));
    int * n_d;
    int* m_d;
    cudaMalloc(&n_d, sizeof(int));
    cudaMalloc(&m_d, sizeof(int));
    cudaMalloc(&iterations_d, sizeof(int));
    cudaMalloc(&iter_max_d,sizeof(int));
    cudaMalloc(&tol_d, sizeof(double));
    cudaMemcpy(iter_max_d, &iter_max_g, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tol_d, &tol_g, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(n_d, &n_g, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, &m_g, sizeof(double), cudaMemcpyHostToDevice);
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

    dim3 threadPerBlock {THREAD_PER_BLOCK_DEFINED(n_l, m_l, MAXIMUM_THREADS_PER_BLOCK)}; // ПЕРЕОСМЫСЛИТЬ
    if(threadPerBlock.x > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.x = MAXIMUM_THREADS_PER_BLOCK;
    }
    if(threadPerBlock.y > MAXIMUM_THREADS_PER_BLOCK){
        threadPerBlock.y = MAXIMUM_THREADS_PER_BLOCK;
    } 
    dim3 blocksPerGrid {BLOCK_COUNT_DEFINED(n_l, m_l, threadPerBlock)}; // ПЕРЕОСМЫСЛИТЬ

    DEBUG_PRINTF("%d: %d %d %d %d\n", rank, threadPerBlock.x, threadPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

    MPI_Barrier(MPI_COMM_WORLD);
    
#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif
    do {

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(F_D, Fnew_D, m_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data(rank, ranks_count, F_H, F_D, m_l,n_l, stream);

        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(Fnew_D, F_D, m_d);

        // ОБМЕН ГРАНИЧНЫМИ УСЛОВИЯМИ
        transfer_data(rank, ranks_count, F_H, Fnew_D, m_l,n_l, stream);
        
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
                    DEBUG1_PRINTF("iters: %d error: %lf\n", iterationsElapsed, error);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    } while(error > tol_g && iterationsElapsed < iter_max_g);
#ifdef NVPROF_
    nvtxRangePop();
#endif
    cudaStreamDestroy(stream);
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
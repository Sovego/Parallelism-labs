#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <random>
#include <stdexcept>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include <fstream>
#include <memory>
#include <unordered_map>
#ifdef NVPROF_
#include <nvToolsExt.h>
#endif
enum class PARAMETR_SOURCE{
    FILE, GENERATE
};

enum class ARRAY_SOURCE{
    DEVICE, HOST
};

using calc_type = float;

#pragma once

template<typename data_type>
class MDT_array{
public:

    MDT_array();
    MDT_array(int, data_type*, ARRAY_SOURCE src_type = ARRAY_SOURCE::HOST);
    MDT_array(int);
    ~MDT_array();

    __device__ data_type& operator[] (int index); // Получение I-ого элемента на видеокарте
    data_type operator() (int index); // Получение I-ого элемента и копирование его сразу на хост
    data_type* operator&() const; // Отдает указатель на массив на видеокарта
    int get_size();
private:
    data_type* array_;
    int size_;
};

template<typename data_type>
MDT_array<data_type>::MDT_array(int size, data_type* src, ARRAY_SOURCE src_type) : size_(size){
    cudaMalloc(&array_, size_*sizeof(data_type));
    if(src_type == ARRAY_SOURCE::DEVICE){
        cudaMemcpy(array_, src, sizeof(data_type) * size, cudaMemcpyDeviceToDevice);
    }
    else if(src_type == ARRAY_SOURCE::HOST){
        cudaMemcpy(array_, src, sizeof(data_type) * size, cudaMemcpyHostToDevice);
    }
}

template<typename data_type>
MDT_array<data_type>::MDT_array(){
     throw std::runtime_error("You should specifed size");
}

template<typename data_type>
MDT_array<data_type>::MDT_array(int size) : size_(size){
    cudaMalloc(&array_, size_*sizeof(data_type));
}

template<typename data_type>
MDT_array<data_type>::~MDT_array(){
    cudaFree(array_);
}

template<typename data_type>
__device__ data_type& MDT_array<data_type>::operator[](int index){
    return array_[index];
}

template<typename data_type>
data_type* MDT_array<data_type>::operator&() const{
    return array_;
}

template<typename data_type>
data_type MDT_array<data_type>::operator()(int index){
    data_type tmp;
    cudaMemcpy(&tmp, array_ + index, sizeof(data_type), cudaMemcpyDeviceToHost);
    return tmp;
}

template<typename data_type>
int MDT_array<data_type>::get_size(){
    return size_;
}

class Layer {
public:
    virtual MDT_array<calc_type>& forward(MDT_array<calc_type>& input) = 0;
    MDT_array<calc_type>& operator()(MDT_array<calc_type>& input){
        return forward(input);
    }
};

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(int input_size, int output_size, std::string weigth_path = "", std::string bias_path = "");

    void set_weights(const PARAMETR_SOURCE source_type, const std::string file_path = "");
    void set_bias(const PARAMETR_SOURCE source_type, const std::string file_path = "");

    ~FullyConnectedLayer();

    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;

    int get_size();
    std::pair<int, int> get_dims();

private:
    void generate_parmeter(calc_type* target, const std::pair<int, int> size, const int random_seed = -1);
    void load_parameter(calc_type* target, const std::string file_path, const std::pair<int, int> size);

    int input_size_;
    int output_size_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    MDT_array<calc_type>* d_output_;
    calc_type* d_weight_;
    calc_type* d_bias_;
    cublasHandle_t handle_;
};

using FC = FullyConnectedLayer;

class Sigmoid : public Layer {
    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;
};

MDT_array<calc_type>& Sigmoid_F(MDT_array<calc_type>& arr);

class Sequential : public Layer {
public:

    Sequential();

    void add_layer(std::unique_ptr<Layer> layer);

    MDT_array<calc_type>& forward(MDT_array<calc_type>& input) override;

private:
    std::vector<std::unique_ptr<Layer>> layers;
};



template <typename T, typename... Args>
std::unique_ptr<T> create_layer(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

static void load_binary_file(calc_type* target, const std::string file_path, const std::pair<int, int> size){
    std::ifstream file(file_path, std::ios::binary);

    calc_type* file_info;
    cudaMallocHost(&file_info, size.first*size.second*sizeof(calc_type));
    std::ifstream file_input(file_path, std::ios::binary);
    for(size_t i = 0; i < size.first*size.second; file >> file_info[i++]);

    calc_type* arr_host;
    cudaMallocHost(&arr_host, size.first*size.second*sizeof(calc_type));
    for (int i = 0; i < size.first; ++i) {
        for (int j = 0; j < size.second; ++j) {
            arr_host[i * size.second + j] = file_info[i * size.second + j];
        }
    }

    cudaMemcpy(target, arr_host, size.first*size.second*sizeof(calc_type), cudaMemcpyHostToDevice);
    
    cudaFreeHost(file_info);
    cudaFreeHost(arr_host);
}
Sequential::Sequential(){}

void Sequential::add_layer(std::unique_ptr<Layer> layer){
    layers.push_back(std::move(layer));
}

MDT_array<calc_type>& Sequential::forward(MDT_array<calc_type>& x){
    for(auto& lr : layers){
        x = lr->forward(x);
    }
    return x;
}
__global__ void sigmoid_cuda(calc_type* arr, int n){
    if(threadIdx.x < n){
        arr[threadIdx.x] = 1 / (1 + exp(-arr[threadIdx.x]));
    }
}

MDT_array<calc_type>& Sigmoid::forward(MDT_array<calc_type>& arr) {
    sigmoid_cuda<<<1, arr.get_size()>>>(&arr, arr.get_size());
    return arr;
}

MDT_array<calc_type>& Sigmoid_F(MDT_array<calc_type>& arr){
    return Sigmoid()(arr);
}


FullyConnectedLayer::FullyConnectedLayer(){
    throw std::runtime_error("You should specifed in_size and out_size");
}

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, std::string weigth_path, std::string bias_path) : input_size_(input_size), output_size_(output_size){
    d_output_ = new MDT_array<calc_type>(output_size);
    cudaMalloc(&d_weight_, input_size_ * output_size_ * sizeof(calc_type));
    cudaMalloc(&d_bias_, output_size_ * sizeof(calc_type));
    cublasCreate(&handle_);

    if(weigth_path == ""){
        set_weights(PARAMETR_SOURCE::GENERATE);
    }
    else{
        set_weights(PARAMETR_SOURCE::FILE, weigth_path);
    }
    if(bias_path == ""){
        set_bias(PARAMETR_SOURCE::GENERATE);
    }
    else{
        set_bias(PARAMETR_SOURCE::FILE, bias_path);
    }

}

FullyConnectedLayer::~FullyConnectedLayer() {
    cublasDestroy(handle_);
    d_output_->~MDT_array();
    free(d_output_);
    cudaFree(d_weight_);
    cudaFree(d_bias_);
}

MDT_array<calc_type>& FullyConnectedLayer::forward(MDT_array<calc_type>& x) {
    calc_type* input = &x;
    cublasSgemv(handle_, CUBLAS_OP_T, input_size_, output_size_, &alpha_, d_weight_, input_size_, input, 1, &beta_, &*d_output_, 1);
    cublasSaxpy(handle_, output_size_, &alpha_, d_bias_, 1, &*d_output_, 1);
    return *d_output_;
}

int FullyConnectedLayer::get_size(){
    return output_size_ * input_size_;
}
std::pair<int, int> FullyConnectedLayer::get_dims(){
    return {input_size_, output_size_};
}

void FullyConnectedLayer::set_weights(const PARAMETR_SOURCE source_type, const std::string file_path){
    if(source_type == PARAMETR_SOURCE::FILE){
        load_parameter(d_weight_, file_path, {input_size_, output_size_});
    }
    else if(source_type == PARAMETR_SOURCE::GENERATE){
        generate_parmeter(d_weight_, {input_size_, output_size_});
    }
}

void FullyConnectedLayer::set_bias(const PARAMETR_SOURCE source_type, const std::string file_path){
    if(source_type == PARAMETR_SOURCE::FILE){
        load_parameter(d_bias_, file_path, {1, output_size_});
    }
    else if(source_type == PARAMETR_SOURCE::GENERATE){
        generate_parmeter(d_bias_, {1, output_size_});
    }
}

void FullyConnectedLayer::generate_parmeter(calc_type* target, const std::pair<int, int> size, const int random_seed){
    int in_size = size.first;
    int out_size = size.second;
    
    std::default_random_engine rangen;
    if(random_seed != -1){
        rangen.seed(random_seed);
    }

    calc_type disp = std::sqrt((double)2/out_size);
    calc_type mean = 0;

    std::normal_distribution<calc_type> distribution(mean, disp);

    if(target == nullptr){
        cudaMalloc(&target, in_size*out_size*sizeof(calc_type));
    }

    calc_type* temp = (calc_type*)malloc(in_size*out_size*sizeof(calc_type));

    for(size_t i = 0; i < in_size*out_size; i++){
        temp[i] = distribution(rangen);
    }

    cudaMemcpy(target, temp, in_size*out_size*sizeof(calc_type), cudaMemcpyHostToDevice);

    free(temp);
}

void FullyConnectedLayer::load_parameter(calc_type* target, const std::string file_path, const std::pair<int, int> size){
    load_binary_file(target, file_path, size);
}



int main(){
    
    Sequential Net;

    MDT_array<calc_type> a {1024};

    load_binary_file(&a, "torch/input", {1, 1024});
    
    Net.add_layer(create_layer<FC>(1024, 256, "torch/fc1_weights", "torch/fc1_bias"));
    Net.add_layer(create_layer<Sigmoid>());


    Net.add_layer(create_layer<FC>(256, 16, "torch/fc2_weights", "torch/fc2_bias"));
    Net.add_layer(create_layer<Sigmoid>());

    Net.add_layer(create_layer<FC>(16, 1, "torch/fc3_weights", "torch/fc3_bias"));
    Net.add_layer(create_layer<Sigmoid>());


#ifdef NVPROF_
    nvtxRangePush("Forward_pass");
#endif
    MDT_array<calc_type> b = Net.forward(a);
#ifdef NVPROF_
    nvtxRangePop();
#endif

    std::cout << b(0) << std::endl;

    return 0;
}
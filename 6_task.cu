#define IDX2C(i,j,ld) (((j)*(ld))+(i))



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

enum class ARRAY_SOURCE{
    DEVICE, HOST
};

using calc_type = float;




class MDT_array{
public:

    MDT_array();
    MDT_array(int, float*, ARRAY_SOURCE src_type = ARRAY_SOURCE::HOST);
    MDT_array(int);
    ~MDT_array();

    __device__ float& operator[] (int index); // Получение I-ого элемента на видеокарте
    float operator() (int index); // Получение I-ого элемента и копирование его сразу на хост
    float* operator&() const; // Отдает указатель на массив на видеокарта
    int get_size() const;
private:
    float* array_{};
    int size_;
};


MDT_array::MDT_array(int size, float* src, ARRAY_SOURCE src_type) : size_(size){
    cudaMalloc(&array_, size_*sizeof(float));
    if(src_type == ARRAY_SOURCE::DEVICE){
        cudaMemcpy(array_, src, sizeof(float) * size, cudaMemcpyDeviceToDevice);
    }
    else if(src_type == ARRAY_SOURCE::HOST){
        cudaMemcpy(array_, src, sizeof(float) * size, cudaMemcpyHostToDevice);
    }
}


MDT_array::MDT_array(){
     throw std::runtime_error("You should specifed size");
}


MDT_array::MDT_array(int size) : size_(size){
    cudaMalloc(&array_, size_*sizeof(float));
}


MDT_array::~MDT_array(){
    cudaFree(array_);
}


__device__ float& MDT_array::operator[](int index){
    return array_[index];
}


float* MDT_array::operator&() const{
    return array_;
}


float MDT_array::operator()(int index){
    float tmp;
    cudaMemcpy(&tmp, array_ + index, sizeof(float), cudaMemcpyDeviceToHost);
    return tmp;
}


int MDT_array::get_size() const{
    return size_;
}

class Layer {
public:
    virtual MDT_array& forward(MDT_array& input) = 0;
    MDT_array& operator()(MDT_array& input){
        return forward(input);
    }
};

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();

    [[maybe_unused]] FullyConnectedLayer(int input_size, int output_size, const std::string& weigth_path = "", const std::string& bias_path = "");

    void set_weights(const std::string& file_path = "");
    void set_bias(const std::string& file_path = "");

    ~FullyConnectedLayer();

    MDT_array& forward(MDT_array& x) override;

    [[nodiscard]] int get_size() const;
    std::pair<int, int> get_dims();

private:
    static  void load_parameter(calc_type* target, const std::string& file_path, std::pair<int, int> size);

    int input_size_;
    int output_size_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    MDT_array* d_output_;
    calc_type* d_weight_;
    calc_type* d_bias_;
    cublasHandle_t handle_;
};

using FC = FullyConnectedLayer;

class Sigmoid : public Layer {
    MDT_array& forward(MDT_array& arr) override;
};

MDT_array& Sigmoid_F(MDT_array& arr);

class Sequential : public Layer {
public:

    Sequential();

    void add_layer(std::unique_ptr<Layer> layer);

    MDT_array& forward(MDT_array& x) override;

private:
    std::vector<std::unique_ptr<Layer>> layers;
};



template <typename T, typename... Args>
std::unique_ptr<T> create_layer(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

static void load_binary_file(calc_type* target, const std::string& file_path, const std::pair<int, int> size){
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
Sequential::Sequential()= default;

void Sequential::add_layer(std::unique_ptr<Layer> layer){
    layers.push_back(std::move(layer));
}

MDT_array& Sequential::forward(MDT_array& x){
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

MDT_array& Sigmoid::forward(MDT_array& arr) {
    sigmoid_cuda<<<1, arr.get_size()>>>(&arr, arr.get_size());
    return arr;
}

MDT_array& Sigmoid_F(MDT_array& arr){
    return Sigmoid()(arr);
}


FullyConnectedLayer::FullyConnectedLayer(){
    throw std::runtime_error("You should specifed in_size and out_size");
}

[[maybe_unused]] FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, const std::string& weigth_path, const std::string& bias_path) : input_size_(input_size), output_size_(output_size){
    d_output_ = new MDT_array(output_size);
    cudaMalloc(&d_weight_, input_size_ * output_size_ * sizeof(calc_type));
    cudaMalloc(&d_bias_, output_size_ * sizeof(calc_type));
    cublasCreate(&handle_);
    set_weights(weigth_path);
    set_bias( bias_path);


}

FullyConnectedLayer::~FullyConnectedLayer() {
    cublasDestroy(handle_);
    d_output_->~MDT_array();
    free(d_output_);
    cudaFree(d_weight_);
    cudaFree(d_bias_);
}

MDT_array& FullyConnectedLayer::forward(MDT_array& x) {
    calc_type* input = &x;
    cublasSgemv(handle_, CUBLAS_OP_T, input_size_, output_size_, &alpha_, d_weight_, input_size_, input, 1, &beta_, &*d_output_, 1);
    cublasSaxpy(handle_, output_size_, &alpha_, d_bias_, 1, &*d_output_, 1);
    return *d_output_;
}

int FullyConnectedLayer::get_size() const{
    return output_size_ * input_size_;
}
std::pair<int, int> FullyConnectedLayer::get_dims(){
    return {input_size_, output_size_};
}

void FullyConnectedLayer::set_weights(const std::string& file_path){

        load_parameter(d_weight_, file_path, {input_size_, output_size_});

}

void FullyConnectedLayer::set_bias(const std::string& file_path){

        load_parameter(d_bias_, file_path, {1, output_size_});

}

void FullyConnectedLayer::load_parameter(calc_type* target, const std::string& file_path, const std::pair<int, int> size){
    load_binary_file(target, file_path, size);
}



int main(){

    Sequential Net;

    MDT_array a {1024};

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
    MDT_array b = Net.forward(a);
#ifdef NVPROF_
    nvtxRangePop();
#endif

    std::cout << b(0) << std::endl;

    return 0;
}
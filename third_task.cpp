#include <iostream>
#include <cmath>
#include <openacc.h>
#include <exception>
#include <ctype.h>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#define at(arr, x, y) (arr[(x)*n+(y)]) 
/// @brief This function checks that the value obtained is a number
/// @param s string with the value to be checked
/// @return bool value: 1 if it`s number 0 if it`s anything else
bool isUint(const std::string& s){
    return s.find_first_not_of("0123456789") == std::string::npos;
}
/// @brief Сommand line help
void commandHelp()
{
    std::cout << "-a <Accuracy> -s <Grid size> -m <Max iteration count>" << std::endl;
}
constexpr double negOne = -1;
/// @brief Main function. For command line arguments help use -h
/// @param argc argument count
/// @param argv argument array
/// @return exit code (default 0)
int main(int argc, char* argv[])
{
    // Arguments check Begin
    if (argv[1]=="-h") // check help argument
    {
        commandHelp();
        return 0;
    }
    auto begin = std::chrono::steady_clock::now(); // start time
    if (argc-1!=6) // check count of arguments
    {
        throw std::runtime_error("Argument count not enough. Use -h for help"); 
    }
    if (!isUint(argv[4]) || !isUint(argv[6])) // check type of arguments
    {
        throw std::runtime_error("Argument invalid. Use -h for help");
    }
    // Arguments check End

    // Init variables Begin
    int n {std::stoi(argv[4])},m {std::stoi(argv[4])}; // n m - grid size
    double error {1},tol {std::stod(argv[2])}; // error - error value. tol - accuracy
    int iter {0},iter_max {std::stoi(argv[6])}; // iter - iterator. iter_max - max iteration count
    double* A {new double[n*n]}; // A - initial grid
    double* Anew {new double[n*n]}; // Anew - calculated grid
    double* tmp = new double[n*m];
    
    int max_idx = 0;
    // Init variables End

    // Init information output
    std::cout << "-----------------------------" << std::endl;
    std::cout << "- Accuracy: " << tol << std::endl << "- Max iteration count: "<< iter_max << std::endl << "- Grid size: " << n << std::endl;
    std::cout << "-----------------------------" << std::endl;
    

    // Filling corners Begin
    at(A, 0, 0) = 10;
    at(A, 0, m-1) = 20;
    at(A, n-1, 0) = 20;
    at(A, n-1, m-1) = 30;
    at(Anew, 0, 0) = 10;
    at(Anew, 0, m-1) = 20;
    at(Anew, n-1, 0) = 20;
    at(Anew, n-1, m-1) = 30;
    // Filling corners End
    // Grid edge filling Begin
    for (int i{1};i<n-1;++i)
    {
        at(A,0,i) = (at(A,0,m-1)-at(A,0,0))/(m-1)*i+at(A,0,0);
        at(A,i,0) = (at(A,n-1,0)-at(A,0,0))/(n-1)*i+at(A,0,0);
        at(A,n-1,i) = (at(A,n-1,m-1)-at(A,n-1,0))/(m-1)*i+at(A,n-1,0);
        at(A,i,m-1) = (at(A,n-1,m-1)-at(A,0,m-1))/(m-1)*i+at(A,0,m-1);
        at(Anew,0,i) = (at(A,0,m-1)-at(A,0,0))/(m-1)*i+at(A,0,0);
        at(Anew,i,0) = (at(A,n-1,0)-at(A,0,0))/(n-1)*i+at(A,0,0);
        at(Anew,n-1,i) = (at(A,n-1,m-1)-at(A,n-1,0))/(m-1)*i+at(A,n-1,0);
        at(Anew,i,m-1) = (at(A,n-1,m-1)-at(A,0,m-1))/(m-1)*i+at(A,0,m-1);
    }
    std::cout << "Init array\n";
    std::cout << "-----------------------------" << std::endl;
    for( int i{0}; i < n; ++i)
        {
            for( int j{0}; j < m; ++j )
                {
                    std::cout<<at(A,i,j) << " ";
                }
            std::cout<< std::endl;
        }
    std::cout << "-----------------------------" << std::endl;
    //std::memcpy(Anew, A, sizeof(double)*(n)*(m));
    // Grid edge filling End
    acc_set_device_num(3,acc_device_default);
    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasStatus_t status;
    #pragma acc enter data copyin(A[0:n*n],Anew[0:n*n],tmp[0:n*n])
    #pragma acc data present(A,Anew)
    {
        // Main algorithm loop Begin
        while ( error > tol && iter < iter_max )
        {
           
            //#pragma acc update device(error) // Update variable on GPU

            // Value calculation loop Begin
            //#pragma acc kernels
            #pragma acc parallel loop collapse(2) present(Anew[:n*m], A[:n*m])
            for( int i{1}; i < n-1; ++i)
            {
                for( int j{1}; j < m-1; ++j )
                    {
                        at(Anew,i,j) = 0.25 * (at(A,i,j+1) + at(A,i,j-1)+ at(A,i-1,j) + at(A,i+1,j)); // Calculate new values
                        //error = fmax( error, fabs(at(Anew,i,j) - at(A,i,j))); // Find new error value
                    }
            }
            // Value calculation loop End
        
            // Array swap Begin
            double* buf = A;
            A = Anew;
            Anew = buf;
            // Array swap End
            #pragma acc data present(tmp[:n*m], Anew[:n*m], A[:n*m]) 
            {
                #pragma acc host_data use_device(Anew, A, tmp)
                {

                    status = cublasDcopy(handle, n*m, A, 1, tmp, 1);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "copy error" << std::endl, exit(30);

                    status = cublasDaxpy(handle, n*m, &negOne, Anew, 1, tmp, 1);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "sum error" << std::endl, exit(40);
                    
                    status = cublasIdamax(handle, n*m, tmp, 1, &max_idx);
                    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "abs max error" << std::endl, exit(41);
                }
            }

            #pragma acc update self(tmp[max_idx-1])
            error = fabs(tmp[max_idx-1]);
            iter++; // Increase iterator
            //#pragma acc update self(error) // update variable on host for `while` compare
        }
    }
    // Main algorithm loop End

    #pragma acc exit data delete(Anew[0:n*n],tmp[0:n*n]) copyout(A[0:n*n])

    // Output calculated information
    std::cout << "Iteration count: " << iter << " " << "Error value: "<< error << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Delete pointers Begin
    
    
    delete[] Anew;
    Anew = nullptr;
    // Delete pointers End

    auto end = std::chrono::steady_clock::now(); // Code end time
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); // Calculate execution time
    std::cout << "The time: " << elapsed_ms.count() << " ms\n"; // Output execution time
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Result array\n";
    std::cout << "-----------------------------" << std::endl;
    for( int i{1}; i < n-1; ++i)
        {
            for( int j{1}; j < m-1; ++j )
                {
                    std::cout<<at(A,i,j) << " ";
                }
            std::cout<< std::endl;
        }
    std::cout << "-----------------------------" << std::endl;
    delete[] A;
    A = nullptr;
    return 0;
}
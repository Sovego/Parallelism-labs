#include <iostream>
#include <cmath>
#include <openacc.h>
#include <exception>
#include <ctype.h>
#include <string>
#include <chrono>
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
    double** A {new double*[n]}; // A - initial grid
    double** Anew {new double*[n]}; // Anew - calculated grid
    for (int i{0};i<n;++i)
    {
        A[i] = new double[m];
    }
    
    for (int i{0};i<n;++i)
    {
        Anew[i] = new double[m];
    }
    // Init variables End

    // Init information output
    std::cout << "-----------------------------" << std::endl;
    std::cout << "- Accuracy: " << tol << std::endl << "- Max iteration count: "<< iter_max << std::endl << "- Grid size: " << n << std::endl;
    std::cout << "-----------------------------" << std::endl;
    #pragma acc enter data create(A[0:n][0:m],Anew[0:n][0:m],iter) copyin(n,m,error)

    // Filling corners Begin
    A[0][0]=10;
    A[n-1][0]=20;
    A[0][m-1]=20;
    A[n-1][m-1]=30;
    Anew[0][0]=10;
    Anew[n-1][0]=20;
    Anew[0][m-1]=20;
    Anew[n-1][m-1]=30;
    // Filling corners End

    #pragma acc parallel loop present(A[0:n][0:m]) independent

    // Grid edge filling Begin
        for (int i{1};i<n-1;++i)
        {
            A[0][i]=10+((A[0][m-1] - A[0][0])/ (n-1))*i;
            A[i][0]=10+((A[n-1][0] - A[0][0])/ (n-1))*i;
            A[n-1][i]=10+((A[n-1][0] - A[n-1][m-1])/ (n-1))*i;
            A[i][m-1]=10+((A[n-1][m-1] - A[n-1][0])/ (n-1))*i;
            Anew[0][i]=10+((Anew[0][m-1] - Anew[0][0])/ (n-1))*i;
            Anew[i][0]=10+((Anew[n-1][0] - Anew[0][0])/ (n-1))*i;
            Anew[n-1][i]=10+((Anew[n-1][0] - Anew[n-1][m-1])/ (n-1))*i;
            Anew[i][m-1]=10+((Anew[n-1][m-1] - Anew[n-1][0])/ (n-1))*i;
        }
    // Grid edge filling End

    #pragma acc data present(A,Anew)
    {
        // Main algorithm loop Begin
        while ( error > tol && iter < iter_max )
        {
            error = 0.0; // Error reset
            #pragma acc update device(error) // Update variable on GPU

            // Value calculation loop Begin
            #pragma acc parallel loop collapse(2) present(A[0:n][0:m],Anew[0:n][0:m],n,m,error) independent reduction(max:error)
            for( int i{1}; i < n-1; ++i)
            {
                for( int j{1}; j < m-1; ++j )
                    {
                        Anew[i][j] = 0.25 * ( A[i][j+1] + A[i][j-1]+ A[i-1][j] + A[i+1][j]); // Calculate new values
                        error = fmax( error, fabs(A[i][j] - Anew[i][j])); // Find new error value
                    }
            }
            // Value calculation loop End
            
            // Array swap Begin
            double** buf = A;
            A = Anew;
            Anew = buf;
            // Array swap End

            iter++; // Increase iterator
            #pragma acc update self(error) // update variable on host for `while` compare
        }
    }
    // Main algorithm loop End

    #pragma acc exit data delete(A[0:n][0:m],Anew[0:n][0:m],n,m) copyout(error)

    // Output calculated information
    std::cout << "Iteration count: " << iter << " " << "Error value: "<< error << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Delete pointers Begin
    delete[] A;
    A = nullptr;
    delete[] Anew;
    Anew = nullptr;
    // Delete pointers End

    auto end = std::chrono::steady_clock::now(); // Code end time
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); // Calculate execution time
    std::cout << "The time: " << elapsed_ms.count() << " ms\n"; // Output execution time
    return 0;
}
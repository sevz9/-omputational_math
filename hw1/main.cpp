using namespace std;

#include <chrono>
#include <iostream>
#include <vector>
#include "omp.h"

pair<int, double>  differential_equation_solver_sequential_version(vector<vector<double>> &f, int N, double eps){

    auto start = std::chrono::high_resolution_clock::now();
    double h = 1 / (N - 1);
    int iter_cnt = 0;
    double dmax = 10000;

    int k = 2;

    vector<vector<double>> u(N, vector<double>(N, 0));

    for(int i = 0; i < N; i++){
        u[0][i] = f[0][i];
        u[N - 1][i] = f[N - 1][i];
        u[i][0] = f[i][0];
        u[i][N - 1] = f[i][N - 1];
    }
        

    while(dmax > eps){
        dmax = 0;
        for(int i = 1; i < N - 1; i++){
            for(int j= 1; j < N - 1; j++){
                double tmp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h + f[i][j]);
                dmax = max(dmax, abs(tmp - u[i][j]));
            }
        }
        iter_cnt++;
        

    } 
    auto end = std::chrono::high_resolution_clock::now();
    return {iter_cnt, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()};
}

pair<int, double>  differential_equation_solver_parallel_version_bad(vector<vector<double>> &f, int N, double eps){
  
    auto start = std::chrono::high_resolution_clock::now();
    double h = 1 / (N - 1);
    int iter_cnt = 0;
    double dmax = 10000;

    omp_lock_t dmax_lock; 
    omp_init_lock (&dmax_lock);

    vector<vector<double>> u(N, vector<double>(N, 0));

    for(int i = 0; i < N; i++){
        u[0][i] = f[0][i];
        u[N - 1][i] = f[N - 1][i];
        u[i][0] = f[i][0];
        u[i][N - 1] = f[i][N - 1];
    }
   

    while(dmax > eps){
        dmax = 0;
        int i = 1, j = 1;
        double tmp;
        #pragma omp parallel for shared(u,N,dmax) private(i,j,tmp) collapse(2)
        for(i = 1; i < N - 1; i++){
            for(j = 1; j < N - 1; j++){
                tmp = u[i][j];
                
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h + f[i][j]);
                
                omp_set_lock(&dmax_lock);
                dmax = max(dmax, abs(tmp - u[i][j]));
                omp_unset_lock(&dmax_lock);
            }
        }
         
        iter_cnt++;
    } 
    auto end = std::chrono::high_resolution_clock::now();
  
    return {iter_cnt, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()};
}

double process_block(int i, int j, int N, int block_number, vector<vector<double>> &f, vector<vector<double>> &u){
    
    double h = 1 / (N - 1);
    int block_size = (N - 2) / block_number;
    int last_block_size = (N - 2)  - block_size * (block_number - 1);
    
    int i_BS = (i == block_number - 1 ? last_block_size : block_size);
    int j_BS = (j == block_number - 1 ? last_block_size : block_size);
    
    int i_B_start = i * block_size + 1;
    int j_B_start = j * block_size + 1;

    int i_B_end = i_B_start + i_BS;
    int j_B_end = j_B_start + j_BS;
    double dmax = 0;
  
    for(int i = i_B_start; i < i_B_end; i++){
        for(int j= j_B_start; j < j_B_end; j++){
            double tmp = u[i][j];
         

            u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h + f[i][j]);
            dmax = max(dmax, abs(tmp - u[i][j]));
        }
        
    }
    return dmax;
}

pair<int, double>  differential_equation_solver_parallel_version_good(vector<vector<double>> &f, int N, int block_number, double eps){
    auto start = std::chrono::high_resolution_clock::now();

    int iter_cnt = 0;
    double dmax = 10000;

    vector<vector<double>> u(N, vector<double>(N, 0));

    for(int i = 0; i < N; i++){
        u[0][i] = f[0][i];
        u[N - 1][i] = f[N - 1][i];
        u[i][0] = f[i][0];
        u[i][N - 1] = f[i][N - 1];
    }
   

    while(dmax > eps){
        int i = 1, j = 1, nx = 1;
        double d;
        vector<double> dm(block_number, 0);
        dmax = 0;
        
        for(nx = 1; nx < block_number + 1; nx++){
          
            #pragma omp parallel for shared(nx)
            for(i = 0; i < nx; i++){
                j = nx - i - 1;
                
                d = process_block(i, j, N, block_number, f, u);
                
                dm[i] = max(dm[i], d);
                
            }
      
        }
        
        for(nx = block_number - 1; nx > 0; nx--){
            #pragma omp parallel for shared(nx)
            for(i = block_number - nx; i < block_number; i++){
                j = 2*block_number - nx - 1 - i;
              
                d = process_block(i, j, N, block_number, f, u);
             
                dm[i] = max(dm[i], d);
            }
        }
     
        for(int i = 0; i < block_number; i++)
            dmax = max(dmax, dm[i]);

        iter_cnt++;
       
    
    } 
    auto end = std::chrono::high_resolution_clock::now();
  
    return {iter_cnt, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()};
}



int main(){
    double eps = 1e-3;
    int block_number = 5;
    omp_set_num_threads(8);
    

    for(int N = 100; N < 600; N += 100){
        vector<vector<double>> f(N, vector<double>(N, 0));

        for(int i = 0; i < N; i++){
            f[0][i] = 100 - 200 * i;
            f[N - 1][i] = -100 + 200 * i;
            f[i][0] = 100 - 200 * i;
            f[i][N - 1] = -100 + 200 * i;
        }

        pair<int, double> des_res = differential_equation_solver_sequential_version(f, N, eps);

        pair<int, double> dep_bad_res = differential_equation_solver_parallel_version_bad(f, N, eps);

        pair<int, double> dep_good_res = differential_equation_solver_parallel_version_good(f, N, block_number, eps);

        cout << 
        "Размер сетки: " << N <<
        " Количессвто итераций: " << des_res.first <<
        " Время 11.1:  " << des_res.second <<
        " Количессвто итераций: " << dep_bad_res.first <<
        " Время 11.2:  " << dep_bad_res.second << 
        " Количессвто итераций: " << dep_good_res.first <<
        " Время 11.6:  " << dep_good_res.second << endl;
    }

}
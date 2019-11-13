#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <fstream>

using namespace std;

const unsigned long MASK = 0xFFFF000000000000;

__host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b) {
    if      (a.x < b.x) return true;
    else if (a.x == b.x && (a.y&MASK) <= (b.y&MASK)) return true;
    else return false;
}

typedef struct
{
    ulong2 key;
    unsigned long value;
} mystruct1;

__host__ __device__ bool operator<(const mystruct1 &a, const mystruct1 &b){
    if      (a.key.x < b.key.x) return true;
    else if (a.key.x == b.key.x && a.key.y <= b.key.y) return true;
    else return false;
}

void sort(mystruct1 *H, long int len)
{
    thrust::host_vector<mystruct1> H_vec(H, H+len);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalseconds = 0;

    thrust::device_vector<mystruct1> D_vec = H_vec;
    int iterations = 3;
    for(int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start, 0);

        thrust::sort(D_vec.begin(), D_vec.end());

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalseconds = totalseconds + milliseconds;
        if (i == iterations - 1) break;
        D_vec = H_vec;
    }
        
    printf("Elapsed time: %f s.\n\n", totalseconds/(iterations*1000));
    H_vec = D_vec;
    for(int i = 0; i < 32; i++)
    {
        cout << H_vec[i].key.x << " ";
    }
    cout << endl;

/*
    thrust::sort(H, H + len);
    for(int i = 0; i < 32; i++)
    {
        cout << H[i] << " ";
    }
    cout << endl;
*/
}

int main(void)
{
    //long int len = 1024*1024*1024;
    //unsigned long *H = (unsigned long *)malloc(len*sizeof(unsigned long));
    //char filename[50] = "ascii_1g_64.out";

/*    const int N = 10;
    long double keys[N] = {1, 4, 4, 5, 4, 5, 3, 1, 4, 5};
    char values[N] = {'1', '1', '2', '1', '3', '2', '1', '2', '4', '3'};
    cout << sizeof(long double) << endl;
    for (int i = 0; i < N; i++)
    {
        cout << keys[i] << "->" << values[i] << endl;
    }
    cout << "After:" << endl;
    thrust::sort_by_key(keys, keys + N, values);
    for (int i = 0; i < N; i++)
    {
        cout << keys[i] << "->" << values[i] << endl;
    }    
*/
    uint64_t len = 512L*1024*1024;
    mystruct1 *H = (mystruct1 *)malloc(len*sizeof(mystruct1));
 
    for (uint64_t i = 0; i < len; i++)
    {
        H[i].key.x = (unsigned long)rand();
        H[i].key.y = (unsigned long)rand();
        H[i].value = (unsigned long)rand();
    } 

    sort(H, len);

/*    thrust::host_vector<unsigned long> h_vec(512*1024*1024);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    thrust::device_vector<unsigned long> d_vec = h_vec;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalseconds = 0;
    
    for(int i = 0; i < 50; i++)
    {
        cudaEventRecord(start, 0);
        thrust::sort(d_vec.begin(), d_vec.end());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalseconds = totalseconds + milliseconds;
        d_vec = h_vec;
    }
    printf("Elapsed time: %f s.", totalseconds/50000);
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    for(int i = 0; i < 32; i++)
    {
        cout << h_vec[i] << " ";
    }
    cout << endl;
*/    
    return 0;
}

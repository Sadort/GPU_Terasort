#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <fstream>
#include "power.h"

using namespace std;

const unsigned long MASK = 0xFFFF000000000000;

void PowerStart(GPUPower* objpower[], int num)
{
    for (int i = 0; i < num; i++)
    {
        objpower[i]->startPowerThread( );
        objpower[i]->setStart(true) ;
    }
}

void PowerStop(GPUPower* objpower[], int num)
{
    for (int i = 0; i < num; i++)
    {
        objpower[i]->setStop(true) ;
        objpower[i]->stopPowerThread();
    }
}

__host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b) {
    if      (a.x < b.x) return true;
    //else if (a.x == b.x && (a.y&MASK) <= (b.y&MASK)) return true;
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

void sort(ulong2 *H, long int len)
{
    thrust::host_vector<ulong2> H_vec(H, H+len);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalseconds = 0;

    thrust::device_vector<ulong2> D_vec = H_vec;
    int iterations = 1;

    int num_threads = 1;
    printf("number of threads: %d\n", num_threads);
    GPUPower* powerObj[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        powerObj[i] = new GPUPower(0, i);
    }
    PowerStart(powerObj, num_threads);
/*
    for(int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start, 0);
*/
        thrust::sort(D_vec.begin(), D_vec.end());
    PowerStop(powerObj, num_threads); //comment
    system("cat powerprofile* > powerresults.txt");
    system("rm -rf powerprofile*");    
/*
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
    uint64_t len = 256L*1024*1024;
    ulong2 *H = (ulong2 *)malloc(len*sizeof(ulong2));
 
    for (uint64_t i = 0; i < len; i++)
    {
        H[i].x = (unsigned long)rand();
        H[i].y = (unsigned long)rand();
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

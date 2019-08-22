#include <thrust/sort.h>
#include <thrust/device_ptr.h>

const unsigned long MASK = 0xFFFF000000000000;

struct sort_ulong2 {
    __host__ __device__
    bool operator()(const ulong2 &a, const ulong2 &b) const {
    if      (a.x < b.x) return true;
    else if (a.x == b.x && a.y < b.y) return true;
    //else if (a.x == b.x && (a.y&MASK) < (b.y&MASK)) return true;
    else return false;
    }
};

int main(void)
{
    //std::cout << sizeof(unsigned long int) << std::endl;
    //std::cout << sizeof(unsigned long) << std::endl;
    int number_of_elements = 256*1024*1024;
    ulong2 *h_array = (ulong2*)malloc(number_of_elements * sizeof(ulong2));
    // Define your host array
    for (int i = 0; i < number_of_elements; i++)
    {
        h_array[i].x = rand() ;
        h_array[i].y = rand() ;
    }
    
    ulong2 *d_array;
    cudaMallocHost( (void**)&d_array, number_of_elements * sizeof(ulong2) );      
    cudaMemcpy( d_array,
                h_array, 
                number_of_elements * sizeof(ulong2),
                cudaMemcpyHostToDevice );

    thrust::device_ptr<ulong2> th_array( d_array );
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalseconds = 0;
    for(int i = 0; i < 10; i++)
    {
        cudaEventRecord(start, 0);
        thrust::sort( th_array, 
                      th_array+number_of_elements , 
                      sort_ulong2() );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalseconds = totalseconds + milliseconds;
        if(i == 9) break;
        cudaMemcpy( d_array,
                h_array,
                number_of_elements * sizeof(ulong2),
                cudaMemcpyHostToDevice );
    }
    printf("Elapsed time: %f s.", totalseconds/10000);

    //thrust::copy(d_array.begin(), d_array.end(), h_array.begin()); 
    for (int i = 0; i < 16; i++)
    {
        std::cout << d_array[i].x << " " << (d_array[i].y & MASK) << std::endl;
    }
    
    return 0;
}

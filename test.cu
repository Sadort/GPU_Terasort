#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <fstream>

using namespace std;

const unsigned long MASK = 0x000000000000FFFF;

struct sort_ulong2 {
    __host__ __device__
    bool operator()(const ulong2 &a, const ulong2 &b) const {
    if      ((a.x&MASK) < (b.x&MASK)) return true;
    else if ((a.x&MASK) == (b.x&MASK) && a.y < b.y) return true;
    else return false;
    }
};

int main(void)
{
    unsigned long number_of_elements = 512*1024*1024;
    ulong2 *h_array = (ulong2*)malloc(number_of_elements * sizeof(ulong2));
    
    // Define your host array
    // Read from file
    //char filename[50] = "ascii_1g_64.out";  
    //ifstream in(filename);
    //if (in == NULL) {
    //    printf("Error Reading File\n");
    //    exit(0);
    //}
    //for (unsigned long int i = 0; i < number_of_elements; i++) {
    //    in >> h_array[i].x;
    //    in >> h_array[i].y;
    //}
    //in.close();
    
    for (unsigned long i = 0; i < number_of_elements; i++)
    {
        h_array[i].x = (unsigned long)rand() ;
        h_array[i].y = (unsigned long)rand() ;
    }

    //std::cout << sizeof(h_array[0].x) << " " << sizeof(h_array[0].y) << endl;
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
        thrust::sort( th_array, th_array+number_of_elements , sort_ulong2() );
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
    //for (unsigned long i = number_of_elements - 32; i < number_of_elements; i++)
    //{
    //    std::cout << (d_array[i].x & MASK) << " " << (d_array[i].y) << std::endl;
    //}
    
    return 0;
}

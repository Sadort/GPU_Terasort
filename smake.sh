rm -rf *.o

#test
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include test.cu
#/usr/local/cuda/bin/nvcc -o test.out test.o -lnvidia-ml -lcudart

#thrust only key
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include thrust_test.cu
/usr/local/cuda/bin/nvcc -o thrust_test.out thrust_test.o 

#thrust key-value
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include thrust_test_value.cu
#/usr/local/cuda/bin/nvcc -o thrust_test_value.out thrust_test_value.o

#cub
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include -I/home/jhuan308/cub example_device_radix_sort.cu -lcudart -O3 -std=c++11
#/usr/local/cuda/bin/nvcc -o example_device_radix_sort.out example_device_radix_sort.o

rm -rf *.o

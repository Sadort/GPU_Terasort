rm -rf *.o

#test
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include test.cu
/usr/local/cuda/bin/nvcc -o test.out test.o 

#thrust
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include thrust_test.cu
#/usr/local/cuda/bin/nvcc -o thrust_test.out thrust_test.o 

#cub
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include -I/home/jhuan308/cub cub_test.cu -lcudart -O3
#/usr/local/cuda/bin/nvcc -o cub_test.out cub_test.o

rm -rf *.o

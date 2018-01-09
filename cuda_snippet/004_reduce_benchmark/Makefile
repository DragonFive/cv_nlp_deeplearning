NVCC = nvcc
CFLAGS = -O3
INCLUDES = -I.

# 编译 utils 库
utils.o: utils.cc utils.h
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 编译主程序
reduce_v0: utils.o reduce_sum.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $^ -o bin/$@

reduce_v1: utils.o reduce_sum3.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $^ -o bin/$@

all: reduce_v0 reduce_v1

clean:
	rm -f bin/* *.o

.PHONY: all clean
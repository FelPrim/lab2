# Переменные
CC = gcc
CFLAGS = -Ofast -Wall -Wextra -march=native -mtune=native -flto=auto -fipa-pta
LDFLAGS = -lmatio -fopenmp -lz -lm
INCLUDE = -I/usr/include
LIBRARY = -L/usr/lib
TARGET = exe.exe
SOURCE = 9.c

THREAD_NUM ?= 12
CACHE ?= 512000
ITERATIONS ?= 100

DEFINES = -DMAXTHREADS=$(THREAD_NUM) -DCACHE=$(CACHE)

# Сборка с генерацией профиля
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBRARY) $^ -o $@ $(LDFLAGS) -fprofile-generate $(DEFINES) -fprefetch-loop-arrays
	./$(TARGET) NODEBUG $(THREAD_NUM) $(ITERATIONS)
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBRARY) $^ -o $@ $(LDFLAGS) -fprofile-use $(DEFINES) -fprefetch-loop-arrays
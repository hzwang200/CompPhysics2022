CC = g++
PYBIND = `python3 -m pybind11 --includes`

ijulia.so : ijulia.cc
	$(CC) $(PYBIND) -O3 -fopenmp -shared -std=c++11 -fPIC ijulia.cc -o ijulia.so

clean :
	rm ijulia.so

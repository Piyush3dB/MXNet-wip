
# use this to build for GPU support

# Clean stuff
rm -rf main.o app

# Copy files
cp ../../mxnet/lib/libmxnet.so .

# From cpp test
g++ -O3 -c main.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -c MXNetForwarder.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -o app main.o MXNetForwarder.o libmxnet.so -L../../OpenBLAS/installed/lib -lopenblas


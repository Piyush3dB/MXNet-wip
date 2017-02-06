
# use this to build for GPU support

# Clean stuff
rm -rf main.o app

# Copy files
cp ../../mxnet-head/amalgamation/../lib/libmxnet_predict.so .

# Copy Header files
cp ../../mxnet-head/include/mxnet/c_api.h .
cp ../../mxnet-head/include/mxnet/c_predict_api.h  .


# From cpp test
g++ -O3 -c main.cc -std=c++11 -I../../OpenBLAS/installed/include -I../../mxnet-head/include/mxnet
g++ -O3 -c MXNetForwarder.cc -std=c++11 -I../../OpenBLAS/installed/include -I../../mxnet-head/include/mxnet -I./
g++ -O3 -o app main.o MXNetForwarder.o libmxnet_predict.so -L../../OpenBLAS/installed/lib -lopenblas -I../../mxnet-head/include/mxnet




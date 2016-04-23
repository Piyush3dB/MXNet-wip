
# Clean stuff
rm -rf mxnet_predict-all.o mxnet_predict-all.cc c_predict_api.h main.o app


# Copy files
cp ../../mxnet/amalgamation/mxnet_predict-all.cc .
cp ../../mxnet/include/mxnet/c_predict_api.h .

# From amalgamation
g++ -std=c++11 -I../../OpenBLAS/installed -fPIC -o mxnet_predict-all.o -c mxnet_predict-all.cc

# From cpp test
g++ -O3 -c main.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -c MXNetForwarder.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -o app main.o MXNetForwarder.o mxnet_predict-all.o -L../../OpenBLAS/installed/lib -lopenblas




# Clean stuff
rm -rf main.o app

# From cpp test
g++ -O3 -c main.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -c MXNetForwarder.cc -std=c++11 -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -o app main.o MXNetForwarder.o mxnet_predict-all.o -L../../OpenBLAS/installed/lib -lopenblas



# copy files
cp ../../mxnet/amalgamation/mxnet_predict-all.cc .

# From amalgamation
#g++ -std=c++11 -Wno-unknown-pragmas -Wall -I`pwd`/../../OpenBLAS/installed -MD -MF mxnet_predict0.d \
g++ -std=c++11 -Wno-unknown-pragmas -Wall -I`pwd`/../../OpenBLAS/installed -fPIC -o mxnet_predict-all.o -c mxnet_predict-all.cc
g++ -std=c++11 -Wno-unknown-pragmas -Wall -I`pwd`/../../OpenBLAS/installed -shared -o libmxnet_predict.so mxnet_predict-all.o -L`pwd`/../../OpenBLAS/installed -lopenblas -lrt



# From cpp test
g++ -O3 -c image-classification-predict.cc -std=c++11 -Wno-unknown-pragmas -Wall -I/home/piyush/Downloads/GitHub/OpenBLAS/installed/include -I`pwd`/../../mxnet/include 
g++ -O3 -o image-classification-predict image-classification-predict.o -L/home/piyush/Downloads/GitHub/OpenBLAS/installed/lib -lopenblas  `pwd`/../../mxnet/lib/libmxnet_predict.so



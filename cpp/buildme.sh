
# Clean stuff
rm -rf mxnet_predict-all.o mxnet_predict-all.cc c_predict_api.h libmxnet_predict.so image-classification-predict.o image-classification-predict


# Copy files
cp ../../mxnet/amalgamation/mxnet_predict-all.cc .
cp ../../mxnet/include/mxnet/c_predict_api.h .

# From amalgamation
#g++ -std=c++11 -Wno-unknown-pragmas -Wall -I../../OpenBLAS/installed -MD -MF mxnet_predict0.d \
g++ -std=c++11 -Wno-unknown-pragmas -Wall -I../../OpenBLAS/installed -fPIC -o mxnet_predict-all.o -c mxnet_predict-all.cc
g++ -std=c++11 -Wno-unknown-pragmas -Wall -I../../OpenBLAS/installed -shared -o libmxnet_predict.so mxnet_predict-all.o -L../../OpenBLAS/installed -lopenblas -lrt



# From cpp test
g++ -O3 -c image-classification-predict.cc -std=c++11 -Wno-unknown-pragmas -Wall -I../../OpenBLAS/installed/include -I./ 
g++ -O3 -o image-classification-predict image-classification-predict.o -L../../OpenBLAS/installed/lib -lopenblas libmxnet_predict.so



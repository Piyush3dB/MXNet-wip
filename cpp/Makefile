# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
#export OPENBLAS_ROOT=/home/piyush/Downloads/GitHub/OpenBLAS/installed

#CFLAGS+= -I${OPENBLAS_ROOT}/include
#CFLAGS+= -I/usr/local/include
#LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas 
#LDFLAGS=-L/usr/local/lib/

# Added for opencv
#CFLAGS+= `pkg-config --cflags opencv`
#LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
export MXNET_ROOT=`pwd`/../../mxnet

CFLAGS+= -I$(MXNET_ROOT)/include 
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so

image-classification-predict: image-classification-predict.o
	g++ -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	g++ -O3 -c image-classification-predict.cc ${CFLAGS}
	
clean: 
	rm image-classification-predict
	rm -f *.d *.o

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./

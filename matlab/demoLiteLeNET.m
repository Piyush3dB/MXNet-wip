format compact;
close all;
clear;
clc;

%% Add path and load library
if libisloaded('libmxnet')
    unloadlibrary('libmxnet');
end
MXNET_ROOT = [fileparts(mfilename('fullpath')), '/../../mxnet/'];
MXNET_LIB  = [MXNET_ROOT, '/lib'];
MXNET_SO   = [MXNET_ROOT, '/include/mxnet'];
MXNET_HDR  = 'c_predict_api.h';
addpath(MXNET_LIB);
addpath(MXNET_SO);
[err, warn] = loadlibrary('libmxnet.so', MXNET_HDR);
assert(isempty(err));


%% Load Symbol and Params
DATA_DIR = '../../MXNetModels/lenetMnistModel/mnist';
MNIST_DATA   = fullfile(DATA_DIR, 't10k-images-idx3-ubyte');
MNIST_LABELS = fullfile(DATA_DIR, 't10k-labels-idx1-ubyte');

symblFile = '../../MXNetModels/lenetMnistModel/lenet-symbol.json';
paramFile = '../../MXNetModels/lenetMnistModel/lenet-0010.params';

%% Init object
mxObj = MXNetForwarder(symblFile, paramFile);

%% Load and resize the image
[img, labels] = readMNIST(MNIST_DATA, MNIST_LABELS, 1, 5223);

%% Forward the image
mxObj = mxObj.forward(img);

%% Retrieve output
pred = mxObj.getOutput();

%% Free object
mxObj = mxObj.free();

%% find the predict label
for i = 0:9
    fprintf('%2d. Prob=%3f\n', i, pred(i+1));
end

figure(1);
subplot(2,1,1);
imshow(img);
subplot(2,1,2);
bar([0:9], pred, 'b')

format compact;
close all;
clear;
clc;

%% Load Symbol and Params
DATA_DIR = '../../MXNetModels/lenetMnistModel/mnist';
MNIST_DATA   = fullfile(DATA_DIR, 't10k-images-idx3-ubyte');
MNIST_LABELS = fullfile(DATA_DIR, 't10k-labels-idx1-ubyte');

%% Load and resize the image
[img, labels] = readMNIST(MNIST_DATA, MNIST_LABELS, 1, 0);

%% Forward the image
%% Forward the image
siz = size(img);
assert(length(siz) >= 2);

img = permute(img, [2 1 3:length(siz)]);

X_data  = single(img(:)); % take cols and concat
X_len   = uint32(numel(X_data));

%% Save X_data as binary file
fid = fopen('mnist_7.bin','w');
fwrite(fid, X_data, 'uint8');
fclose(fid);
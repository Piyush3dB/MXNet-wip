format compact;
close all;
clear;
clc;

%% Add path and load library
if libisloaded('libmxnet')
    unloadlibrary('libmxnet');
end
LABELS_FILE = '../../MXNetModels/cifar1000VGGmodel/synset.txt';
MXNET_ROOT = [fileparts(mfilename('fullpath')), '/../../mxnet/'];
MXNET_LIB  = [MXNET_ROOT, '/lib'];
MXNET_SO   = [MXNET_ROOT, '/include/mxnet'];
MXNET_HDR  = 'c_predict_api.h';
addpath(MXNET_LIB);
addpath(MXNET_SO);
[err, warn] = loadlibrary('libmxnet.so', MXNET_HDR);
assert(isempty(err));


%% Init model pointer
pCtx = libpointer('voidPtr', 0);
assert(pCtx.Value == 0);

%% Load Symbol and Params
symblFile = '../../MXNetModels/cifar1000VGGmodel/Inception_BN-symbol.json';
paramFile = '../../MXNetModels/cifar1000VGGmodel/Inception_BN-0039.params';
Symbol = fileread(symblFile);

fid = fopen(paramFile, 'rb');
Params = fread(fid, inf, '*ubit8');
fclose(fid);
ParamsLen = length(Params);
ParamsPtr = libpointer('voidPtr', Params);


%% Load and resize the image
img = imresize(imread('cat.png'), [224 224]);
img = single(img) - 120;
siz = size(img);
assert(length(siz) >= 2);
img = permute(img, [2 1 3:length(siz)]);

X_data  = single(img(:));
X_len   = uint32(numel(X_data));
X_dim   = int32([ones(1, 4-length(siz)), siz(end:-1:1)]);
X_shape = uint32([0, 4]);

%% Run prediction
fprintf('create predictor with input size ');
fprintf('%d ', siz);
fprintf('\n');

callmxnet('MXPredCreate', ...
    Symbol, ...
    ParamsPtr, ...
    ParamsLen, ...
    1, ...
    0, ...
    1, ...
    {'data'}, ...
    X_shape, ...
    X_dim, ...
    pCtx);

%% feed input
callmxnet('MXPredSetInput', pCtx, 'data', X_data, X_len);

%% forward
callmxnet('MXPredForward', pCtx);

%% Get the output size and allocate pointer
Y_dim   = libpointer('uint32Ptr', 0);
Y_shape = libpointer('uint32PtrPtr', zeros(4,1));
callmxnet('MXPredGetOutputShape', ...
    pCtx, ...
    0, ...
    Y_shape, ...
    Y_dim);
assert(Y_dim.Value <= 4);
Y_size = Y_shape.Value(1:Y_dim.Value);
Y_size = double(Y_size(end:-1:1))';

%% Get the output daya
Y_data = libpointer('singlePtr', single(zeros(Y_size)));

callmxnet('MXPredGetOutput', ...
    pCtx, ...
    0, ...
    Y_data, ...
    uint32(prod(Y_size)));

% TODO convert from c order to matlab order...
pred = reshape(Y_data.Value, Y_size);

%% Free the model
callmxnet('MXPredFree', pCtx);


%% load the labels
labels = {};
fid = fopen(LABELS_FILE, 'r');
assert(fid >= 0);
tline = fgetl(fid);
while ischar(tline)
    labels{end+1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

%% find the predict label
[pR, iD] = sort(pred, 'descend');
for i = 1:10
    fprintf('Prob=%3f %s\n', pR(i), labels{iD(i)});
end

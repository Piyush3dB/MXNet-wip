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


%% Load the model
pCtx = libpointer('voidPtr', 0);
assert(pCtx.Value == 0);

%% Load Symbol and params
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
inImg  = single(img(:));
lenImg = uint32(numel(inImg));


%% Run prediction
dev_type = int32(1); % cpu in default
dev_id   = int32(0);
fprintf('create predictor with input size ');
fprintf('%d ', siz);
fprintf('\n');
csize = int32([ones(1, 4-length(siz)), siz(end:-1:1)]);
callmxnet('MXPredCreate', ...
    Symbol, ...
    ParamsPtr, ...
    ParamsLen, ...
    dev_type, ...
    dev_id, ...
    1, ...
    {'data'}, ...
    uint32([0, 4]), ...
    csize, ...
    pCtx);

%% feed input
callmxnet('MXPredSetInput', pCtx, 'data', inImg, lenImg);

%% forward
callmxnet('MXPredForward', pCtx);

%% Get the output size and allocate pointer
out_dim   = libpointer('uint32Ptr', 0);
out_shape = libpointer('uint32PtrPtr', ones(4,1));
callmxnet('MXPredGetOutputShape', ...
    pCtx, ...
    0, ...
    out_shape, ...
    out_dim);
assert(out_dim.Value <= 4);
out_siz = out_shape.Value(1:out_dim.Value);
out_siz = double(out_siz(end:-1:1))';

%% Get the output daya
out = libpointer('singlePtr', single(zeros(out_siz)));

callmxnet('MXPredGetOutput', ...
    pCtx, ...
    0, ...
    out, ...
    uint32(prod(out_siz)));

% TODO convert from c order to matlab order...
pred = reshape(out.Value, out_siz);

%% Free the model
callmxnet('MXPredFree', pCtx);


%% load the labels
labels = {};
fid = fopen('../../MXNetModels/cifar1000VGGmodel/synset.txt', 'r');
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

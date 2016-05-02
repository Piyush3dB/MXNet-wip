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
symblFile = '/home/piyush/Downloads/GitHub/mxnet-face/model/lightened_cnn/lightened_cnn-symbol.json';
paramFile = '/home/piyush/Downloads/GitHub/mxnet-face/model/lightened_cnn/lightened_cnn-0166.params';

%% Init object
mxObj = MXNetForwarder(symblFile, paramFile);

%% Load and resize the image
%fl = '/home/piyush/Downloads/temp/lfw_funneled/Drew_Bledsoe/Drew_Bledsoe_0001.jpg'
fl = '/home/piyush/Downloads/temp/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0001.jpg';
img = imread(fl);
img = imresize(img(:,:,1), 128/250);
img = single(img);

%% Forward the image
mxObj = mxObj.forward(img);

%% Retrieve output
pred = mxObj.getOutput();

%% Free object
mxObj = mxObj.free();

find(pred == max(pred))
max(pred)
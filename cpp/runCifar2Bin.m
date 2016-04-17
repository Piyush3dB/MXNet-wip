format compact;
close all;
clear;
clc;

%% Convert a CIFAR image to binary

%% Load and resize the image
img = imread('cat_224x224x3.png');
img = single(img) - 0; % -120

%% Forward the image
siz = size(img);
assert(length(siz) >= 2);

img = permute(img, [2 1 3:length(siz)]);

X_data  = single(img(:)); % take cols and concat
X_len   = uint32(numel(X_data));

%save X_data as binary file
fid = fopen('cat_224x224x3.bin','w');
fwrite(fid, X_data, 'uint8');
fclose(fid);

disp('== DONE ==');

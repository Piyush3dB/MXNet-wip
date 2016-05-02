format compact;
close all;
clear;
clc;

%% Add path and load library

%% Load Symbol and Params
DATA_DIR = '../../MXNetModels/lenetMnistModel/mnist';
MNIST_DATA   = fullfile(DATA_DIR, 't10k-images-idx3-ubyte');
MNIST_LABELS = fullfile(DATA_DIR, 't10k-labels-idx1-ubyte');

%% Load and resize the image
[img, labels] = readMNIST(MNIST_DATA, MNIST_LABELS, 100, 700);

%save assets

for i = 0:9
    for j = 1:100
       if i == labels(j)
          figure;
          im = img(:,:,j);
          im = floor(im*255)
          im = im(1:2:end, 1:2:end);
          imshow(im);
          str = sprintf('i=%d, %d', i, labels(j) );
          disp(str);
          
          tow = floor(im(:)*255)
          
          %fid = fopen(['digit' num2str(i) '.bin'], 'wb');
          %fwrite(fid, im(:));
          %fclose(fid);
          
          
          break;
       end
    end
    
end
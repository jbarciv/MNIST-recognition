
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Master in Robotics
%                    Applied Artificial Intelligence
%
% Final project:  Visual Handwritten Digits Recognition
% Students:
%
%   - David Redondo Quintero (23147)
%
% First version: 02/05/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%%% Loading data %%%
load Trainnumbers.mat;
X = abalone_dataset;

%% Separate training set from test set

X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%
% X_train=double(X_train/255);
% X_test=double(X_test/255);

%%
hiddenSize = [25];
autoenc = trainAutoencoder(X_train,hiddenSize,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.15);


%%
XReconstructed_train = predict(autoenc,X_train);
mseError_train = mse(X_train-XReconstructed_train)


XReconstructed_test = predict(autoenc,X_test);
mseError_test = mse(X_test-XReconstructed_test)


%%
figure;
subplot(4,2,1);
imshow(print_digit(X_train,1),[0,255]);
subplot(4,2,2);
imshow(print_digit(XReconstructed_train,1),[0,255]);
subplot(4,2,3);
imshow(print_digit(X_train,2),[0,255]);
subplot(4,2,4);
imshow(print_digit(XReconstructed_train,2),[0,255]);
subplot(4,2,5);
imshow(print_digit(X_train,3),[0,255]);
subplot(4,2,6);
imshow(print_digit(XReconstructed_train,3),[0,255]);
subplot(4,2,7);
imshow(print_digit(X_train,4),[0,255]);
subplot(4,2,8);
imshow(print_digit(XReconstructed_train,4),[0,255]);

%%

figure;
subplot(4,2,1);
imshow(print_digit(X_test,1),[0,255]);
subplot(4,2,2);
imshow(print_digit(XReconstructed_test,1),[0,255]);
subplot(4,2,3);
imshow(print_digit(X_test,2),[0,255]);
subplot(4,2,4);
imshow(print_digit(XReconstructed_test,2),[0,255]);
subplot(4,2,5);
imshow(print_digit(X_test,3),[0,255]);
subplot(4,2,6);
imshow(print_digit(XReconstructed_test,3),[0,255]);
subplot(4,2,7);
imshow(print_digit(X_test,4),[0,255]);
subplot(4,2,8);
imshow(print_digit(XReconstructed_test,4),[0,255]);
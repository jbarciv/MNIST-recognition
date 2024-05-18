
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


%% Separate training set from test set

X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%
X_train=double(X_train/255);
X_test=double(X_test/255);

%%
hiddenSize = 25;
autoenc = trainAutoencoder(X_train,hiddenSize,...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.15);

XReconstructed = predict(autoenc,X_train);
mseError = mse(X_train-XReconstructed)


XReconstructed = predict(autoenc,X_test);
mseError = mse(X_test-XReconstructed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------- SOM CLASSIFIER ----------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Master in Robotics
%                    Applied Artificial Intelligence
%
% Final project:  Visual Handwritten Digits Recognition
% Students:
%   - Alberto Ibernon Jimenez (23079)
%   - David Redondo Quintero (23147)
%   - Josep Maria Barbera Civera (17048)
% First version: 29/04/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

%% Loading data %%
load Trainnumbers.mat;
load SOM_entrega.mat;

%% Inputs
name = {'Chema','David','Alberto'};
PCA  = 0;

debugging_flag = 0; % Flag to activate plots and debugging intermediate variables

%% Separate training set from test set
X_train = Trainnumbers.image(:,1:10000);
y_train = Trainnumbers.label(:,1:10000);
% X_test  = Trainnumbers.image(:,8001:10000);
% y_test  = Trainnumbers.label(:,8001:10000);


%% SOM


%% Prediction
% yntest=somnet(X_test);
% yntest_ind=vec2ind(yntest);
% prediction = predict_class(neuron_class, yntest_ind);

% Save result
class = prediction';
save('Group08_som.mat','name','PCA','class');
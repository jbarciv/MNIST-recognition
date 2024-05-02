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
hiddenSizes=[200,10];

net = feedforwardnet(hiddenSizes,'trainscg');
net.layers{2}.transferFcn = 'softmax';
net.trainParam.epochs = 1000; % Number of epochs
% net.trainParam.lr = 0.08; % Number of epochs
net.trainParam.goal = 0.0001; % Training goal
net.trainParam.max_fail = 100;
view(net)
[net, tr] = train(net, X_train, y_train);


%%
% Calculate output of the network for test set
y_prediction = net(X_test);


%%
% Calculate errors
testError = perform(net, y_test, y_prediction);
disp(['Test Error: ' num2str(testError)]);

%%
final = round(y_prediction);
no_errors_nn = sum(final ~= y_test);
accuracy = ((length(y_test) - no_errors_nn) / length(y_test)) * 100;
disp(['Misclassification error: ', num2str(no_errors_nn)]);
disp(['Accuracy: ', num2str(accuracy), '%']);





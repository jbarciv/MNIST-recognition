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
clf

%%% Loading data %%%
load Trainnumbers.mat;

%% Separate training set from test set

X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%
X_train = double(X_train/255);
X_test = double(X_test/255);

%%
% hiddenSizes=[20,10];
% numInputs = 784;
% 
% net = feedforwardnet(hiddenSizes,'trainscg');
% net.layers{end}.transferFcn = 'softmax';
% 
% net.trainParam.epochs = 1000; % Number of epochs
% % net.trainParam.lr = 0.08; % Number of epochs
% net.trainParam.goal = 0.0001; % Training goal
% net.trainParam.max_fail = 100;
% [net, tr] = train(net, X_train, y_train);
% 
% 
% %%
% % Calculate output of the network for test set
% y_prediction = net(X_test);
% 
% 
% %%
% % Calculate errors
% testError = perform(net, y_test, y_prediction);
% disp(['Test Error: ' num2str(testError)]);
% 
% %%
% final = round(y_prediction);
% no_errors_nn = sum(final ~= y_test);
% accuracy = ((length(y_test) - no_errors_nn) / length(y_test)) * 100;
% disp(['Misclassification error: ', num2str(no_errors_nn)]);
% disp(['Accuracy: ', num2str(accuracy), '%']);


%%
% % Assuming X_train is originally of shape [num_samples, 28, 28]
% X_train = reshape(X_train, [], 784); % Reshape to [num_samples, 784]
% 
% % Ensure y_train is a column vector
% y_train = y_train(:);

% num_classes = 10;
% layers = [
%     featureInputLayer(784) % Input layer for 784 features (28*28 pixels flattened)
%     % fullyConnectedLayer(50) % First fully connected layer with 50 neuron
%     % reluLayer
%     fullyConnectedLayer(50) % First fully connected layer with 50 neurons
%     sigmoidLayer % ReLU activation layer
%     % fullyConnectedLayer(10) % First fully connected layer with 50 neurons
%     % sigmoidLayer
%     % fullyConnectedLayer(10) % First fully connected layer with 50 neurons
%     % sigmoidLayer % ReLU activation layer
%     % fullyConnectedLayer(10) % First fully connected layer with 50 neurons
%     % reluLayer
%     fullyConnectedLayer(num_classes) % Fully connected layer with output neurons equal to num_classes
%     softmaxLayer % Softmax activation layer
%     classificationLayer]; % Classification layer
% 
% % Training options
% options = trainingOptions("adam", ...
%     'MaxEpochs', 5, ...
%     'InitialLearnRate', 5e-3, ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% % Training the network
% net = trainNetwork(X_train', categorical(y_train'), layers, options);
% %%
% 
% % Classify the test data
% y_pred = classify(net, X_test');
% 
% % Calculate the accuracy
% accuracy = sum(y_pred == categorical(y_test')) / numel(y_test');
% fprintf('Test accuracy: %.2f%%\n', accuracy * 100);











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------DEEP LEARNING CONVOLUTIONAL NEURAL NETWORKS (CNN) -----------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Master in Robotics
%                    Applied Artificial Intelligence
%
% Final project:  Visual Handwritten Digits Recognition
% Students:
%   - Alberto Ibernon Jimenez (23079)
%   - David Redondo Quintero (23147)
%   - Josep Maria Barbera Civera (17048)
% First version: 08/06/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
tic;

%% Loading data %%
load Trainnumbers.mat;
load Test_numbers_HW1.mat;

%% Inputs
name = {'Chema','David','Alberto'};
PCA  = 784;

debugging_flag = 0; % Flag to activate plots and debugging intermediate variables

%% Separate training set from test set
X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
% X_test  = Trainnumbers.image(:,8001:10000);
% y_test  = Trainnumbers.label(:,8001:10000);

X_test   = Test_numbers.image;

%% Reshaping
X_train_reshaped = reshape(X_train,[28 28 1 length(X_train(1,:))]);
X_test_reshaped  = reshape(X_test,[28 28 1 length(X_test(1,:))]);

%% Define DL parameters
layers = 15;
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)

    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',3,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress');
net = trainNetwork(X_train_reshaped,categorical(y_train),layers, options);

%% Trainning
dln_class_training  = classify(net,X_train_reshaped);
dln_class_training_double = double(dln_class_training) - ones(length(y_train),1);
errors_dln_training = length(find(dln_class_training_double'~=y_train));
dlnTrainingErrorPerc = 100 - 100*errors_dln_training/length(y_train);


%% Testing
dln_class_testing  = classify(net,X_test_reshaped);
dln_class_testing_double = double(dln_class_testing) - ones(length(dln_class_testing),1);

% % Performance
% errors_dln_testing = length(find(dln_class_testing_double'~=y_test));
% dlnTestErrorPerc = 100 - 100*errors_dln_testing/length(dln_class_testing);

% Confusion Chart
if debugging_flag == 1
    figure();
    cm = confusionchart(y_test',dln_class_testing_double');
    cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    title('DLN CNN Testing - Confusion Matrix')
end

% Save result
class = int8(dln_class_testing_double');
save('Group08_dln.mat','name','PCA','class');

% Print results
fprintf('********************************\n')
fprintf('Método de Clasificador por Deep Convolutional Neural Networking (CNN)\n')
fprintf('********************************\n')
fprintf('Porcentaje de Aciertos para el Training Dataset: %f %%\n', dlnTrainingErrorPerc)
% fprintf('Porcentaje de Aciertos para el Testing  Dataset: %f %%\n', dlnTestErrorPerc)
fprintf('Dimensión reducida por PCA: %d \n',PCA)
fprintf('Tiempo de Computación: %f s \n',toc)

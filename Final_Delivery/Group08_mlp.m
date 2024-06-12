%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Master in Robotics
%                    Applied Artificial Intelligence
%
% Final project:  Visual Handwritten Digits Recognition
% Students:
%
%   - David Redondo Quintero (23147)
%   - Josep M. Barbera Civera (17048)
%   - Alberto Ibernon Jimenez (23079)
%
% First version: 02/05/2024
% Second version: 11/06/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
tic;

training = false; %%%%% <- change here as needed! %%%%%
testing = true; %%%%% <- change here as needed! %%%%%

%% Inputs
name = {'Chema','David','Alberto'};
PCA  = int32(784);

% Load dataset for training
if training
    load Trainnumbers.mat;
    images_train = Trainnumbers.image(:,1:8000);
    labels_train = Trainnumbers.label(:,1:8000);
    images_test = Trainnumbers.image(:,8001:10000);
    labels_test = Trainnumbers.label(:,8001:10000);
    
    images_train = double(images_train/255);
    images_test = double(images_test/255);
    
    images = [images_train, images_test];
    labels = [labels_train, labels_test];
    labels = full(ind2vec(labels + 1));

    nn = patternnet([600 150 20], 'traincgf');
    nn.divideFcn = 'divideind';
    shuffled_indices = randperm(6000);
    nn.divideParam.trainInd = shuffled_indices;
    nn.divideParam.valInd = 6001:8000;
    nn.divideParam.testInd = 8001:10000;

    [nn, tr] = train(nn, images, labels);
    save(sprintf('nn.mat'), 'nn');
else 
    load nn.mat;
end

if testing
    load Test_numbers_HW1.mat;
    images_test   = Test_numbers.image;
    images_test = double(images_test/255);
    class = nn(images_test);
    class = vec2ind(class)-1;
end 



% Save result
class = int8(class);
save('Group08_mlp.mat','name','PCA','class');

% Print results
fprintf('********************************\n')
fprintf('Método de Clasificador por Multi-Layer Perceptron (MLP)\n')
fprintf('********************************\n')
fprintf('Porcentaje de Aciertos para el Training Dataset: %f %%\n', dlnTrainingErrorPerc)
fprintf('Dimensión reducida por PCA: %d \n',PCA)
fprintf('Tiempo de Computación: %f s \n',toc)


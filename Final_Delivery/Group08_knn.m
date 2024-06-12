%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------- KNN CLASSIFIER ----------------------------
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

%% Inputs
name = {'Chema','David','Alberto'};
PCA  = 40;

debugging_flag = 0; % Flag to activate plots and debugging intermediate variables

%% Separate training set from test set
X_train = Trainnumbers.image(:,1:10000);
y_train = Trainnumbers.label(:,1:10000);
% X_test  = Trainnumbers.image(:,8001:10000);
% y_test  = Trainnumbers.label(:,8001:10000);

%% Normalization
[D,N] = size(X_train); 
meanp = mean(X_train')';
stdp = std(X_train')';

for i = 1:D
    if stdp(i) == 0 
        stdp(i) = 0.0001;
    end
end

%%% Training Data Normalized
for i = 1:N
    value = (X_train(:, i) - meanp)./stdp; % Without dividing by the standard deviation
    X_train_normalized(:, i) = value;
end

% [D,N]=size(X_test); 

%%% Test Data Normalized
% for i = 1:N
%     value = (X_test(:, i) - meanp)./stdp; % Without dividing by the standard deviation
%     X_test_normalized(:,i) = value;
% end

%% PCA Reduction

[train, reconst1,W] = processing_data_norm(X_train_normalized, PCA, meanp, stdp,X_train);
% [test, reconst2,W] = processing_data_post(X_test_normalized, PCA, meanp, stdp,W);

%% KNN

knnMdl = fitcknn(train',y_train','NumNeighbors',5);
% knnclass = predict(knnMdl,test');

% Save result
% class = knnclass';
% save('Group08_knn.mat','name','PCA','class');

%% Functions
function [train, reconst, W] = processing_data(X_train, n_dim, meanp, stdp)

    data = X_train;   

    % Compute PCA transformation matrix using normalized training data
    [Wc,Diag] = eig(cov(data'));
    [D,N] = size(data); 
    total = sum(sum(Diag));
    eval = max(Diag);

    for i = 1:n_dim
        W(i,:) = Wc(:, D+1-i)'; 
    end
    pnproj = W*data;
    ExpectedError = 0;
    Diag(D, D);
    for j = 0:n_dim
        ExpectedError = ExpectedError + Diag(D-j, D-j);
    end    
    ExpectedError = ExpectedError/total;

    pnproj = W*data;
    reconstructed = W'*pnproj;
    for i = 1:N       
        p_reconstructed(:,i) = reconstructed(:,i).* stdp + meanp;
    end
    reconst = p_reconstructed;
    train = pnproj;
end

function [train,reconst,W,ExpectedError,actual_MSE]=processing_data_norm(X_train,n_dim,meanp,stdp,original_data)

    data = X_train;   

    % Compute PCA transformation matrix using normalized training data
    [Wc,Diag] = eig(cov(data'));
    [D,N] = size(data); 
    total = sum(sum(Diag));
    eval = max(Diag);
    
    for i=1:n_dim
        W(i,:)=Wc(:,D+1-i)'; 
    end
    pnproj=W*data;
    ExpectedError=0;
    Diag(D,D);
    for j=0:n_dim
        ExpectedError=ExpectedError+Diag(D-j,D-j);
    end
    ExpectedError=ExpectedError/total;

    pnproj = W*data;
    reconstructed=W'*pnproj;
    reconst = reconstructed;
    for i=1:N       
       p_reconstructed(:,i)=reconstructed(:,i).*stdp+meanp;
    end
    reconst=p_reconstructed;
    train=pnproj;
    reconst_unit = uint8(reconst);
    X_train_uint = uint8(original_data);
    actual_MSE = immse(reconst_unit,X_train_uint);
    disp(['Actual MSE of normalized data: ' num2str(actual_MSE)]);

end

function [train, reconst] = processing_data_post(X_train, meanp, stdp, W)
    data = X_train;  
    [D, N] = size(data);   
    pnproj = W*data;
    reconstructed = W'*pnproj;
    for i = 1:N       
        p_reconstructed(:, i) = reconstructed(:, i).* stdp + meanp;
    end
    reconst = p_reconstructed;
    train = pnproj;
end 

function print_digit(digits, num)
    digit = zeros(28, 28); % Initialize the digit matrix
    % Extracting two different elements from each column
    for i = 1:28
        for j = 1:28
            digit(i, j) = digits((i - 1) * 28 + j, num);
        end
    end
    imshow(digit); % Display the digit
 end
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

%% No Neural Network classification %%

%%% Separate training set from test set
X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%% Normalization %%%
[D,N] = size(X_train); 
clear meanp stdp
meanp = mean(X_train')';
stdp = std(X_train')';

for i = 1:D
    if stdp(i) == 0 
        stdp(i) = 0.0001;
    end
end

%%% Training Data Normalized:
for i = 1:N
    value = (X_train(:, i) - meanp);% ./stdp; %% Application of Stdp??
    X_train_normalized(:, i) = value;
end

[D, N] = size(X_test); 

%%% Test Data Normalized:
for i = 1:N
    value = (X_test(:, i) - meanp);% ./ stdp; %% Application of Stdp??
    X_test_normalized(:,i) = value;
end

%% PCA Reduction
print_digit(Trainnumbers.image, 10);
n_dim = 50;
[train, reconst1, W] = processing_data(X_train_normalized, n_dim, meanp, stdp);
[test, reconst2] = processing_data_post(X_test_normalized, meanp, stdp, W);

figure();
print_digit(reconst1, 10)

%% Bayesian Classifier 
% Normalization but without applying standard deviation
[bayModel,bayTrainError] = bayesian_classifier_training(train,y_train);
bayTrainErrorPerc = 100 - 100*bayTrainError/length(train(1,:));
[bayclass,bayTestError]  = bayesian_classifier_testing(test,y_test,bayModel);
bayTestErrorPerc = 100 - 100*bayTestError/length(test(1,:));

%% Knn Classifier
knnMdl = fitcknn(train', y_train', 'NumNeighbors', 5);
knnclass = predict(knnMdl, test');
no_errors_nn = length(find(knnclass' ~= y_test));
disp(['Misclassification error: ', num2str(no_errors_nn)]);
disp(['Acierto: ', num2str((2000 - no_errors_nn) / 2000)]);

cm = confusionchart(y_test, knnclass', ...
    'Title','Matriz de confusión', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% LDA Reduction



%% Neural Network classification %%
% to do

%% Functions
function [train, reconst, W] = processing_data(X_train, n_dim, meanp, stdp)

    data = X_train;   

    % Compute PCA transformation matrix using normalized training data
    [Wc,Diag] = eig(cov(data'));
    [D,N] = size(data); 
    total = sum(sum(Diag));
    eval = max(Diag);

    % explained_variance = 0.7;
    % for i = 0:783
    %     test = sum(eval((784-i):784))/total;
    %     if test >= explained_variance
    %         k = i;
    %         break
    %     end
    % end
    % Informacion_mantenida = sum(eval((784-n_dim):784))/total;
    % Vk = Wc(:,((784-(n_dim-1)):784));
    % k=n_dim;
    % train_tilde=data;  
    % F = Vk'*train_tilde;
    % pnproj=F;
    % compression_ratio = k/784;
    % train_tilde_k = Vk*F;
    % reconst = train_tilde_k;
    % for i=1:N       
    %     p_reconstructed(:,i)=train_tilde_k(:,i).*stdp+meanp;
    % end
    % reconst=p_reconstructed;
%%
    for i = 1:n_dim
        W(i,:) = Wc(:, D+1-i)'; 
    end
    pnproj = W*data;
    ExpectedError = 0;
    Diag(D, D);
    for j = 0:n_dim
        ExpectedError = ExpectedError + Diag(D-j, D-j);
    end    
    ExpectedError = ExpectedError/total

    pnproj = W*data;
    reconstructed = W'*pnproj;
    for i = 1:N       
        p_reconstructed(:,i) = reconstructed(:,i) .* stdp + meanp;
    end
    reconst = p_reconstructed;
    train = pnproj;
end

function [train, reconst] = processing_data_post(X_train, meanp, stdp, W)
    data = X_train;  
    [D, N] = size(data);   
    pnproj = W*data;
    reconstructed = W'*pnproj;
    for i = 1:N       
        p_reconstructed(:, i) = reconstructed(:, i) .* stdp + meanp;
    end
    reconst = p_reconstructed;
    train = pnproj;
end 

function print_digit(digits, num)
    digit = zeros(28, 28); % Initialize the digit matrix
    n = 1;
    % Extracting two different elements from each column
    for i = 1:28
        for j = 1:28
            digit(i, j) = digits((i - 1) * 28 + j, num);
        end
    end
    imshow(digit); % Display the digit
 end

%% Bayesian Classifier
function [bayMdl_Prior,errors_bay_Prior] = bayesian_classifier_training(X_train, y_train)
    warning('off')
    
    % Digit Probability
    digits = unique(y_train);
    count_digits = zeros(1,length(digits));
    prob_digits  = zeros(1,length(digits));
    for i = 1:length(digits)
        count_digits(i) = length(find(y_train == digits(i)));
        prob_digits(i)  = length(find(y_train == digits(i)))/length(y_train);
    end
    prior = prob_digits;
    
    % % Digit Mean
    % if length(X_train(:,1)) == 784
    %     digit_mean_image = zeros(length(X_train(:,1)),length(digits));
    %     for ii = 1:length(digits)
    %         digit_mean_image(:,ii) = round(mean(X_train(:,find(y_train==digits(ii))),2));
    %         figure();
    %         image(reshape(digit_mean_image(:,ii),28,28)');
    %     end
    % end
    
    % Include elements different from 0
    [r_train,c_train] = size(X_train);
    standard_deviation = 0.0001;
    X_train = X_train + standard_deviation.*randn(r_train,c_train);
    
    % Distribution "normal" - (Gaussian Distribution)
    % With Prior Probability
    bayMdl_Prior     = fitcnb(X_train',y_train','Prior',prior,'DistributionNames','normal');
    bayclass_Prior   = predict(bayMdl_Prior,X_train');
    errors_bay_Prior = length(find(bayclass_Prior'~=y_train));
    
    % Confusion Chart
    figure();
    cm = confusionchart(y_train',bayclass_Prior);
    cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    title('Bayesian Training - Confusion Matrix')
end

function [bayclass,errors_bay] = bayesian_classifier_testing(X_test,y_test,bayMdl)
    bayclass = predict(bayMdl,X_test');
    errors_bay = length(find(bayclass'~=y_test));

    figure();
    cm = confusionchart(y_test',bayclass);
    cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    title('Bayesian Testing - Confusion Matrix')
end


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
    value = (X_train(:, i) - meanp) ./stdp;
    X_train_normalized(:, i) = value;
end

[D,N]=size(X_test); 

%%% Test Data Normalized:
for i = 1:N
    value = (X_test(:, i) - meanp) ./ stdp;
    X_test_normalized(:,i) = value;
end

%%%% Classification %%%%

print_digit(Trainnumbers.image, 10);
n_dim = 100;
[train, reconst1,W] = processing_data(X_train_normalized, n_dim, meanp, stdp);
[test, reconst2] = processing_data_post(X_test_normalized, meanp, stdp, W);

print_digit(reconst1, 50)
print_digit(reconst2, 100)

knnMdl = fitcknn(train', y_train', 'NumNeighbors', 5);
knnclass = predict(knnMdl, test');
no_errors_nn = length(find(knnclass' ~= y_test));
disp(['Misclassification error: ', num2str(no_errors_nn)]);
disp(['Acierto: ', num2str((2000 - no_errors_nn) / 2000)]);

cm = confusionchart(y_test, knnclass', ...
    'Title','Matriz de confusión', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');


%%%% LDA %%%%



%% Neural Network classification
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



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


%%% Loading data %%%
load Trainnumbers.mat;

%% No Neural Network classification

% Separate training set from test set

X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%%% Clasification %%%%

print_digit(Trainnumbers.image,10)

[train,reconst1] = processing_data(X_train,y_train,154);
[test,reconst2] = processing_data(X_test,y_test,154);

print_digit(reconst1,33)

% test=processing_data(X_test,y_test,100);
train = real(train);
test = real(test);

knnMdl = fitcknn(train,y_train','NumNeighbors',5);
knnclass = predict(knnMdl,test);

no_errors_nn = length(find(knnclass'~=y_test));
disp(['Misclassification error: ', num2str(no_errors_nn)]);
disp(['Acierto: ', num2str((2000-no_errors_nn)/2000)]);

%% Neural Network classification

% to do


%% Functions
function [train, reconst] = processing_data(X_train, y_train, n_dim)
    %%% Normalization
    [D,N] = size(X_train); 
    clear meanp stdp
    meanp = mean(X_train')';
    stdp = std(X_train')';

    %%% Normalized data
    for i = 1:N
        value = (X_train(:,i)-meanp)./stdp;
        for j = 1:D
            pixel = value(j);
            if isnan(pixel)
                value(j) = 0;
            end
        end
        image_normalized(:,i) = value;
    end
    
    %%% PCA %%%%%%%%%%%%%%%%%%
    % image_normalized = X_train;
    data = image_normalized;
    original_data = X_train;
    
    % Compute PCA transformation matrix using normalized training data
    [Wc, Diag] = eig(cov(data'));
    [D, N] = size(data); 
    total = sum(sum(Diag));
    eval = max(Diag);

    explained_variance = 0.7;
    for i = 0:783
        test = sum(eval((784-i):784))/total;
        if (test >= explained_variance)
            k = i;
            break
        end
    end
    Vk = Wc(:,((784-(n_dim-1)):784));
    k = n_dim;
    train_tilde = data;
    % F = train_tilde'*Vk;
    F = Vk'*train_tilde;
    pnproj = F;
    compression_ratio = k/784;
    train_tilde_k = Vk*F;
    reconst = train_tilde_k;


    % for i=1:n_dim
    %     W(i,:)=Wc(:,D+1-i)'; 
    % end
    % pnproj=W*data;
    % ExpectedError=0;
    % Diag(D,D)
    % for j=0:n_dim
    %     ExpectedError=ExpectedError+Diag(D-j,D-j);
    % end
    % %error=Vk-W'
    % % ExpectedError
    % p_reconstructed=W'*pnproj;
    % 
    % p_reconstructed=p_reconstructed.*stdp+meanp;
    % 
    % reconst=p_reconstructed;

    %%% LDA %%%%%%%%%%%%%%%%%%
    
    % Step 1: Compute array Wnlda
    % Assuming X_train is your training data and y_train is the corresponding labels
    X_train = pnproj';
    y_train = y_train';
    class_labels = unique(y_train);
    num_classes = numel(class_labels);
    num_features = k;

    % Calculate class mean
    class_means = zeros(num_classes, num_features);
    for i = 1:num_classes
        class_means(i, :) = mean(X_train(y_train == class_labels(i), :));
    end

    % Within-class scatter matrix
    S_W = zeros(num_features, num_features);
    for i = 1:num_classes
        class_data = X_train(y_train == class_labels(i), :);
        class_mean_centered = class_data - class_means(i, :);
        S_W = S_W + (class_mean_centered' * class_mean_centered);
    end

    % Between-class scatter matrix
    overall_mean = mean(pnproj');
    S_B = zeros(num_features, num_features);
    for i = 1:num_classes
        n = sum(y_train == class_labels(i));
        mean_diff = class_means(i, :) - overall_mean;
        S_B = S_B + n * (mean_diff' * mean_diff);
    end

    % Computing W (eigenvecs) matrix: maximizing the Fisher Discriminants
    [eigvecs, eigvals] = eig(pinv(S_W) * S_B);

    % Choose the top k eigenvectors
    [~, idx] = sort(diag(eigvals), 'descend');
    Wland = eigvecs(:, idx(1:k-1));

    % Step 2: Compute 1D coordinates of test data
    X_test_projected_10D = X_train*Wland;

    train = X_test_projected_10D;
    % train=pnproj;
end

function print_digit(digits,num)
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



%% LDA for MNIST

clear all
close all
clc

%%% Loading data %%%
load Trainnumbers.mat;

%% No Neural Network classification %%

% Separate training set from test set
X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%%% Normalization %%%
[D, N] = size(X_train); 
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

[D, N]=size(X_test); 

%%% Test Data Normalized:
for i = 1:N
    value = (X_test(:, i) - meanp) ./ stdp;
    X_test_normalized(:,i) = value;
end

% print_digit(Trainnumbers.image,10);
n_dim = 9;

%% Only LDA


% Variables initialization
dimension = size(X_train_normalized,1);
Sw = zeros(dimension);
Sb = zeros(dimension);   % Could consider sparse here
N = size(X_train_normalized, 2); % Train data size
Nt = size(y_train, 2); % Test data size
Mu = mean(X_train_normalized, 2);  % Get mean vector of train data
scrsz = get(groot, 'ScreenSize'); % Get screen width and height

for i = 0:9

    % LDA Step 2. Construct Si matrix of each category
    mask = (y_train ==  i);
    x = X_train_normalized(:, mask);
    ni = size(x, 2);
    pi = ni / N;
    mu_i = mean(x, 2);

    Si = (1/ni) * (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';

    % LDA Step 3. Construct Sw within class covariance
    Sw = Sw + Si * pi;

    % LDA Step 4. Construct Sb between class covariance
    Sb = Sb + pi * (mu_i - Mu) * (mu_i - Mu)';
end


% LDA Step 5. Singular Value Decomposition of Sw\Sb
M = pinv(Sw) * Sb;  % Sw maybe singular, use pseudo-inverse
[U, D, V] = svd(M);

% LDA Step 6 Reduce dimension to 2
G2 = U(:, 1:n_dim);

% LDA Step 7 Reconstruct the train data matrix
Y2 = G2' * X_test_normalized;

recons = G2 * Y2;

print_digit(recons,10);


% Plot 2d figure
% data2d_fig = figure('Name', '2-D Plot');
% set(data2d_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);
% for number = 0:9
% 
%     mask = (y_train ==  number);
%     a = Y2(1,mask);
%     b = Y2(2,mask);
%     c = y_train(mask);
% 
%     % Draw 2D visualization in separate view
%     subplot(2, 5, number+1)
%     scatter(a',b');
%     title(['Number ', num2str(number)]);
% end


% n_dim = 3; % Dimensionality
% G3 = U(:, 1:n_dim);
% Y3 = G3' * X_train_normalized;
% 
% % Plot 3d figure
% data3d_fig = figure('Name', '3-D Plot');
% set(data3d_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);
% for number = 0:9
% 
%     mask = (y_train ==  number);
%     a = Y3(1, mask);
%     b = Y3(2, mask);
%     c = Y3(3, mask);
%     subplot(2,5,number+1);
%     scatter3(a',b',c');
%     title(['Number ', num2str(number)]);
% end


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


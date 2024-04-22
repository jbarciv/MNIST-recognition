%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% LDA for MNIST %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%% Loading data
load Trainnumbers.mat;

%% Separate training set from test set
X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

%% Normalize Training Data
[D_train, N_train] = size(X_train);
[X_train_normalized, meanp, stdp] = normalize(X_train);

%% Normalize Test Data 
[D_test, N_test] = size(X_test); 
for i = 1:N_test
    value = (X_test(:, i) - meanp) ./ stdp;
    X_test_normalized(:,i) = value;
end

%% Visualize initial numbers with and without normalization
% print_digit(X_train,10);
% print_digit(X_train_normalized,10);

%% LDA for 2 and 3 dimensions - plotting
n_dim = 2;
%%% Choose between normalized data and raw data
train_data = X_train_normalized; 
%%% Variables initialization
Sw = zeros(D_train);
Sb = zeros(D_train);
mu = meanp;
%%% Compute Sw and Sb matrices
for i = 0:9
    mask = (y_train ==  i);
    x = train_data(:, mask);
    ni = size(x, 2);
    pi = ni / N_train;
    mu_i = mean(x, 2);
    Si = (1/ni) * (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';
    Sw = Sw + Si * pi;
    Sb = Sb + pi * (mu_i - mu) * (mu_i - mu)';
end
%%% Compute Singular Value Decomposition (SVD) of Sw\Sb
M = pinv(Sw) * Sb;
[U, D, V] = svd(M);
%%% Compute projection matrix of n_dim
W_2 = U(:, 1:n_dim);
% Project the train data
Y = W_2' * train_data;
%%% Plot 2D and 3D projection for each number
% subplot2D3Ddata(y_train, Y, n_dim)
plot2D3Ddata(y_train, Y, n_dim)
n_dim = 3;
W_3 = U(:, 1:n_dim);
Y = W_3' * train_data;
% subplot2D3Ddata(y_train, Y, n_dim)
plot2D3Ddata(y_train, Y, n_dim)

%% LDA for Classification

% disp('LDA and kNN classification');
% %%% Choose between normalized data and raw data
% train_data = X_train;
% test_data = Y_train;
% for p = 1:9
%     W = U(:, 1:p);
% 
%     Y = W' * train_data;
%     Y_t = W' * T;
% 
%     % LDA Step 9 Classify test data using Nearest Neighbor
%     accuracy = classifyNN(Y_t, Y, test_label, train_label);
% 
%     % Display Messages on the screen 
%     message = ['Reduced dimension: ', num2str(p), ', ', ...
%         'Classification accuracy: ', num2str(sum(accuracy)*100), '%, '];
% 
%     disp(message);
% 
%     % restore classification result in accuracy map
%     iterator = iterator + 1;
%     accuracy_mat(iterator, :) = accuracy;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Normalized a given data. Also returns mean and stdeviation
function [normalized, meanp, stdp] = normalize(data)
    [D, N] = size(data);
    meanp = mean(data')';
    stdp = std(data')';
    for i = 1:D
        if (stdp(i) == 0) 
            stdp(i) = 0.0001;
        end
    end
    for i = 1:N
        value = (data(:, i) - meanp) ./stdp;
        normalized(:, i) = value;
    end
end

% Print a digit from whole matrix of numbers
function print_digit(digits,num)
    digit = zeros(28, 28);
    for i = 1:28
        for j = 1:28
            digit(i, j) = digits((i - 1) * 28 + j, num);
        end
    end
    figure;
    imshow(digit);
end

%Plot 2D or 3D subplots in the same figure
function subplot2D3Ddata(y_train, Y, dim)
    scrsz = get(groot, 'ScreenSize');
    if dim == 2
        data_fig = figure('Name', '2-D Plot');
    elseif dim == 3
        data_fig = figure('Name', '3-D Plot');
    else
        error("only 2D and 3D dimensions are allowed!");
    end
    set(data_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);
    for number = 0:9
        mask = (y_train ==  number);
        a = Y(1, mask);
        b = Y(2, mask);
        if dim == 2
            subplot(2, 5, number+1)
            scatter(a', b');
        elseif dim == 3
            c = Y(3, mask);
            subplot(2,5,number+1);
            scatter3(a', b', c');
        end
        axis([-0.6 0.6 -10 14])
        title(['Number ', num2str(number)]);
    end
end

% Plot 2D or 3D in only one figure
function plot2D3Ddata(y_train, Y, dim)
    scrsz = get(groot, 'ScreenSize'); 
    if dim == 2
        data_fig = figure('Name', '2-D Plot');
    elseif dim == 3
        data_fig = figure('Name', '3-D Plot');
    else
        error("only 2D and 3D dimensions are allowed!");
    end
    set(data_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);

    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250;...
              0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330;...
              0.6350 0.0780 0.1840; 0 0.5 0; 1 0.4 0.6; 0.5 0.2 0.8];
    hold on;
    for number = 0:9
        mask = (y_train ==  number);
        a = Y(1, mask);
        b = Y(2, mask);
        if dim == 2
            scatter(a, b, [], colors(number+1,:), 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', ['Number ', num2str(number)]);
        elseif dim == 3
            c = Y(3, mask);
            scatter3(a, b, c, [], colors(number+1,:), 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', ['Number ', num2str(number)]);
        end
    end
    hold off;

    axis([-0.6 0.6 -10 14])
    if dim == 2
        legend('Location', 'best');
    elseif dim == 3
        legend('Location', 'bestoutside');
    end
end


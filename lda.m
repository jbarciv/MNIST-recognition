%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% LDA for MNIST %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
% close all
% clc

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
    value = (X_test(:, i) - meanp)./ stdp;
    X_test_normalized(:,i) = value;
end

%% Visualize initial numbers with and without normalization
% print_digit(X_train,10);
% print_digit(X_train_normalized,10);

%% Choose between normalized data and raw data
train_data = X_train_normalized;
test_data = X_test_normalized;

%% LDA for 2 and 3 dimensions - plotting
LDA_ = false;
if LDA_
    n_dim = 9;
    fprintf("-> computing LDA to dim = %s\n", num2str(n_dim));
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
    W = U(:, 1:n_dim);
    % Project the train data
    Y = W' * train_data;
end

%% PCA reduction
PCA_ = false;
if PCA_
    n_dim = 3;
    fprintf("-> computing PCA to dim = %s\n", num2str(n_dim));
    [Y, ~] = myPCA(Y, n_dim);
end

%% Plot 2D and 3D projection for each number
plot_ = false;
if plot_
    % fprintf("-> plotting subplots\n"); 
    % subplot2D3Ddata(y_train, Y, n_dim)
    fprintf("-> plotting one big plot\n"); 
    plot2D3Ddata(y_train, Y, n_dim)
end

%% LDA for Classification
run_ = false;
if run_
    fprintf("-> LDA and kNN classification\n");
    tic
    for p = 9 %1:9
        W = U(:, 1:p);
        Y = W' * train_data;
        Y_t = W' * test_data;
        %%% Classify test data using Nearest Neighbor
        knnMdl = fitcknn(Y', y_train', 'NumNeighbors', 5);
        knnclass = predict(knnMdl, Y_t');
        no_errors_nn = length(find(knnclass' ~= y_test));
    
        % disp(['Misclassification error: ', num2str(no_errors_nn)]);
        % disp(['Acierto: ', num2str((2000 - no_errors_nn) / 2000)]);
        % cm = confusionchart(y_test, knnclass', ...
        %     'Title','Matriz de confusión', ...
        %     'RowSummary','row-normalized', ...
        %     'ColumnSummary','column-normalized');
    end
    toc
    disp(['Acierto: ', num2str((2000 - no_errors_nn) / 2000)]);
end

plot_ = false;
if plot_
    normalized_errors = [1720, 1158, 867, 634, 469, 437, 374, 353, 308];
    normalized_aciertos = [0.14, 0.421, 0.5665, 0.683, 0.7655, 0.7815, 0.813, 0.8235, 0.846];

    raw_errors = [1756, 1216, 869, 622, 481, 446, 383, 348, 293];
    raw_aciertos = [0.122, 0.392, 0.5655, 0.689, 0.7595, 0.777, 0.8085, 0.826, 0.8535];

    % Define bar width and displacement
    bar_width = 0.4; % Adjust as needed
    displacement = 0.1; % Adjust as needed

    % Define x-axis values for each group of bars
    x_normalized = 1:numel(normalized_aciertos);
    x_raw = 1:numel(raw_aciertos);

    % Plotting both data in the same figure
    hold on;
    bar(x_normalized - displacement, normalized_aciertos, bar_width, 'DisplayName', 'Normalized Data');
    bar(x_raw + displacement, raw_aciertos, bar_width, 'DisplayName', 'Raw Data');
    hold off;

    xlabel('Dimension Reduction');
    ylabel('Success');
    title('Succes vs Dimension Reduction for Normalized and Raw Data');
    legend('Location', 'northwest');
    xticks(1:numel(normalized_aciertos));
    xticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9'});

    % Adjusting figure
    grid on;
end

%% LDA and kNN with PCA
run_ = false;
if run_
    fprintf('PCA{748-0} -> LDA{x-9}? -> kNN classification\n');
    success = [];
    for i = 180 %784:-2:10
        tic
        %%% Compute PCA
        % fprintf('--> computing PCA to dim = %d\n', i);
        [train_data, W_pca] = myPCA(train_data, i);
        test_data = W_pca*test_data;
        [D_train, N_train] = size(train_data);
        %%% Compute LDA
        LDA_ = true;
        if LDA_
            n_dim = 9;
            % fprintf('--> computing LDA to dim = %d\n', n_dim);
            Sw = zeros(D_train);
            Sb = zeros(D_train);
            mu = mean(train_data');
            %%% Compute Sw and Sb matrices
            for j = 0:9
                mask = (y_train ==  j);
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
            W = U(:, 1:n_dim);
            Y = W' * train_data;
            Y_t = W' * test_data;
        else
            Y = train_data;
            Y_t = test_data;
        end
        kNN_ = false;
        if kNN_
            %%% Classify test data using Nearest Neighbor
            knnMdl = fitcknn(Y', y_train', 'NumNeighbors', 5);
            knnclass = predict(knnMdl, Y_t');
            no_errors_nn = length(find(knnclass' ~= y_test));
            % disp(['Misclassification error: ', num2str(no_errors_nn)]);
            % disp(['Acierto: ', num2str((2000 - no_errors_nn) / 2000)]);
            % cm = confusionchart(y_test, knnclass', ...
            %     'Title','Matriz de confusión', ...
            %     'RowSummary','row-normalized', ...
            %     'ColumnSummary','column-normalized');
            success = [success, (2000 - no_errors_nn) / 2000];
        end
        bayes_ = true;
        if bayes_
            %%% Classify test data using Naive Bayes
            bayMdl_Prior = bayesian_classifier_training(Y, y_train);
            [bayclass, errors_bay] = bayesian_classifier_testing(Y_t, y_test, bayMdl_Prior);
            % disp(['Acierto: ', num2str((2000 - errors_bay) / 2000)]);
            success = [success, (2000 - errors_bay) / 2000];
        end
        toc
        disp(num2str(success(1)));
    end 
end



% kNN -> negras 
% bayes -> verdes
% raw -> continua
% normalized -> discontinua


plot_ = true;
if plot_ 
    % Load the data
    y_pca_knn_raw = load("pca_knn.mat");
    y_pca_knn_normalized = load("pca_knn_normalized.mat");
    y_pca_lda_knn_raw = load("pca_lda_knn.mat");
    y_pca_lda_knn_normalized = load("pca_lda_knn_normalized.mat");
    x = 784:-2:10;

    % Create a new figure
    figure;
    % Plot the first array with a solid blue line
    plot(x, y_pca_knn_raw.success, 'k', 'LineWidth', 2);
    hold on; % Keep the current plot and add to it

    % Plot the second array with a dashed red line
    plot(x, y_pca_knn_normalized.success, 'g--', 'LineWidth', 2);

    % Plot the third array with a dotted green line
    plot(x, y_pca_lda_knn_raw.success, 'b:', 'LineWidth', 2);

    % Plot the fourth array with a dash-dot black line
    plot(x, y_pca_lda_knn_normalized.success, 'r-.', 'LineWidth', 2);

    % % Add labels and title
    % xlabel('Dimension Reduction');
    % ylabel('Success');
    % title('Success vs Dimension Reduction for Normalized and Raw Data');
    % 
    % % Add legend
    % legend('PCA+KNN Raw', 'PCA+kNN Normalized', 'PCA+LDA+kNN Raw', 'PCA+LDA+KNN Normalized', 'Location',  'north');
    % 
    % % Hold off to reset the hold state
    % hold off;
end

plot_ = true;
if plot_ 
    % Load the data
    y_pca_bayes_raw = load("pca_bayes_raw.mat");
    y_pca_bayes_normalized = load("pca_bayes_normalized.mat");
    y_pca_bayes_normalized_without_std = load("pca_bayes_normalized_without_std.mat");
    x = 784:-2:10;

    % Create a new figure
    % figure;
    % Plot the first array with a solid blue line
    plot(x, y_pca_bayes_raw.success, 'r', 'LineWidth', 2);
    hold on; % Keep the current plot and add to it

    % Plot the second array with a dashed red line
    plot(x, y_pca_bayes_normalized.success, 'b--', 'LineWidth', 2);

    % Plot the third array with a dotted green line
    % plot(x, y_pca_bayes_normalized_without_std.success, 'g:', 'LineWidth', 2);

    % % Add labels and title
    % xlabel('Dimension Reduction');
    % ylabel('Success');
    % title('Bayesian Classifier Success vs Dimension Reduction for Normalized and Raw Data');
    % 
    % % Add legend
    % legend('PCA+Bayes Raw', 'PCA+Bayes Normalized', 'PCA+Bayes Normalized no std division', 'Location',  'north');
    % 
    % % Hold off to reset the hold state
    % hold off;
end

plot_ = true;
if plot_ 
    % Load the data
    y_pca_lda_bayes_raw = load("pca_lda_bayes_raw.mat");
    y_pca_lda_bayes_normalized = load("pca_lda_bayes_normalized.mat");
    y_pca_lda_bayes_normalized_without_std = load("pca_lda_bayes_normalized_without_std.mat");
    x = 784:-2:10;

    % Create a new figure
    % figure;
    % Plot the first array with a solid blue line
    plot(x, y_pca_lda_bayes_raw.success, 'g:', 'LineWidth', 2);
    hold on; % Keep the current plot and add to it

    % Plot the second array with a dashed red line
    plot(x, y_pca_lda_bayes_normalized.success, 'k-.', 'LineWidth', 2);

    % Plot the third array with a dotted green line
    % plot(x, y_pca_lda_bayes_normalized_without_std.success, 'g:', 'LineWidth', 2);


    % Add labels and title
    xlabel('Dimension Reduction');
    ylabel('Success');
    title('Bayesian Classifier Success vs Dimension Reduction for Normalized and Raw Data');

    % Add legend
    legend('PCA+KNN Raw', 'PCA+kNN Normalized', 'PCA+LDA+kNN Raw', 'PCA+LDA+KNN Normalized', ...
           'PCA+Bayes Raw', 'PCA+Bayes Normalized', ...
           'PCA+LDA+Bayes Raw', 'PCA+LDA+Bayes Normalized', 'Location',  'best');

    % Hold off to reset the hold state
    hold off;
end

%% LDA with Bayes
run_ = false;
if run_
    tic
    success = [];
    % fprintf("-> LDA and Bayes classification\n");
    for p = 9 % 1:9
        W = U(:, 1:p);
        Y = W' * train_data;
        Y_t = W' * test_data;
        %%% Classify test data using Naive Bayes
        bayMdl_Prior = bayesian_classifier_training(Y, y_train);
        % disp(['Misclassification error  in Trinning: ', num2str(errors_bay_Prior/8000)]);
        % disp(['Acierto: ', num2str(1 - errors_bay_Prior / 8000)]);
        [bayclass, errors_bay] = bayesian_classifier_testing(Y_t,y_test,bayMdl_Prior);
        % disp(['Misclassification error in Testing: ', num2str(errors_bay/2000)]);
        % disp(['Acierto: ', num2str(1 - errors_bay / 2000)]);
        success = [success, 1-errors_bay/2000];
    end
    toc
    disp(['Acierto: ', num2str(1 - errors_bay / 2000)]);
end

plot_ = false;
if plot_
    y_lda_bayes_raw = load("lda_bayes_raw.mat");
    y_lda_bayes_raw = y_lda_bayes_raw.success;
    y_lda_bayes_normalized = load("lda_bayes_normalized.mat");
    y_lda_bayes_normalized = y_lda_bayes_normalized.success;
    y_lda_bayes_normalized_std = load("lda_bayes_normalized_std.mat");
    y_lda_bayes_normalized_std = y_lda_bayes_normalized_std.success;
    
    % Define bar width and displacement
    bar_width = 0.4; % Adjust as needed
    displacement = 0.1; % Adjust as needed
    
    % Define x-axis values for each group of bars
    x_raw = 1:numel(y_lda_bayes_raw);
    x_normalized = 1:numel(y_lda_bayes_normalized);
    x_normalized_std = 1:numel(y_lda_bayes_normalized_std);
    
    % Plotting both data in the same figure
    hold on;
    bar(x_raw - displacement, y_lda_bayes_raw, bar_width, 'DisplayName', 'Raw Data');
    bar(x_normalized + displacement, y_lda_bayes_normalized, bar_width, 'DisplayName', 'Normalized Data no std division');
    bar(x_normalized_std + 2*displacement, y_lda_bayes_normalized_std, bar_width, 'DisplayName', 'Normalized Data');
    hold off;
    
    xlabel('LDA Dimension Reduction (from the initial 784 dimensions)');
    ylabel('Success');
    title('Bayes Classification Succes Rate vs Dimension Reduction for Normalized (with and without std division) and Raw Data');
    legend('Location', 'northwest');
    xticks(1:numel(y_lda_bayes_raw));
    xticklabels({'1', '2', '3', '4', '5', '6', '7', '8', '9'});
    
    % Adjusting figure
    grid on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [train, W] = myPCA(data, n_dim)   
    [Wc, Diag] = eig(cov(data'));
    [D, N] = size(data); 
    for i = 1:n_dim
        W(i,:) = Wc(:, D+1-i)'; 
    end
    train = W*data;
end

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
        value = (data(:, i) - meanp)./stdp;
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
        % axis([-0.6 0.6 -10 14])
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

    % axis([-0.6 0.6 -10 14])
    if dim == 2
        legend('Location', 'best');
    elseif dim == 3
        legend('Location', 'bestoutside');
    end
end

%% Bayesian Classifier
function bayMdl_Prior = bayesian_classifier_training(X_train, y_train)
    % warning('off')
    
    % Digit Probability
    digits = unique(y_train);
    count_digits = zeros(1,length(digits));
    prob_digits  = zeros(1,length(digits));
    for i = 1:length(digits)
        count_digits(i) = length(find(y_train == digits(i)));
        prob_digits(i)  = length(find(y_train == digits(i)))/length(y_train);
    end
    prior = prob_digits;
    
    % Include elements different from 0
    [r_train,c_train] = size(X_train);
    standard_deviation = 0.0001;
    X_train = X_train + standard_deviation.*randn(r_train,c_train);
    
    % Distribution "normal" - (Gaussian Distribution)
    % With Prior Probability
    bayMdl_Prior     = fitcnb(X_train',y_train','Prior',prior,'DistributionNames','normal');
    % bayclass_Prior   = predict(bayMdl_Prior,X_train');
    % errors_bay_Prior = length(find(bayclass_Prior'~=y_train));
    
    % Confusion Chart
    % figure();
    % cm = confusionchart(y_train',bayclass_Prior);
    % cm.NormalizedValues;
    % cm.RowSummary = 'row-normalized';
    % cm.ColumnSummary = 'column-normalized';
    % title('Bayesian Training - Confusion Matrix')
end

function [bayclass, errors_bay] = bayesian_classifier_testing(X_test, y_test, bayMdl)
    bayclass = predict(bayMdl, X_test');
    errors_bay = length(find(bayclass' ~= y_test));
    % figure();
    % cm = confusionchart(y_test',bayclass);
    % cm.NormalizedValues;
    % cm.RowSummary = 'row-normalized';
    % cm.ColumnSummary = 'column-normalized';
    % title('Bayesian Testing - Confusion Matrix')
end

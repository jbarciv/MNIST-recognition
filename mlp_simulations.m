%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Master in Robotics
%                    Applied Artificial Intelligence
%
% Final project:  Visual Handwritten Digits Recognition
% Students:
%
%   - David Redondo Quintero (23147)
%   - Josep M. Barberá Civera (17048)
%
% First version: 02/05/2024
% Second version: 11/06/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
% clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. Training set and MLP structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. One full training (1 layer with 300 neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
full_training = false; %%%%% <- change here as needed! %%%%%
if full_training
    % Net creation
    nn = patternnet(300);
    
    nn.divideFcn = 'divideind';
    nn.divideParam.trainInd = 1:6000;
    nn.divideParam.valInd = 6001:8000;
    nn.divideParam.testInd = 8001:10000;
    
    [nn, tr] = train(nn, images, labels);
    
    %%%%Plotting MNIST training performance
    plot(tr.perf, 'LineWidth', 2); % plot training error 
    hold on;
    plot(tr.vperf, 'r-.', 'LineWidth', 2); % plot validation error 
    plot ( tr.tperf, 'g:', 'LineWidth', 2); % plot test error
    set(gca, 'yscale', 'log'); % setting log scale 
    axis ([1, 107, 0.001, 1.8]); 
    xlabel('Training Epochs', 'FontSize', 14); 
    ylabel('Cross−Entropy', 'FontSize', 14); 
    title ('Performance Trend on MNIST', 'FontSize', 16);
    h = legend({'Training', 'Validation', 'Test'}, 'Location', 'NorthEast');
    set(h, 'FontSize', 14);
    
    % Network evaluation
    f = nn(images(:, 1:6000));          % Training predictions
    fv = nn(images(:, 6001:8000));      % Validation predictions
    ft = nn(images(:, 8001:end));       % Test predictions
    
    % Classification accuracy
    
    A = mean(vec2ind(f) == vec2ind(labels(:, 1:6000)));
    Av = mean(vec2ind(fv) == vec2ind(labels(:, 6001:8000)));
    At = mean(vec2ind(ft) == vec2ind(labels(:, 8001:end)));
    
    % Confusion matrix
    C = confusionmat(vec2ind(f), vec2ind(labels(:, 1:6000)));
    Cv = confusionmat(vec2ind(fv), vec2ind(labels(:, 6001:8000)));
    Ct = confusionmat(vec2ind(ft), vec2ind(labels(:, 8001:end)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Number of neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shuffle = false  ; %%%%% <- change here as needed! %%%%%

if (shuffle)
    folder_name = "net_structure";
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    neurons = 140:1:150;
    trainFcn = 'trainscg';

    for i = 1:numel(neurons)
        fprintf("Training with %d neurons in the hidden layer\n", neurons(i));
        myneurons = neurons(i);
        for j = 1:3
            nn = patternnet(myneurons, trainFcn);
            nn.divideFcn = 'divideind';
            nn.divideParam.trainInd = 1:6000;
            nn.divideParam.valInd = 6001:8000;
            nn.divideParam.testInd = 8001:10000;
            [nn, tr] = train(nn, images, labels);
            % Network evaluation
            f = nn(images(:, 1:6000));          % Training predictions
            fv = nn(images(:, 6001:8000));     % Validation predictions
            ft = nn(images(:, 8001:end));       % Test predictions
            % Classification accuracy
            A_j(j) = mean(vec2ind(f) == vec2ind(labels(:, 1:6000)));
            Av_j(j) = mean(vec2ind(fv) == vec2ind(labels(:, 6001:8000)));
            At_j(j) = mean(vec2ind(ft) == vec2ind(labels(:, 8001:end)));
            epochs_j(j) = tr.num_epochs;
        end
        A(i) = mean(A_j);
        Av(i) = mean(Av_j);
        At(i) = mean(At_j);
        epochs (i) = mean(epochs_j);
        plot(tr.perf, 'LineWidth', 2); % plot training error 
        hold on;
        plot(tr.vperf, 'r-.', 'LineWidth', 2); % plot validation error 
        plot ( tr.tperf, 'g:', 'LineWidth', 2); % plot test error
        set(gca, 'yscale', 'log'); % setting log scale 
        % axis ([1, 107, 0.001, 1.8]); 
        xlabel('Training Epochs', 'FontSize', 14); 
        ylabel('Cross−Entropy', 'FontSize', 14); 
        my_title = sprintf("MLP 1 hidden layer with %d neurons", neurons(i));
        title(my_title, 'FontSize', 16);
        my_subtitle = sprintf("Test: %.4f, Validation: %.4f, Train: %.4f", ...
                              At(i), Av(i), A(i));
        subtitle(my_subtitle);
        h = legend({'Training', 'Validation', 'Test'}, 'Location', 'NorthEast');
        set(h, 'FontSize', 14);
        hold off;
        figure_name = sprintf("/1_layer_%d_neurons.png", neurons(i));
        saveas(gcf, strcat(folder_name, figure_name));
        fprintf("Test: %.4f, Validation: %.4f, Train: %.4f, Epochs: %3.1f\n", ...
                              At(i), Av(i), A(i), epochs(i));
        disp("***********************************");
    end
    % Plot errors
    plot(neurons, At, 'g-', 'LineWidth', 2, 'DisplayName', 'Test');
    hold on;
    plot(neurons, A, 'b-', 'LineWidth', 2, 'DisplayName', 'Train');
    plot(neurons, Av, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation');
    
    % Title and labels
    title('Accuracy vs. Number of Neurons in Hidden Layer');
    xlabel('Number of Neurons');
    ylabel('Error');
    legend('Location', 'best');
    
    % Adjust figure
    grid on;
    figure_name = sprintf("/error_plot.png");
    saveas(gcf, strcat(folder_name, figure_name));
end

accuracy_plot = false; %%%%% <- change here as needed! %%%%%
if (accuracy_plot)
    % neurons = 50:50:800;
    % test_accuracies = [0.9305, 0.9367, 0.9337, 0.9368, 0.9372, 0.9182, 0.9355, 0.9322, 0.8917, 0.9294, 0.9121, 0.9320, 0.9322, 0.9297, 0.8949, 0.9127];
    % validation_accuracies = [0.9275, 0.9370, 0.9365, 0.9352, 0.9357, 0.9152, 0.9341, 0.9322, 0.8923, 0.9305, 0.9092, 0.9335, 0.9288, 0.9267, 0.8949, 0.9096];

    neurons = 100:10:250;
    test_accuracies = [0.9328, 0.9360, 0.9407, 0.9343, 0.9327, 0.9435, 0.9382, 0.9392, 0.9382, 0.9397, 0.9342, 0.9360, 0.9130, 0.9350, 0.9380, 0.9362];
    validation_accuracies = [0.9350, 0.9375, 0.9363, 0.9370, 0.9335, 0.9380, 0.9360, 0.9373, 0.9387, 0.9402, 0.9348, 0.9365, 0.9100, 0.9358, 0.9410, 0.9380];


    [max_test_acc, idx_test] = max(test_accuracies);
    max_test_neuron = neurons(idx_test);
    [max_val_acc, idx_val] = max(validation_accuracies);
    max_val_neuron = neurons(idx_val);
    
    figure;
    plot(neurons, test_accuracies, '-o', 'LineWidth', 2.5, 'DisplayName', 'Test Accuracy');
    hold on;
    plot(neurons, validation_accuracies, '-s', 'LineWidth', 2.5, 'DisplayName', 'Validation Accuracy');
    xlabel('Number of Neurons in the Hidden Layer', 'FontSize', 15, 'Interpreter', 'latex');
    ylabel('Classification Accuracy', 'FontSize', 15, 'Interpreter', 'latex');
    title('Classification Accuracy vs. Nº Neurons', 'FontSize', 17, 'Interpreter', 'latex');
    legend('Location', 'NorthEast');
    grid on;
    
    text(max_test_neuron, max_test_acc, sprintf('%.4f', max_test_acc), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontSize', 12);
    text(max_val_neuron, max_val_acc, sprintf('%.4f', max_val_acc), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 12);

    ylim([0.89 1]);
    set(gca, 'XTick', neurons);
    
    % ax = gca;
    % ax.XLabel.Position = [200, 0.89, 0]; % Adjust x-axis label position
    % ax.YLabel.Position = [140, 0.95, 0]; % Adjust y-axis label position
    
    saveas(gcf, 'classification_accuracy_vs_neurons_smaller_GOOD.png');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Others
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shuffled = true; %%%%% <- cambiar según sea necesario! %%%%%

if (shuffled)
    num_trials = 10; % Número de intentos a ejecutar
    accuracies = zeros(1, num_trials);
    epochs = zeros(1, num_trials);
    times = zeros(1, num_trials); % Vector para almacenar los tiempos

    for trial = 1:num_trials
        nn = patternnet([600 150 20], 'traincgf');
        nn.divideFcn = 'divideind';
        shuffled_indices = randperm(6000); % Permutar los índices
        nn.divideParam.trainInd = shuffled_indices;
        nn.divideParam.valInd = 6001:8000;
        nn.divideParam.testInd = 8001:10000;
        try
            tic; 
            [nn, tr] = train(nn, images, labels);
            times(trial) = toc; 
            
            % Evaluación de la red
            f = nn(images(:, 1:6000));          
            fv = nn(images(:, 6001:8000));      
            ft = nn(images(:, 8001:end));       
            % Exactitud de clasificación
            A = mean(vec2ind(f) == vec2ind(labels(:, 1:6000)));
            Av = mean(vec2ind(fv) == vec2ind(labels(:, 6001:8000)));
            At = mean(vec2ind(ft) == vec2ind(labels(:, 8001:end)));
            accuracies(trial) = At;
            epochs(trial) = tr.num_epochs;
            save(sprintf('nn_trial_%d.mat', trial), 'nn');
        catch exception
            fprintf("Fallo en el entrenamiento con el método %s: %s\n", trainFcn, exception.message);
        end
    end

    mean_accuracy = mean(accuracies);
    mean_epochs = mean(epochs);
    mean_time = mean(times); % Calcular el tiempo promedio

    fprintf("Exactitud media de prueba en %d intentos: %.4f\n", num_trials, mean_accuracy);
    fprintf("Épocas medias en %d intentos: %.1f\n", num_trials, mean_epochs);
    fprintf("Tiempo medio de entrenamiento en %d intentos: %.4f segundos\n", num_trials, mean_time);
end

not_shuffled = false; %%%%% <- change here as needed! %%%%%

if (not_shuffled)
    layers = 150;
    trainFcn = 'traincgf';
    num_trials = 10; % Number of trials to run
    
    accuracies = zeros(1, num_trials);
    epochs = zeros(1, num_trials);
    
    for trial = 1:num_trials
        nn = patternnet(layers, trainFcn);
        nn.divideFcn = 'divideind';
        nn.divideParam.trainInd = 1:6000; % Without shuffling
        nn.divideParam.valInd = 6001:8000;
        nn.divideParam.testInd = 8001:10000;
        try
            [nn, tr] = train(nn, images, labels);
            % Network evaluation
            f = nn(images(:, 1:6000));          % Training predictions
            fv = nn(images(:, 6001:8000));      % Validation predictions
            ft = nn(images(:, 8001:end));       % Test predictions
            % Classification accuracy
            A = mean(vec2ind(f) == vec2ind(labels(:, 1:6000)));
            Av = mean(vec2ind(fv) == vec2ind(labels(:, 6001:8000)));
            At = mean(vec2ind(ft) == vec2ind(labels(:, 8001:end)));
            accuracies(trial) = At; % Store test accuracy
            epochs(trial) = tr.num_epochs;
        catch exception
            fprintf("Training failed with method %s: %s\n", current_trainFcn, exception.message);
        end
    end
    mean_accuracy = mean(accuracies);
    mean_epochs = mean(epochs);
    fprintf("Mean test accuracy over %d trials: %.4f\n", num_trials, mean_accuracy);
    fprintf("Mean epochs over %d trials: %.1f\n", num_trials, mean_epochs);
end

% Data
methods = {'kNN', 'Bayes', 'k-means', 'MLP', 'SOM', 'CNN'};
accuracy = [95.7, 87.45, 84.6, 94.49, 93.6, 96.95];

% Colors
colors = [
    0, 0.4470, 0.7410;   % Dark blue
    0.8500, 0.3250, 0.0980;  % Dark orange
    0.9290, 0.6940, 0.1250;  % Dark yellow
    0.4940, 0.1840, 0.5560;  % Dark purple
    0.4660, 0.6740, 0.1880;  % Medium green
    0.3010, 0.7450, 0.9330;  % Light blue
    0.6350, 0.0780, 0.1840   % Dark red
];

% Create the bar plot
figure;
b = bar(accuracy);

% Apply colors to each bar
b.FaceColor = 'flat';
for k = 1:length(accuracy)
    b.CData(k, :) = colors(k, :);
end

% Add text labels on top of each bar
for k = 1:length(accuracy)
    text(k, accuracy(k) + 1, num2str(accuracy(k), '%.2f'), 'HorizontalAlignment', 'center');
end

set(gca, 'XTickLabel', methods);
xlabel('Methods');
ylabel('Accuracy (%)');
title('Accuracy for the different methods');
ylim([0 100]);
grid on;

% Save the figure
saveas(gcf, 'accuracy_bar_plot.png');


% Training with 2 layers, each with 75 neurons
% Test: 0.9380, Validation: 0.9410, Train: 0.9937, Epochs: 61.0
% ***********************************
% Training with 3 layers, each with 50 neurons
% Test: 0.9305, Validation: 0.9310, Train: 0.9890, Epochs: 72.0
% ***********************************
% Training with 4 layers, each with 37 neurons
% Test: 0.9215, Validation: 0.9265, Train: 0.9910, Epochs: 82.0
% ***********************************
% Training with 5 layers, each with 30 neurons
% Test: 0.9160, Validation: 0.9140, Train: 0.9730, Epochs: 92.0

% Training with 2 layers, each with 150 neurons
% Test: 0.9365, Validation: 0.9335, Train: 0.9913, Epochs: 54.0
% ***********************************
% Training with 3 layers, each with 150 neurons
% Test: 0.9365, Validation: 0.9410, Train: 0.9993, Epochs: 62.0
% ***********************************
% Training with 4 layers, each with 150 neurons
% Test: 0.9415, Validation: 0.9330, Train: 0.9935, Epochs: 60.0
% ***********************************
% Training with 5 layers, each with 150 neurons
% Test: 0.9350, Validation: 0.9325, Train: 0.9805, Epochs: 59.0
% ***********************************

% Training with 2 layers, each with 100 50 neurons
% Test: 0.9450, Validation: 0.9390, Train: 0.9930, Epochs: 54.0
% ***********************************
% Training with 3 layers, each with 150 100 50 neurons
% Test: 0.9455, Validation: 0.9445, Train: 0.9988, Epochs: 66.0
% ***********************************
% Training with 4 layers, each with 200 150 100 50 neurons
% Test: 0.9400, Validation: 0.9360, Train: 0.9977, Epochs: 71.0
% ***********************************
% Training with 5 layers, each with 250 200 150 100 50 neurons
% Test: 0.9395, Validation: 0.9345, Train: 0.9937, Epochs: 65.0
% ***********************************

% Training with 2 layers, each with 400 400 neurons
% Test: 0.9330, Validation: 0.9320, Train: 0.9885, Epochs: 55.0
% ***********************************
% Training with 3 layers, each with 700 500 200 neurons
% Test: 0.9380, Validation: 0.9415, Train: 0.9953, Epochs: 63.0
% ***********************************
% Training with 4 layers, each with 700 600 300 100 neurons
% Test: 0.9450, Validation: 0.9465, Train: 0.9982, Epochs: 67.0
% ***********************************
% Training with 5 layers, each with 700 700 500 300 100 neurons
% Test: 0.9375, Validation: 0.9430, Train: 0.9973, Epochs: 62.0
% ***********************************
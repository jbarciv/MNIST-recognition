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
            stdp(i)=0.00001;
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
    value = (X_test(:, i) - meanp)./ stdp; %% Application of Stdp??
    X_test_normalized(:,i) = value;
end

%%%% Clasification %%%%

% print_digit(Trainnumbers.image,50)
% print_digit(Trainnumbers.image,8100)
iterator=1;
ite = 1;
figure;
for j=1:764
    if mod(j, 50) == 0 || j < 200
        if mod(j, 10) == 0 || j < 50
            n_dim=j;
            % [train,reconst1,W,information,MSE]=processing_data_norm(X_train_normalized,n_dim,meanp,stdp,X_train);
            [train,reconst4,W,information,MSE]=processing_data(X_train,n_dim,meanp,stdp);
            [test,reconst2]=processing_data_post(X_test,meanp,stdp,W);
            
            % digit = print_digit(reconst1,50);
            % subplot(2, 2, 1);
            % imshow(print_digit(X_train_normalized,50));
            % title('Normalised image');
            % subplot(2, 2, 2);
            % imshow(print_digit(X_train,50),[0,255]);
            % title('Non-normalised image');
            % subplot(2, 2, 3);
            % imshow(print_digit(reconst1,50),[0,255]);
            % title('Normalised image after PCA 50');
            % subplot(2, 2, 4);
            % imshow(print_digit(reconst4,50),[0,255]);
            % title('Non-normalised image after PCA 50');
            
            tic
            knnMdl = fitcknn(X_train',y_train','NumNeighbors',5);
            knnclass = predict(knnMdl,X_test');
            no_errors_nn=length(find(knnclass'~=y_test));
            acierto=((2000-no_errors_nn)/2000);
            disp(['Misclassification error: ', num2str(no_errors_nn)]);
            disp(['Acierto: ', num2str(acierto)]);
            toc
            results(iterator)=acierto;
            n_dims(iterator)=n_dim;
            informations(iterator)=information;
            MSEs(iterator)=MSE;
            iterator=iterator+1;

            if j == 2 || j == 5 || j == 10 || j == 20 || j == 30 || j == 40
                subplot(2, 3, ite);
                imshow(print_digit(reconst4,50),[0,250]);
                str =['Non-normalised image PCA ', num2str(j),'D' ];
                newStr = join(str);
                title(newStr);
                ite = ite +1;
            end
        end
    end
end

%%
values=[1,3,5,7,9,11,13,15,17,19,21];
iterator=1;
for j=1:11
    n_dim=40;
    [train,reconst1,W,information,MSE]=processing_data(X_train_normalized,n_dim,meanp,stdp);
    [test,reconst2]=processing_data_post(X_test_normalized,meanp,stdp,W);
    
    % print_digit(reconst1,50)
    % print_digit(reconst2,100)
    
    knnMdl = fitcknn(train',y_train','NumNeighbors',values(j));
    knnclass = predict(knnMdl,test');
    no_errors_nn=length(find(knnclass'~=y_test));
    acierto=((2000-no_errors_nn)/2000);
    disp(['Misclassification error: ', num2str(no_errors_nn)]);
    disp(['Acierto: ', num2str(acierto)]);
    results(iterator)=acierto;
    n_dims(iterator)=n_dim;
    informations(iterator)=information;
    MSEs(iterator)=MSE;
    iterator=iterator+1;
end
%%
plot(values,results)
title("Accuracy of the KNN according to dimension reduction")
xlabel("Reduced to # dimension")
ylabel("Accuaracy of model")

%%

[train,reconst4,W,information,MSE]=processing_data(X_train_normalized,40,meanp,stdp);
[test,reconst2]=processing_data_post(X_test_normalized,meanp,stdp,W);
knnMdl = fitcknn(train',y_train','NumNeighbors',5);
knnclass = predict(knnMdl,test');
no_errors_nn=length(find(knnclass'~=y_test));
acierto=((2000-no_errors_nn)/2000);
disp(['Misclassification error: ', num2str(no_errors_nn)]);
disp(['Acierto: ', num2str(acierto)]);
cm = confusionchart(y_test,knnclass');

%% Bayesian Classifier

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
%%
plot(n_dims,results)
title("Accuracy of the KNN according to dimension reduction")
xlabel("Reduced to # dimension")
ylabel("Accuaracy of model")

%%
plot(n_dims,informations)
title("Preservation of information according to dimension reduction")
xlabel("Reduced to # dimension")
ylabel("Preservation of information")
%%
plot(n_dims,MSEs)
title("MSE according to dimension reduction")
xlabel("Reduced to # dimension")
ylabel("MSE")

%%
[train,reconst1,W,information,MSE]=processing_data(X_train_normalized,2,meanp,stdp);
[test,reconst2]=processing_data_post(X_test_normalized,meanp,stdp,W);
figure;
for i=0:9
    hold on;
    labels=find(y_train==i);
    tam=size(labels);
    for j=1:tam(2)
        array(:,j)=train(:,labels(j));
    end
    scatter(array(1,:),array(2,:))
    array=[];
    title("PCA 2D with normalised data")
    legend("0","1","2","3","4","5","6","7","8","9");
end






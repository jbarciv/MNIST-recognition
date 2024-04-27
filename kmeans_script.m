%%
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

%%% Normalizacion %%%
[D,N]=size(X_train); 
clear meanp stdp
meanp=mean(X_train')';
stdp=std(X_train')';

for i=1:D
    if stdp(i) == 0 
            stdp(i)=0.0000001;
    end
end
%%% Datos Train normalizados:
for i=1:N
    value=(X_train(:,i)-meanp)./stdp;
    X_train_normalized(:,i)=value;
end
[D,N]=size(X_test); 

%%% Datos Test normalizados:
for i=1:N
    value=(X_test(:,i)-meanp)./stdp;
    X_test_normalized(:,i)=value;
end

% % Crear la figura
% figure;
% 
% % Primer subplot
% subplot(2, 2, 1);
% imshow(print_digit(X_train_normalized,50));
% title('Normalised image');
% 
% % Segundo subplot
% subplot(2, 2, 2);
% digit=print_digit(X_train,50);
% imshow(print_digit(X_train,50),[0,255]);
% title('Non-normalised image');
% 
% % Tercer subplot
% subplot(2, 2, 3);
% digit = print_digit(X_train_normalized,10);
% imshow(print_digit(X_train_normalized,10));
% title('Normalised image');
% 
% % Cuarto subplot
% subplot(2, 2, 4);
% imshow(print_digit(X_train,10),[0,255]);
% title('Non-normalised image');

%% PCA & LDA
n_dim=40;
[PCA_data,reconst,W,information,MSE]=processing_data_norm(X_train_normalized,n_dim,meanp,stdp,X_train);
% [LDA_data]=LDA(X_train_normalized,20);



%%

numClusters = 20;
[idx,C]=kmeans(X_train_normalized',numClusters);
[most_probable_label,indexes] = infer_cluster_labels(idx,numClusters,y_train);
most_probable_label
%% Muestra el centroide de los clusters que es el numero más representativo del cluster
digit = print_digit(C',18);
imshow(digit);



%%

% %% muestra el primer numero de cada cluster
% figure;
% for i = 1:numClusters
%     clusterIdx = find(idx == i);
%     subplot(2, numClusters/2, i);
%     imshow(print_digit(X_train,clusterIdx(1)),[0,255]);
%     title(['Cluster ', num2str(i)]);
% end
%% Muestra los 10 primeros de cada cluster
figure;
for i = 1:numClusters
    clusterIdx = find(idx == i);
    for j = 1:10
        subplot(2, 5, j);
        [A,B]=size(clusterIdx);
        if j<=A
            imshow(print_digit(X_train,clusterIdx(j)),[0,255]);
            title(['Cluster ', num2str(i)]);
        end
    end
    clf
end

%%




















%% functions

function [train,reconst,W,ExpectedError,actual_MSE]=processing_data_norm(X_train,n_dim,meanp,stdp,original_data)

    data = X_train;   

    % Compute PCA transformation matrix using normalized training data
    [Wc,Diag]=eig(cov(data'));
    [D,N]=size(data); 
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
    reconst=acotar_matriz(reconst,0,255);
    squared_diff = (reconst - original_data).^2;
    sum_squared_diff=sum(squared_diff, 'all');
    actual_MSE = sum_squared_diff / N;
    disp(['Actual MSE of normalized data: ' num2str(mean(actual_MSE))]);

end

function [train,reconst]=processing_data_post(X_train,meanp,stdp,W)
data = X_train;  
    [D,N]=size(data);   
    pnproj = W*data;
    reconstructed=W'*pnproj;
    reconst=reconstructed;
    for i=1:N       
        p_reconstructed(:,i)=reconstructed(:,i).*stdp+meanp;
    end
    reconst=p_reconstructed;train_data
    train=pnproj;
end 



function [Y]=LDA(X_train_normalized,n_dim)
    %% Choose between normalized data and raw data
    train_data = X_train_normalized;
    fprintf("-> computing LDA to dim = %s\n", num2str(n_dim));
    %%% Variables initialization
    Sw = zeros(train_data);
    Sb = zeros(train_data);
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
    Y = W' *train_data;
end

function matriz_acotada = acotar_matriz(matriz, limite_inferior, limite_superior)
    % Acotar los valores de la matriz
    matriz_acotada = matriz;
    matriz_acotada(matriz_acotada < limite_inferior) = limite_inferior;
    matriz_acotada(matriz_acotada > limite_superior) = limite_superior;
end


function [most_probable_label,indexes] = infer_cluster_labels(idx,numClusters,actual_labels)
    % Associates most probable label with each cluster in KMeans model
    % returns: dictionary of clusters assigned to each label

    inferred_labels = [];
    indexes = [];
    labels = [];
    most_probable_label = [];

    for i = 1:numClusters

        % find index of points in cluster
        index = find(idx == i);

        % append actual labels for each point in cluster
        labels = actual_labels(index);
        number = mode(labels);
        indexes = [indexes labels];
        most_probable_label = [most_probable_label number];
    end 
end

function predicted_labels = infer_data_labels(X_labels, cluster_labels)
    % Determines label for each array, depending on the cluster it has been assigned to.
    % returns: predicted labels for each array

    % empty array of len(X)
    predicted_labels = zeros(length(X_labels), 1, 'uint8');
    
    for i = 1:length(X_labels)
        cluster = X_labels(i);
        keys = cell2mat(keys(cluster_labels));
        values = cell2mat(values(cluster_labels));
        for j = 1:length(keys)
            if ismember(cluster, values{j})
                predicted_labels(i) = keys(j);
                break;
            end
        end
    end
end



   
       
   
   
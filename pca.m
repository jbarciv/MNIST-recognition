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
            stdp(i)=0.00001;
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

% Crear la figura
figure;

% Primer subplot
subplot(2, 2, 1);
imshow(print_digit(X_train_normalized,50));
title('Normalised image');

% Segundo subplot
subplot(2, 2, 2);
digit=print_digit(X_train,50);
imshow(print_digit(X_train,50),[0,255]);
title('Non-normalised image');

% Tercer subplot
subplot(2, 2, 3);
digit = print_digit(X_train_normalized,10);
imshow(print_digit(X_train_normalized,10));
title('Normalised image');

% Cuarto subplot
subplot(2, 2, 4);
imshow(print_digit(X_train,10),[0,255]);
title('Non-normalised image');
%%
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
cm = confusionchart(y_test,knnclass', ...
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


%% Neural Network classification

% to do


%% Functions
function [train,reconst,W,ExpectedError,actual_MSE]=processing_data(X_train,n_dim,meanp,stdp)

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
    train=pnproj;
    % acotar_matriz(reconst,0,255);
    squared_diff = (reconst - X_train).^2;
    sum_squared_diff=sum(squared_diff, 'all');
    actual_MSE = sum_squared_diff / N;
    disp(['Actual MSE of normalized data: ' num2str(mean(actual_MSE))]);

end

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
    % reconst=acotar_matriz(reconst,0,255);
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
    reconst=p_reconstructed;
    train=pnproj;
end 

% function [digit]=print_digit(digits,num)
%     digit = zeros(28, 28); % Initialize the digit matrix
%     for i = 1:28
%         for j = 1:28
%             digit(i, j) = digits((i - 1) * 28 + j, num);
%         end
%     end
%     %imshow(digit); %Display the digit
% end

function matriz_acotada = acotar_matriz(matriz, limite_inferior, limite_superior)
    % Acotar los valores de la matriz
    matriz_acotada = matriz;
    matriz_acotada(matriz_acotada < limite_inferior) = limite_inferior;
    matriz_acotada(matriz_acotada > limite_superior) = limite_superior;
end



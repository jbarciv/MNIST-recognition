clear all
close all
clc

%%% Loading data %%%
load Trainnumbers.mat;

%%

X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);
%%
num_size = 8;
somnet = selforgmap([num_size num_size]);
somnet.trainParam.epochs = 10;
somnet=train(somnet, X_train);

%% Labeling neurons 
num_clases = 10;
yntrain=somnet(X_train);
yntrain_ind=vec2ind(yntrain);
neuron_class = compare_lists(yntrain_ind,y_train,num_size*num_size,num_clases);


%% Prediction
yntest=somnet(X_test);
yntest_ind=vec2ind(yntest);
prediction = predict_class(neuron_class, yntest_ind);
prediction = prediction;
C = confusionchart(y_test,prediction);
C.Title = 'SOM as classifier';
C.RowSummary = 'row-normalized';
C.ColumnSummary = 'column-normalized';

%%
no_errors_nn=length(find(prediction~=y_test));
accuracy = (2000-no_errors_nn)/2000;
disp(['Accuracy: ', num2str(accuracy)]);


%% Show plot
% Visualizar SOM y mostrar las imágenes de los dígitos en las neuronas correspondientes
figure;
image_total = [];
fila = [];
digit_image = [];
j = 1;
for i = 1:num_size*num_size
    idx = find(yntrain_ind == i);
    images = X_train(:,idx);
    [A,B]= size(images);
    if B >1
        mean_image = mean(images');
    else
        mean_image = images;
    end
    digit_image = reshape(mean_image', 28, 28); % Si cada imagen es de 28x28
    fila = [fila digit_image'];
    j = j+1;
    if j == num_size+1
        j = 1;
        image_total = [image_total; fila];
        fila = [];
    end
end
imshow(image_total, [0 255]);


%% Functions
function neuron_class = compare_lists(activated_neurons, input_classes, num_neurons, num_classes)
    % Initialize array to store counts of classes for each neuron
    class_counts = zeros(num_neurons, num_classes);

    % Iterate over activated_neurons and input_classes lists to count class occurrences for each neuron
    for i = 1:length(activated_neurons)
        neuron = activated_neurons(i);
        class = input_classes(i);
        class = class+1;
        class_counts(neuron, class) = class_counts(neuron, class) + 1;
    end

    % Initialize array to store the most frequent class for each neuron
    neuron_class = zeros(num_neurons, 1);

    % Determine the most frequent class for each neuron
    for i = 1:num_neurons
        [~, max_class] = max(class_counts(i,:));
        neuron_class(i) = max_class-1;
    end
end

function prediction = predict_class(neuron_class, yntind)

    prediction = [];
    for i=1:length(yntind)
        neurona=yntind(i);
        tipo = neuron_class(neurona);
        prediction = [prediction tipo];
    end
end
%% Loading data %%
load Trainnumbers.mat;

%% Separate training set from test set
X_train = Trainnumbers.image(:,1:8000);
y_train = Trainnumbers.label(:,1:8000);
X_test = Trainnumbers.image(:,8001:10000);
y_test = Trainnumbers.label(:,8001:10000);

X_tr = reshape(X_train, [28, 28, 1, 8000]);
X_te= reshape(X_test, [28, 28, 1, 8000]); 

%% Define DL parameters
layers = [imageInputLayer([28 28 1])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer];
options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',3,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress');
net = trainNetwork(X_tr, categorical(y_train),layers, options);
class = classify(net,X_te);
accuracy = length(find(eq(categorical(y_test),class)))/length(y_test);
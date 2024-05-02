function [bayMdl_Prior,errors_bay_Prior] = bayesian_classifier_training(X_train, y_train)
    warning('off')
    
    % Digit Probability
    digits = unique(y_train);
    count_digits = zeros(1,length(digits));
    prob_digits  = zeros(1,length(digits));
    for i = 1:length(digits)
        count_digits(i) = length(find(y_train == digits(i)));
        prob_digits(i)  = length(find(y_train == digits(i)))/length(y_train);
    end
    prior = prob_digits;
    
    % % Digit Mean
    % if length(X_train(:,1)) == 784
    %     digit_mean_image = zeros(length(X_train(:,1)),length(digits));
    %     for ii = 1:length(digits)
    %         digit_mean_image(:,ii) = round(mean(X_train(:,find(y_train==digits(ii))),2));
    %         figure();
    %         image(reshape(digit_mean_image(:,ii),28,28)');
    %     end
    % end
    
    % Include elements different from 0
    [r_train,c_train] = size(X_train);
    standard_deviation = 0.0001;
    X_train = X_train + standard_deviation.*randn(r_train,c_train);
    
    % Distribution "normal" - (Gaussian Distribution)
    % With Prior Probability
    bayMdl_Prior     = fitcnb(X_train',y_train','Prior',prior,'DistributionNames','normal');
    bayclass_Prior   = predict(bayMdl_Prior,X_train');
    errors_bay_Prior = length(find(bayclass_Prior'~=y_train));
    
    % Confusion Chart
    figure();
    cm = confusionchart(y_train',bayclass_Prior);
    cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    title('Bayesian Training - Confusion Matrix')
end
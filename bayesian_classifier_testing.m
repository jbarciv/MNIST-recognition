function [bayclass,errors_bay] = bayesian_classifier_testing(X_test,y_test,bayMdl)
    bayclass = predict(bayMdl,X_test');
    errors_bay = length(find(bayclass'~=y_test));

    figure();
    cm = confusionchart(y_test',bayclass);
    cm.NormalizedValues;
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    title('Bayesian Testing - Confusion Matrix')
end
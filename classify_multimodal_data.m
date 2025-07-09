%% Random Forest Classification

function classify_multimodal_data(cca_Fmri, cca_Metabol, labels)
    combined = [cca_Fmri, cca_Metabol];
    cv = cvpartition(labels, 'HoldOut', 0.2);
    train_idx = training(cv);
    test_idx = test(cv);

    model = TreeBagger(100, combined(train_idx,:), labels(train_idx), 'Method', 'classification');
    preds = str2double(predict(model, combined(test_idx,:)));

    cm = confusionmat(labels(test_idx), preds);
    disp('Confusion Matrix:');
    disp(cm);
    acc = sum(diag(cm)) / sum(cm(:));
    fprintf('Accuracy: %.2f%%\n', acc * 100);
end
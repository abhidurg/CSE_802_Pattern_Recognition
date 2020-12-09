% D = importdata('winequality-red.csv');
% D_total = D.data;
% N = zscore(D_total(:,1:11));

% yfit = trainedModel.predictFcn(N);
%  N = [N D_total(:,12)];
% C = confusionmat(D_total(:,12), yfit); 
% confusionchart(C);

D = importdata('True_train_label4.csv');
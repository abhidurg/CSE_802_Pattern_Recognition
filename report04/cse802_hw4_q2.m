D = importdata('hw04_data.txt');
D_training = [D(1:250, :); D(501:750, :); D(1001:1250, :)];
D_test = [D(251:500, :); D(751:1000, :); D(1251:1500, :)];

%Bayesian classification
Bayes_prediction = zeros(750, 1);
for i=1:750
    p1 = mvnpdf(D_test(i,1:2), [0, 0], 4*[1 0; 0 1]);
    p2 = mvnpdf(D_test(i,1:2), [10, 0], 4*[1 0; 0 1]);
    p3 = mvnpdf(D_test(i,1:2), [5, 5], 5*[1 0; 0 1]);
    [M, I] = max([p1 p2 p3]);
    Bayes_prediction(i) = I;
end

% C_bayes = confusionmat(D_test(:,3), Bayes_prediction);
% confusionchart(C_bayes);
% title('Confusion matrix of Bayesian classification for test set');


%MLE
mean_mle_1 = mean(D_training(1:250, 1:2));
mean_mle_2 = mean(D_training(251:500, 1:2));
mean_mle_3 = mean(D_training(501:750, 1:2));
cov_mle_1 = cov(D_training(1:250, 1:2), 1);
cov_mle_2 = cov(D_training(251:500, 1:2), 1);
cov_mle_3 = cov(D_training(501:750, 1:2), 1);
MLE_prediction = zeros(750, 1);
for i=1:750
    p1 = mvnpdf(D_test(i,1:2), mean_mle_1, cov_mle_1);
    p2 = mvnpdf(D_test(i,1:2), mean_mle_2, cov_mle_2);
    p3 = mvnpdf(D_test(i,1:2), mean_mle_3, cov_mle_3);
    [M, I] = max([p1 p2 p3]);
    MLE_prediction(i) = I;
end

% C_MLE = confusionmat(D_test(:,3), MLE_prediction);
% confusionchart(C_MLE);
% title('Confusion matrix of MLE for test set');
%Parzen
n = 750;
h1 = 1;  %try for 0.01, 0.1, 1, and 10
%h_n = h1/sqrt(n);
h_n = 1;  %this will get a better performance 57/750 instead of 82/750
parzen_prediction = zeros(750, 1);
for x=1:n
    p_n_1 = 0;
    p_n_2 = 0;
    p_n_3 = 0;
    for i=1:n/3
        p_n_1 = p_n_1 + (1/h_n)*mvnpdf((D_test(x,1:2)- D_training(i,1:2))/h_n);
        p_n_2 = p_n_2 + (1/h_n)*mvnpdf((D_test(x,1:2)- D_training(i+250,1:2))/h_n);
        p_n_3 = p_n_3 + (1/h_n)*mvnpdf((D_test(x,1:2)- D_training(i+500,1:2))/h_n);
    end
    p_n_1 = p_n_1/n;
    p_n_2 = p_n_2/n;
    p_n_3 = p_n_3/n;
    [M, I] = max([p_n_1 p_n_2 p_n_3]);
    parzen_prediction(x) = I;
end
C_parzen = confusionmat(D_test(:,3), parzen_prediction);
confusionchart(C_parzen);
title('Confusion matrix of parzen window for test set');
%kNN
k = 1;
knn_prediction = zeros(750, 1);
for x=1:750
    euclid_dist = zeros(750, 1);
    for i=1:750
        euclid_dist(i) = norm(D_test(x,1:2) - D_training(i,1:2));
    end
    [M, I] = mink(euclid_dist, k);
    knn_prediction(x) = D_training(I, 3); %this is only for 1-NN
end

% C_knn = confusionmat(D_test(:,3), knn_prediction);
% confusionchart(C_knn);
% title('Confusion matrix of 1-NN for test set');


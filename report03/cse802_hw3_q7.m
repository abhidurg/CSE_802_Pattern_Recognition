D = importdata('imox_data.txt');
A = zeros(32,2);
for i=1:8
    for j=1:4
        if j == 1
            A(i,1) = mean(D(1:48,i));
            A(i,2) = var(D(1:48,i),1);
        elseif j == 2
            A(i+8,1) = mean(D(49:96,i));
            A(i+8,2) = var(D(49:96,i),1);
        elseif j == 3
            A(i+16,1) = mean(D(97:144,i));
            A(i+16,2) = var(D(97:144,i),1);
        else
            A(i+24,1) = mean(D(145:192,i));
            A(i+24,2) = var(D(145:192,i),1);
        end
    end
end
B = zeros(4,8);
for i = 1:8
    B(1,i) = var(D(1:24,i),1);
    B(2,i) = var(D(49:72,i),1);
    B(3,i) = var(D(97:120,i),1);
    B(4,i) = var(D(145:168,i),1);
end
%mle estimate for variance is biased
mu_1 = mean(D(1:24,1:8));
cov_1 = zeros(8);
cov_2 = zeros(8);
cov_3 = zeros(8);
cov_4 = zeros(8);
for i=1:8
    cov_1(i,i) = B(1,i);
    cov_2(i,i) = B(2,i);
    cov_3(i,i) = B(3,i);
    cov_4(i,i) = B(4,i);
end
mu_2 = mean(D(49:72,1:8));
mu_3 = mean(D(97:120,1:8));
mu_4 = mean(D(145:168,1:8));

testing_data = [D(25:48,:);D(73:96,:);D(121:144,:);D(169:192,:)];
R = zeros(96, 6); 
R(:,1) = mvnpdf(testing_data(:,1:8), mu_1, cov_1);
R(:,2) = mvnpdf(testing_data(:,1:8), mu_2, cov_2);
R(:,3) = mvnpdf(testing_data(:,1:8), mu_3, cov_3);
R(:,4) = mvnpdf(testing_data(:,1:8), mu_4, cov_4);

[M, R(:,5)] = max(R(:,1:4), [], 2); %predicted class
R(:,6) = testing_data(:,9); %true class

C = confusionmat(R(:,6), R(:,5));
confusionchart(C)
title('Confusion Matrix')

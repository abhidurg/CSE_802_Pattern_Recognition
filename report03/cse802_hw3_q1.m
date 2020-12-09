
D = importdata('hw3_random_data_q1.txt');
h = histogram(D, 'Normalization', 'probability');
hold on;
sum_of_squares = 0;
for i=1:1000
    sum_of_squares = sum_of_squares + (D(i))^2;
end
theta_mle = size(D,1)/sum_of_squares;
b = sqrt(1/(2*theta_mle));
x = [-5:0.01:21];
p1 = raylpdf(x, b);
% plot(x,p1);
% xlabel('Feature values');
% ylabel('Normalized frequency');
% title('Normalized Histogram of features and Rayleigh pdf with theta-mle = 0.0195');
% legend('Frequency(normalized)', 'Rayleigh pdf');
mu = mean(D);
s = std(D, 1);
p2 = normpdf(x, mu, s);
plot(x,p2);
xlabel('Feature values');
ylabel('Normalized frequency');
title('Normalized Histogram of features and Gaussian pdf with mu = 6.3072 and sigma = 3.3949');
legend('Frequency(normalized)', 'Gaussian pdf');

rng('default');
mu = [0, 0];
sigma = [1 0; 0 1];
mu2 = [5, 5];
sigma2 = [1 0;0 1];
determ = det(sigma);
determ2 = det(sigma2);
invers = inv(sigma);
invers2 = inv(sigma2);
R1 = mvnrnd(mu, sigma, 50000);
R2 = mvnrnd(mu2, sigma2, 50000);
mu_mle = mean(R1);
mu2_mle = mean(R2);
cov_mle = cov(R1, 1);
cov2_mle = cov(R2, 1);
determ_mle = det(cov_mle);
determ2_mle = det(cov2_mle);
invers_mle = inv(cov_mle);
invers2_mle = inv(cov2_mle);

syms x1 x2 real
multiGauss = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu)*invers*([x1 x2]-mu)') == (1/(sqrt(determ2*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu2)*invers2*([x1 x2]-mu2)'); 
multiGauss_mle = (1/(sqrt(determ_mle*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu_mle)*invers_mle*([x1 x2]-mu_mle)') == ...
    (1/(sqrt(determ2_mle*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu2_mle)*invers2_mle*([x1 x2]-mu2_mle)'); 
S = solve(multiGauss, x2);
S_mle = solve(multiGauss_mle, x2);

scatter(R1(:, 1), R1(:, 2), 'filled', 'MarkerFaceColor', 'red');
hold on;
scatter(R2(:, 1), R2(:, 2), 'filled','MarkerFaceColor', 'blue');
f1 = fplot(S);
f1.Color = 'c';
f2 = fplot(S_mle(2));
f2.Color = 'g';
digits(5);
vpa(S_mle(2))
xlabel('Feature value x1');
ylabel('Feature value x2');
title('50000 random samples from class1 and class 2 with Bayes decision boundary using MLE and true parameters');
legend('class 1', 'class 2', 'true boundary', 'estimated boundary');



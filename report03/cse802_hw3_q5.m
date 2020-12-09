D = importdata('iris_data.txt');

%class 1
mu1 = mean(D(1:25, 1:4));
sigma1 = cov(D(1:25, 1:4));
determ1 = det(sigma1);
invers1 = inv(sigma1);

%class 2
mu2 = mean(D(51:75,1:4));
sigma2 = cov(D(51:75, 1:4));
determ2 = det(sigma2);
invers2 = inv(sigma2);

%class 3
mu3 = mean(D(101:125, 1:4));
sigma3 = cov(D(101:125, 1:4));
determ3 = det(sigma3);
invers3 = inv(sigma3);

%THIS WORKS DUMBO
% sample = [5, 3.5, 1.5, 0.25];
% y = mvnpdf(sample, mu2, sigma2);

syms x1 x2 x3 x4
multiGauss1(x1, x2, x3 ,x4) = (1/(sqrt(determ1*(2*pi)^4)))*exp((-1/2)*([x1 x2 x3 x4]-mu1)*invers1*([x1 x2 x3 x4]-mu1)');
multiGauss2(x1, x2, x3 ,x4) = (1/(sqrt(determ2*(2*pi)^4)))*exp((-1/2)*([x1 x2 x3 x4]-mu2)*invers2*([x1 x2 x3 x4]-mu2)'); 
multiGauss3(x1, x2, x3 ,x4) = (1/(sqrt(determ3*(2*pi)^4)))*exp((-1/2)*([x1 x2 x3 x4]-mu3)*invers3*([x1 x2 x3 x4]-mu3)');

% A = zeros(1,3);
% predicted_class = zeros(75,1);
% testing_set = [D(26:50,1:5);D(76:100,1:5);D(126:150,1:5)];
% 
% for i=1:75
%    A(1) = multiGauss1(testing_set(i,1), testing_set(i,2), testing_set(i,3), testing_set(i,4));
%    A(2) = multiGauss2(testing_set(i,1), testing_set(i,2), testing_set(i,3), testing_set(i,4));
%    A(3) = multiGauss3(testing_set(i,1), testing_set(i,2), testing_set(i,3), testing_set(i,4));
%    [M,I] = max(A);
%    predicted_class(i) = I;
% end
% 
% C = confusionmat(testing_set(:,5), predicted_class);
% confusionchart(C)
% title('Confusion Matrix')
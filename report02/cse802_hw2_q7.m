%part a
mu = [0 0]';
cov = [20 10; 10 30];
[eigenvec, eigenval] = eig(cov);
whitening = eigenvec*eigenval^(-1/2); %correct whitening is actually the transpose of this

%part b = desnity function has same mu, but cov = identity matrix

%part c
rng('default');
R = mvnrnd(mu, cov, 10000);
scatter(R(:,1), R(:,2), 'filled', 'MarkerFaceColor', 'r');
%hold on;
W = zeros(10000,2);
xlabel('Feature value x1');
ylabel('Feature value x2');
title('Plot of (x1,x2) for 10,000 random bivariate patterns');

%part d
for i=1:10000
   transform = whitening'*R(i,:)';
   W(i,:) = transform';
end
scatter(W(:,1), W(:,2), 'filled', 'MarkerFaceColor', 'b');
xlabel('Feature value x1');
ylabel('Feature value x2');
title('Plot of 10,000 bivariate patterns after whitening transformation');
%part e: W should be spherical in distribution with covariance = I
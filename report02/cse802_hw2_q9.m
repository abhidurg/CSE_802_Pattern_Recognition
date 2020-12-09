mu = [-1 -1];
mu2 = [1 1];
sigma1 = [1 0;0 1];
sigma2 = [1 0;0 1];
% mu = 5;
% mu2 = 1;
% sigma1 = 1;
% sgima2 = 4;

syms B
chernoff = 1/exp((B*(1-B)/2)*(mu2 - mu)*inv(B*sigma1 + (1-B)*sigma2)*(mu2-mu)' + (1/2)*log(det(B*sigma1 + (1-B)*sigma2)/(((det(sigma1))^B)*(det(sigma2)^(1-B)))));
%ezplot(chernoff, [0 1]);
%type in "chernoff" in the command window to see the simplified expression.

der = diff(chernoff, B);
B_min = solve(der == 0);
f(B) = chernoff;
P_error = sqrt((1/2)*(1/2))*f(1/2);

%take derivative of equation and equal to 0: B = 1/2. Can also see clearly
%in the graph

%Since B =1/2, the Bhattacharya and Chernoff bounds are identical

beta = 1/2;
final = exp(-(beta*(1-beta)/2)*(mu2 - mu)*inv(beta*sigma + (1-beta)*sigma)*(mu2-mu)' + (1/2)*log(det(beta*sigma + (1-beta)*sigma)/(((det(sigma))^beta)*(det(sigma)^(1-beta)))));
%P(error) = sqrt(p(w1)*p(w2))*e^-k(1/2) = sqrt(p(w1)*p(w2))*final =
%1/2(0.3679) = 0.18395

%rng('default');
misclassified = zeros(1000,1); %ran my code 1000 times to test if empirical error can exceed theoretical bounds
for j=1:1000
R = mvnrnd(mu, sigma, 25);
%scatter(R(:,1), R(:,2), 'filled', 'MarkerFaceColor', 'g');
hold on;
W = mvnrnd(mu2, sigma, 25);
%scatter(W(:,1), W(:,2), 'filled', 'MarkerFaceColor', 'm');

% syms x
% y = -x;
% fplot(x,y);
% 
% title('Plot of 25 random patterns from each class (1 and 2)');
% xlabel('Feature value x1');
% ylabel('Feature value x2');
% legend('class 1','class 2','decision boundary x2 = -x1');

tp = 0;
fp = 0;
tn = 0;
fn = 0;

for i=1:25
    if -1*R(i,1) >= R(i,2) %then we assign to class 1 as per decision rule, correctly
        tp = tp + 1;
    else %then we assign to class 2, wrongly
        fp = fp + 1;
    end
    if -1*W(i,1) >= W(i,2) %then we assign to class 1, wrongly
        fn = fn + 1;
    else %then we assign to class 2, correctly
        tn = tn + 1;
    end
end
misclassified(j,1) = fp+fn; %highest misclassified was 12, happened during 134th j

end
%error rate in this random example is (4+3)/50 = 0.14
        

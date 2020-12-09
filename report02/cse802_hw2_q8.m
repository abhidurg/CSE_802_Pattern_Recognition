mu = [0 0];
cov = [2 0;0 2];
determ = det(cov);
invers = inv(cov);

mu2 = [2 2];
cov2 = [1 0;0 1];
determ2 = det(cov2);
invers2 = inv(cov2);

% syms x
% multiGauss = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(x-mu)*invers*(x-mu)') == (1/(sqrt(determ2*(2*pi)^2)))*exp((-1/2)*(x-mu2)*invers2*(x-mu2)'); 
% S = solve(multiGauss, x);


% syms x1 x2
% multiGauss = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu)*invers*([x1 x2]-mu)');
% multiGauss2 = (1/(sqrt(determ2*(2*pi)^2)))*exp((-1/2)*([x1 x2]-mu2)*invers2*([x1 x2]-mu2)');
% fin = multiGauss - multiGauss2;


%part c
rng('default');
R = mvnrnd(mu, cov, 10000);
scatter(R(:,1), R(:,2), 'filled', 'MarkerFaceColor', 'g');

hold on;

W = mvnrnd(mu2, cov2, 10000);
scatter(W(:,1), W(:,2), 'filled', 'MarkerFaceColor', 'b');
title('Plot of 10,000 random points each from class 1 and class 2');
xlabel('Feature value x1');
ylabel('Feature value x2c');
legend('class 1', 'class 2');


tp = 0;
fp = 0;
tn = 0;
fn = 0;

for i=1:10000
    %the first 10000 points, generated in R (class 1)
    lhs = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(R(i,:)-mu)*invers*(R(i,:)-mu)');
    rhs = (1/(sqrt(determ2*(2*pi)^2)))*exp((-1/2)*(R(i,:)-mu2)*invers2*(R(i,:)-mu2)');
    if lhs >= rhs %then assign to class 1. since these points are from R, this is a true positive (tp)
        tp = tp + 1;
    else %then assign to class 2. since these points are from R, this is a false positive (fp)
        fp = fp + 1;
    end
    lhs = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(W(i,:)-mu)*invers*(W(i,:)-mu)');
    rhs = (1/(sqrt(determ2*(2*pi)^2)))*exp((-1/2)*(W(i,:)-mu2)*invers2*(W(i,:)-mu2)');
     if lhs >= rhs %then assign to class 1. since these points are from W, this is a false negative (fn)
        fn = fn + 1;
     else %then assign to class 2. Since these points are from W, this is a true negative(tn)
        tn = tn + 1; 
    end
end

%confusion matrix: tp = 8594, fp = 1406; fn = 818, tn = 9182
%error for class 1: 14.06%
%error for class 2: 8.18%
%total error from the 20,000 samples = 2224/20000 = 11.12%


%ezplot(fin, [[-1,11],[-1,11]]);
viscircles([4,4], sqrt(16+4*log(2)));





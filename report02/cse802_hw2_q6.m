hold on;
plot(-1,1, 'o', 'MarkerEdgeColor', 'm', 'DisplayName', 'Class 1 mean'); %class w1
plot(1,1, 'o', 'MarkerEdgeColor', 'r', 'DisplayName', 'Class 2 mean'); %class w2
plot(0.5,0.5,'o', 'MarkerEdgeColor', 'b', 'DisplayName', 'Class 3 mean, component 1'); %,GMM 1st part
plot(-0.5,-0.5,'o', 'MarkerEdgeColor', 'c', 'DisplayName', 'Class 3 mean, component 2'); %GMM 2nd part
plot(0.1,0.1, 'o', 'MarkerEdgeColor', 'k', 'DisplayName', 'x'); %x
xlabel('Feature value x1');
ylabel('Feature value x2');
title('Mean of each class (1-3) and the point x=(0.1,0.1)');
legend;

mu = [-1,-1];
cov = [1 0;0 1];
determ = det(cov);
invers = inv(cov);
x1 = [0.1, 0.1];
multiGauss = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(x1-mu)*invers*(x1-mu)'); %class 1 desnity

mu = [1,1];
multiGauss2 = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(x1-mu)*invers*(x1-mu)'); %class 2 density

mu = [0.5,0.5];
multiGauss3_1 = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(x1-mu)*invers*(x1-mu)'); %GMM 1st component

mu = [-0.5,-0.5];
multiGauss3_2 = (1/(sqrt(determ*(2*pi)^2)))*exp((-1/2)*(x1-mu)*invers*(x1-mu)'); %GMM 2nt component

multiGauss3 = 0.5*multiGauss3_1 + 0.5*multiGauss3_2; %Total GMM has the highest probability, so we assign to class 3
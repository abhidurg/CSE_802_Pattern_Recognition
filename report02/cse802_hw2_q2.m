mu = [1 1 1];
cov_matrix = [1 0 0; 0 5 2; 0 2 5];
determ = det(cov_matrix);
invers = inv(cov_matrix);
[eigen_vec, eigen_val] = eig(cov_matrix);


x1 = [0,0,0];
multiGauss = (1/(sqrt(determ*(2*pi)^3)))*exp((-1/2)*(x1-mu)*invers*(x1-mu)');



x2 = [5,5,5];
multiGauss2 = (1/(sqrt(determ*(2*pi)^3)))*exp((-1/2)*(x2-mu)*invers*(x2-mu)');

euclid_dist = norm(x2 - mu);

mahalon = sqrt((x2 - mu)*invers*(x2-mu)');
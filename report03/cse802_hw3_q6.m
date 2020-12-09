D = importdata('imox_data.txt');

%PCA
[eigen_vec, eigen_val] = eig(cov(D(:,1:8)));
projected_data = D(:,1:8)*eigen_vec(:,7:8);
hold on;
% scatter(projected_data(1:48,1), projected_data(1:48,2), 'filled', 'MarkerFaceColor', 'b');
% scatter(projected_data(49:96,1), projected_data(49:96,2), 'filled', 'MarkerFaceColor', 'r');
% scatter(projected_data(97:144,1), projected_data(97:144,2), 'filled', 'MarkerFaceColor', 'g');
% scatter(projected_data(145:192,1), projected_data(145:192,2), 'filled', 'MarkerFaceColor', 'y');
% xlabel('PCA projected feature value x1');
% ylabel('PCA projected feature value x2');
% title('PCA projected data for IMOX dataset using the two largest eigenvectors');
% legend('class I', 'class M', 'class O', 'class X');


mean_1 = mean(D(1:48,1:8));
mean_2 = mean(D(49:96,1:8));
mean_3 = mean(D(97:144,1:8));
mean_4 = mean(D(145:192,1:8));
mean_total = mean(D(:,1:8));

difference_matrix_1 = D(1:48,1:8) - mean_1;
difference_matrix_1 = difference_matrix_1';
between_difference_1 = mean_1 - mean_total;
between_difference_1 = between_difference_1';

difference_matrix_2 = D(49:96,1:8) - mean_2;
difference_matrix_2 = difference_matrix_2';
between_difference_2 = mean_2 - mean_total;
between_difference_2 = between_difference_2';

difference_matrix_3 = D(97:144,1:8) - mean_3;
difference_matrix_3 = difference_matrix_3';
between_difference_3 = mean_3 - mean_total;
between_difference_3 = between_difference_3';

difference_matrix_4 = D(145:192,1:8) - mean_4;
difference_matrix_4 = difference_matrix_4';
between_difference_4 = mean_4 - mean_total;
between_difference_4 = between_difference_4';

within_scatter_1 = zeros(8,8);
within_scatter_2 = zeros(8,8);
within_scatter_3 = zeros(8,8);
within_scatter_4 = zeros(8,8);

for i=1:48
   within_scatter_1 = within_scatter_1 + difference_matrix_1(:,i)*difference_matrix_1(:,i)';
   within_scatter_2 = within_scatter_2 + difference_matrix_2(:,i)*difference_matrix_2(:,i)';
   within_scatter_3 = within_scatter_3 + difference_matrix_3(:,i)*difference_matrix_3(:,i)';
   within_scatter_4 = within_scatter_4 + difference_matrix_4(:,i)*difference_matrix_4(:,i)'; 
end
within_scatter_total = within_scatter_1 + within_scatter_2 + within_scatter_3 + within_scatter_4;

between_scatter_1 = between_difference_1*between_difference_1';
between_scatter_2 = between_difference_2*between_difference_2';
between_scatter_3 = between_difference_3*between_difference_3';
between_scatter_4 = between_difference_4*between_difference_4';
between_scatter_total = between_scatter_1 + between_scatter_2 + between_scatter_3 + between_scatter_4;
between_scatter_total = 48*between_scatter_total;
scatter_matrix = cov(D(:,1:8))*191

discr = (inv(within_scatter_total))*between_scatter_total;

[eigen_vec_lda, eigen_val_lda] = eig(discr);
projected_data_lda = D(:, 1:8)*eigen_vec_lda(:,1:2);
scatter(projected_data_lda(1:48,1), projected_data_lda(1:48,2), 'filled', 'MarkerFaceColor', 'b');
scatter(projected_data_lda(49:96,1), projected_data_lda(49:96,2), 'filled', 'MarkerFaceColor', 'r');
scatter(projected_data_lda(97:144,1), projected_data_lda(97:144,2), 'filled', 'MarkerFaceColor', 'g');
scatter(projected_data_lda(145:192,1), projected_data_lda(145:192,2), 'filled', 'MarkerFaceColor', 'y');
xlabel('LDA projected feature value x1');
ylabel('LDA projected feature value x2');
title('LDA projected data for IMOX dataset using the two largest eigenvectors');
legend('class I', 'class M', 'class O', 'class X');

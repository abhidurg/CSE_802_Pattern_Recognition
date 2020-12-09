D = importdata('iris_data.txt');
D_training = [D(1:25, :); D(51:75,:); D(101:125, :)];
D_test = [D(26:50, :); D(76:100,:); D(126:150, :)];


k = 21;
knn_prediction = zeros(75, 1);
for x=1:75
    euclid_dist = zeros(75, 1);
    for i=1:75
        euclid_dist(i) = norm(D_test(x,1:4) - D_training(i,1:4));
    end
    [M, I] = mink(euclid_dist, k);
    neighbor_list = zeros(k, 1);
    for z=1:k
        neighbor_list(z) = D_training(I(z), 5);
    end
    frequency_list = histc(neighbor_list, [1,2,3]); %find the frequency of each class among neighbors
    max_frequency = max(frequency_list);
    index_max_frequency = find(frequency_list == max_frequency);
    %knn_prediction(x) = index_max_frequency(1);
    knn_prediction(x) = index_max_frequency(randperm(length(index_max_frequency), 1));
end
 
% C_knn = confusionmat(D_test(:,5), knn_prediction);
% confusionchart(C_knn);
% title('Confusion matrix of 21-NN for test set');

k_list = [1, 5, 9, 13, 17, 21];
accuracy = [71/75, 69/75, 71/75, 71/75, 72/75, 71/75];
plot(k_list, accuracy);
xlabel('Values of k tested');
ylabel('Classification accuracy');
title('Classifcation accuracy vs k for k-NN on iris dataset');
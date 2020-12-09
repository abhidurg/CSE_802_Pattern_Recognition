%part a
A = readmatrix('hw02_data01.txt');
%h = histogram(A);
h = histogram(A, 'Normalization', 'pdf', 'HandleVisibility', 'off');
h.BinWidth = 2;
%title("Histogram of feature x with bin size of 2");
xlabel("Feature x value");
title("Histogram of feature x with bin size of 2");
%ylabel("Frequency count");
ylabel("Frequency count, normalized to 1");

%part b
M = mean(A);
V = var(A,1); %biased variance

%part c
pd = makedist('Normal', 'mu', M, 'sigma', sqrt(V));
x = [0:1:100];
%x = [floor(min(A)):1:ceil(max(A))];
y = pdf(pd, x);
hold on;
plot(x,y, 'LineWidth', 1, 'DisplayName', 'Gaussian pdf of x');
legend;
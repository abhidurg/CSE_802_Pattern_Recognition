rng('default');
y1 = normrnd(20,sqrt(5),[1,1000]); %try 100, 500, and 1000
y2 = normrnd(35,sqrt(5),[1,1000]);
y = [y1, y2];
n = 2000;
h1 = 10;  %try for 0.01, 0.1, 1, and 10
%h_n = h1/sqrt(n);
h_n = h1;
p_n_list = zeros(1,55);
count = 1;
for x=0:1:55
    p_n = 0;
    for i=1:n
        p_n = p_n + (1/h_n)*normpdf((x-y(i))/h_n);
    end
    p_n = p_n/n;
    p_n_list(1, count) = p_n;
    count = count+1;
end

x = [0:1:55];
plot(x, p_n_list);
xlabel('x values from 0 - 55 with step size of 1');
ylabel('Probability density p(x)');
title('Probability density vs. x using parzen window and h = 10 for n = 2000');
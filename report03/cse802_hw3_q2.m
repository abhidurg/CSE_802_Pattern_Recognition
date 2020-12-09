syms theta
likelihood = 1/(theta^5);
likelihood2 = 0;
fplot(likelihood, [0.6,1], 'Color', 'b');
hold on;
fplot(likelihood2, [0,0.6],'Color', 'b');
ylim([0,60]);
title('Plot of likelihood vs. theta in range 0 - 1');
xlabel('theta');
ylabel('likelihood p(D|theta)');
legend('Uniform distribution 1/(theta^5)');
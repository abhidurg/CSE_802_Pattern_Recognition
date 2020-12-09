syms z
p_error = (1/2) - (1/pi)*atan(abs((1/2)*z));
fplot(p_error, [-2*pi,2*pi]);
f(z) = p_error;
title('Error rate for Cauchy distribution as a function of abs((a2-a1)/b)');
xlabel('Values of abs((a2-a1)/2)');
ylabel('Error rate');
%assume z = abs(a2 - a1)/b
%f(0) = 1/2;
%max P(error) is when a2 = a1. The error is 50%
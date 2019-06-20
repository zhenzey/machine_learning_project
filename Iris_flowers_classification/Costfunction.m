function [J, grad] = Costfunction(theta, x, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
% computes the cost and gradient using theta as the 
% parameter for logistic regression and the gradient of the cost
% size(theta) = [n + 1, 1]

m = length(y);

% logistic regression
z = x * theta;
h = sigmoid(z)'; % size(h) = [1, m]
J = 1 / m * (- log(h) * y - log(1 - h) * (1 - y));
grad = 1 / m *  x' *  (h' - y);


grad = grad(:);
end


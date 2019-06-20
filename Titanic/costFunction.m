function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost function and gradient for logisitc regression
%with regularization
%   J = COSTFUNCTION(theta, X, y, lambda) computes the cost of using theta
%   as the parameter for regularized logistic regression and the gradient
%   of the cost w.r.t. to the parameters


% Initialize some useful values
m = length(y); % number of training examples


% Compute cost function
h = sigmoid(transpose(X * theta));
J = 1 / m * (-log(h) * y - log(1 - h) * (1 - y)) + lambda / 2 / m * transpose(theta) * theta;

% Compute gradient
fprintf(num2str(size(lambda / m * theta)));
grad = 1 / m * transpose(X) * (transpose(h) - y) + lambda / m * theta;
end


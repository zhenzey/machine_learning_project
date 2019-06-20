function h = sigmoid(z)
%SIGMOID sigmoid function(vector implementation)
%   h = 1 / (1 + exp(-z))

h = 1 ./ (1 + exp(-z));
end


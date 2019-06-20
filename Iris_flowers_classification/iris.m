%% Machine learning practice --- Iris Flower Classification
%  Flower types : setosa, versicolor, virginica
%  Features: sepal length (cm); sepal width (cm); petal length (cm); petal width (cm)
%  Logistic regression


% ==================== Split data ======================
flid = fopen('iris.data');
data = textscan(flid, '%f %f %f %f %s', 'Delimiter',',');
X = [data{1}, data{2}, data{3}, data{4}];
Y_str = data{5};


% ===================== Scaling Y =======================
[m, n] = size(X);
Y = zeros(m, 1);
for i=1:m
    if strcmpi(Y_str{i},'Iris-setosa')
        Y(i) = 1;
    elseif strcmpi(Y_str{i},'Iris-versicolor')
        Y(i) = 2;
    elseif strcmpi(Y_str{i},'Iris-virginica')
        Y(i) = 3;
    end    
end

% ===================== Feature scaling =======================
Mean = mean(X, 1); 
for i=1:n
    X(:, i) = (X(:, i) - Mean(:, i)) / Mean(:, i);
end


X = [ones(m, 1), X];
% ============== Split test set and training set ==============
% split test set and training set according to the proportion (0.6: 0.4)
idx = randperm(m);
X = X(idx, :);
Y = Y(idx, :);
num_train = floor(0.6 * m);
X_train = X(1:num_train, :);
Y_train = Y(1:num_train, :);
X_test = X((num_train+1):end, :);
Y_test = Y((num_train+1):end, :);


% ============== Optimizing using fminunc =============
num_labels = 3;
all_theta = zeros(num_labels, n + 1);
initial_theta =  rand(n+1, 1);
for i = 1:num_labels 
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = ...
	fmincg(@(t)(Costfunction(t, X_train, (Y_train == i))), initial_theta, options);
    all_theta(i, :) = theta(:);
end


% ============== Accuracy =============

num_test = m - num_train;
pred = predict(all_theta, X_test);
fprintf('Test set accuracy: %f\n', sum((pred == Y_test)) / num_test)


         







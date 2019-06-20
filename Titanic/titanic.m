%% Machine learning project | TITANIC
% Overview and dataset of the problem can be found here: 
% https://www.kaggle.com/c/titanic/overview

%% Initialization 
clear; close all; clc

%% =========== Part 1: Loading data and processing =============

[X_tr, X_tt, Y_tr] = data_read();

%% ================= Part 2: Feature scaling ===================

X_tr = (X_tr - mean(X_tr, 1)) ./ mean(X_tr, 1);
X_tt = (X_tt - mean(X_tt, 1)) ./ mean(X_tt, 1);


%% ======= Part 3: Compute cost function and gradient ==========

[m, n] = size(X_tr);
X_tr = [ones(m, 1) X_tr];
initial_theta = rand(n + 1, 1);
%  Initialize lamda
lambda = 0.03;

%  Set options for fminunc optimization function
options = optimset('GradObj', 'on', 'MaxIter', 500);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X_tr, Y_tr, lambda)), initial_theta, options);

%% ================ Part 4: Compute accuacy =====================

pred_tr = predict(theta, X_tr);
fprintf("Accuracy of prediction in train set is: %f", sum(Y_tr == pred_tr) / m);

X_tt = [ones(size(X_tt, 1), 1) X_tt];
pred_tt = predict(theta, X_tt);
headers = {'PassengerId', 'Survived'};
PassengerId = linspace(892, 1309, 418)';
m = [PassengerId pred_tt];
csvwrite_with_headers('submission.csv',m,headers)




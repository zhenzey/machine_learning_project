function [X_tr, X_tt, Y_tr] = data_read()
%DATA_READ read and process the data 
%   read data from csv files, process them separately and transfer feature
%   values into numerical value(for both training set and test set)
%   There are 7 features extracted from the data: Pclass, Sex, Group of
%   family, Embark, Fare, Age, Number/Title
train_table = readtable("train.csv");
test_table = readtable("test.csv");
train_cell = table2cell(train_table);
test_cell = table2cell(test_table);

m1 = length(train_cell(:,1));
m2 = length(test_cell(:,1));

%% Missing values
TF_tr = ismissing(train_table);
TF_tt = ismissing(test_table);

%% Survival
Y_tr = cell2mat(train_cell(:, 2));

%% Pclass
class_tr = cell2mat(train_cell(:, 3));
class_tt = cell2mat(test_cell(:, 2));
% Calculate survival rate in training set
A = findgroups(class_tr);
B = findgroups(class_tr, Y_tr);
fprintf("Feature1: Class\n")
fprintf("Class 1 survival rate: %f\n", sum(B == 2) / sum(A == 1));
fprintf("Class 2 survival rate: %f\n", sum(B == 4) / sum(A == 2));
fprintf("Class 3 survival rate: %f\n", sum(B == 6) / sum(A == 3));

%% Sex
sex_tr = str2num(cell2mat(strrep(strrep(train_cell(:, 5), 'female', '0'), 'male', '1')));
sex_tt = str2num(cell2mat(strrep(strrep(test_cell(:, 4), 'female', '0'), 'male', '1')));
% Calculate survival rate in training set
A = findgroups(sex_tr);
B = findgroups(sex_tr, Y_tr);
fprintf("Feature2: Sex/gender\n");
fprintf("Male survival rate: %f\n", sum(B == 4) / sum(A == 2));
fprintf("Female survival rate: %f\n", sum(B == 2) / sum(A == 1));


%% SibSp && Parch(siblings/Spouse; Parents/Children)
sib_tr = cell2mat(train_cell(:, 7));
sib_tt = cell2mat(test_cell(:, 6));
par_tr = cell2mat(train_cell(:, 8));
par_tt = cell2mat(test_cell(:, 7));

% compare the survival rates when a person is alone or not
group_tr = sib_tr + par_tr;
group_tt = sib_tt + par_tt;
A = findgroups(group_tr);
B = findgroups(group_tr, Y_tr);

fprintf("Feature3: Sibsp\n");
fprintf("The survival rate(when the person is alone): %f\n", sum(B == 2) / sum(A == 1));
fprintf("The survival rate(when the person is not alone): %f\n", sum(B > 2 & mod(B, 2) == 0) / sum(A > 1));


%% Embark
em_tr = str2num(cell2mat(strrep(strrep(strrep(train_cell(:, 12), 'C', '0'), 'Q', '1'), 'S', '2')));
em_tt = str2num(cell2mat(strrep(strrep(strrep(test_cell(:, 11), 'C', '0'), 'Q', '1'), 'S', '2')));
% Calculate survival rate in training set
A = findgroups(em_tr);
B = findgroups(em_tr, Y_tr(TF_tr(:, 12) == 0)); %regradless of the missing value
fprintf("Feature4: Embark\n");
fprintf("Embark C survival rate: %f\n", sum(B == 2) / sum(A == 1));
fprintf("Embark Q survival rate: %f\n", sum(B == 4) / sum(A == 2));
fprintf("Embark S survival rate: %f\n", sum(B == 6) / sum(A == 3));
 % Guessing the missing value according to the propoertion
 prop_C = sum(B == 2) / length(A);
 prop_Q = sum(B == 4) / length(A);
 prop_S = 1 - prop_C - prop_Q;
 for i=1:m1
     if TF_tr(i, 12) == 1
         rand_num = rand(1);
         if rand_num < prop_C
             rand_sel = 0;
         elseif rand_num > prop_C && rand_num < prop_C + prop_Q
             rand_sel = 1;
         else
             rand_sel = 2;
         end
     em_tr = [em_tr(1: i - 1); rand_sel; em_tr(i: end)];
     end
 end
 
 %% Fare
fare_tr = cell2mat(train_cell(:, 10));
fare_tt = cell2mat(test_cell(:, 9));
% There is one missing value in test set, since the number is small, we
% just consider the missing fare as the mean value of fare
mean_fare = nanmean([fare_tr; fare_tt]);
std_fare = nanstd([fare_tr; fare_tt]);
for i=1:m2
    if TF_tt(i, 9) == 1
        fare_tt(i) = mean_fare;
    end
end
fprintf("Feature5: Fare\n");
fprintf("Mean value of fare: %f\n", mean_fare);
fprintf("Standard error of fare: %f\n", std_fare);



%% Age
age_tr = cell2mat(train_cell(:, 6));
age_tt = cell2mat(test_cell(:, 5));
mean_age = nanmean([age_tr; age_tt]);
std_age = nanstd([age_tr; age_tt]);

fprintf("Feature6: Age\n");
fprintf("Mean value of age: %f\n", mean_age);
fprintf("Standard error of age: %f\n", std_age);
% Fill out the missing value 
% We assume the age observes normal distribution
for i=1:m1
    if TF_tr(i, 6) == 1
        age_tr(i) = std_age * randn(1,1) + mean_age;
    end
end

for i=1:m2
    if TF_tt(i, 5) == 1
        age_tt(i) = std_age * randn(1,1) + mean_age;
    end
end



 %% Name
 % Split name to extract titles embeded
title_tr = strings([m1, 1]);
title_tt = strings([m2, 1]);
for i=1:m1
    name_str = train_cell{i, 4};
    split_str = strsplit(name_str, {', ', '.'});
    title_tr(i) = split_str(2);
end

for i=1:m2
    name_str = test_cell{i, 3};
    split_str = strsplit(name_str, {', ', '.'});
    title_tt(i) = split_str(2);
end

% Calculate the number of titles 
[A, title_type] = findgroups(title_tr);
fprintf('Feature 7: Name/Title\n');
fprintf('Title catergories:\n');
for i=1:length(title_type)
    num = sum(A == i);
    fprintf('%s: %i\n',title_type(i), num);
end
% Replace some rare titles(<10) with same category 'Rare'
% 'Dona' is not showing up in the train set
old = {'Capt', 'Col', 'Dona', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Mlle', ...
    'Mme', 'Ms', 'Rev', 'Sir', 'the Countess'};
title_tr_new = regexprep(title_tr, old, 'Rare');
title_tt_new = regexprep(title_tt, old, 'Rare');

A = findgroups(title_tr_new);
B = findgroups(title_tr_new, Y_tr);

fprintf('Master Survival rate: %f\n',  sum(B == 2) / sum(A == 1));
fprintf('Miss Survival rate: %f\n', sum(B == 4) / sum(A == 2));
fprintf('Mr Survival rate: %f\n', sum(B == 6) / sum(A == 3) );
fprintf('Mrs Survival rate: %f\n', sum(B == 8) / sum(A == 4));
fprintf('Rare Survival rate: %f\n', sum(B == 10) / sum(A == 5));

% Transfer string to numerical value
old = {'Master', 'Miss', 'Mrs', 'Mr', 'Rare'};
new = {'0', '1', '2', '3', '4'};
for i = 1:5
    title_tr_new = strrep(title_tr_new, old(i), new(i));
    title_tt_new = strrep(title_tt_new, old(i), new(i));
end
title_tr = str2double(title_tr_new);
title_tt = str2double(title_tt_new);

X_tr = [class_tr sex_tr group_tr em_tr fare_tr age_tr title_tr];
X_tt = [class_tt sex_tt group_tt em_tt fare_tt age_tt title_tt];

end


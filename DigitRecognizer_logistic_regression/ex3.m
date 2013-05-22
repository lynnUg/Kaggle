%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 784;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============


% Load Training Data
fprintf('Loading\n')

d1=csvread('train.csv');
X=d1(:,2:end);
y=d1(:,1);
m = size(X, 1);



fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============


fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('read in test file');
d2=csvread('test.csv');

%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, d2);
csvwrite('answer2.csv',pred(2:end));

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


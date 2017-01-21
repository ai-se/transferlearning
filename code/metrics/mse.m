function score = mse(actual, prediction)
%MSE   Computes the mean-squared error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
score = mean(se(actual,prediction));

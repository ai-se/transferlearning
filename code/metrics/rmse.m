function score = rmse(actual, prediction)
%RMSE   Computes the root mean-squared error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
score = sqrt(mse(actual, prediction));

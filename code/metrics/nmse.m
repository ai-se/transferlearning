function score = nmse(actual, prediction)
%NMSE   Computes the normalized mean-squared error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.

score = mse(actual, prediction)/var(prediction);

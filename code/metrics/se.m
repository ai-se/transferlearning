function score = se(actual, prediction)
%SE   Computes the squared error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
score = (actual(:)-prediction(:)).^2;

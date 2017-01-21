function score = sle(actual, prediction)
%MSLE   Computes the squared log error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
score = (log(1+actual(:))-log(1+prediction(:))).^2;

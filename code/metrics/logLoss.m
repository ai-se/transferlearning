function score = logLoss(actual, posterior)
%LOGLOSS   Computes the log loss
%   score = logLoss(actual, posterior)
%
%   actual is a binary vector
%   posterior is a vector of posterior probabilities that actual==1


score = mean(ll(actual, posterior));

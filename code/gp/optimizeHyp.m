function [gps fv1] = optimizeHyp(gps, x, y)
% optimization procedure for hyperparamters.
% Try previous hyper-parameter as well as a grid of hyper-parameters
% Input
%   gps: GP structure from previous step (used as initial state for the update)
%	gps.hyp: hyper parameter structure
%	gps.meanfunc: mean function
%	gps.covfunc: covariance function
%	gps.likfunc: likelihood function
%   x: (N x d) N observed x
%   y: (N x 1) corresponding y to x
%
% Output
%   gps: optimized parameter set
%   fv1: negative log evidence over iterations of conjugate gradient

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)


init_num_opt = 100;

% mnlh = nan(length(gps.hypgrid), 1); % save marginal negative log likelihood
% for k = 1:length(gps.hypgrid)
%     mnlh(k) = gp(gps.hypgrid(k), @infExact, gps.meanfunc, gps.covfunc, ...
% 		gps.likfunc, x, y);
% end
% [minMlh, idx] = min(mnlh);
% 
% % Test previous hyperparameter
% fv1 = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y);
% if fv1 >= minMlh
%     fprintf('Grid point is better than previous hyperparameter\n');
%     hyp = gps.hypgrid(idx);
% else
%     hyp = gps.hyp;
% end

% hyp = gps.hypgrid(idx);

% [hyp1 fv1] = minimize(gps.hyp, @gp, -init_num_opt, @infExact, gps.meanfunc, ...
% 		    gps.covfunc, gps.likfunc, x, y);
% gps.hyp = hyp1;


num_rep=5;

% optimize hyperparameters
for cnt_rep = 1:num_rep
    disp(['Number of rep: ',num2str(cnt_rep)]);

    % optimize hyperparameter
    [results.hyp{cnt_rep}] = minimize(gps.hyp, @gp, -init_num_opt, @infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y);

    % training
    results.nlml(cnt_rep) = gp(results.hyp{cnt_rep}, @infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y);
end
% find best  nlml
[results.nlml, best_hyp] =min(results.nlml);
gps.hyp = results.hyp{best_hyp};
fv1=results.nlml;

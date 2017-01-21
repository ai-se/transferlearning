function res = isCovarianceSPD(gps, x, varargin)
% isCovarianceSPD checks if covariance matrix is positive definite
  lenx = size(x,1);
  K = feval(gps.covfunc, gps.hyp.cov, x);
  sn2 = exp(2*gps.hyp.lik);
  % [L p] = chol(eye(lenx)+K);
  [dummy,p] = chol(K/sn2+eye(lenx));

  if ((nargin > 2) && strcmpi(varargin{1}, 'bool'))
    res = (p == 0);
  else
    res = (lenx - p);
  end

  % L = chol(eye(length(x_))+sW*sW'.*K);
  %
  % sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
  % L = chol(K/sn2+eye(n) + 0.0001*eye(n));               % Cholesky factor of covariance with noise
  % alpha = solve_chol(L,y-m)/sn2;
end


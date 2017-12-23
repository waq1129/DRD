function [wRidge,hprs,SDebars,postHess,logevid] = autoRegress_PoissonAR1(xx,yy,fnlin,nfilt,rhoNull,rhovals,avals,w0)
% [wRidge,hprs,SDebars,postHess,logevid] = autoRegress_PoissonAR1(xx,yy,fnlin,nfilt,rhoNull,rhovals,avals,w0)
%
% Computes empirical Bayes ridge regression filter estimate under a Poisson-GLM.
%
% Inputs:
%        xx [n x p] - stimulus (regressors)
%        yy [n x 1] - spike counts (response)
%     fnlin [1 x 1] - func handle for nonlinearity 
%		    (must return func + 1st & 2nd derivs, see ../nlfuns folder)
%     nfilt [1 x 1] - # coeffs to apply AR1 prior to (allows for separate DC weight prior)
%   rhoNull [1 x 1] - fixed prior precision for other columns of xx
%   rhovals [v x 1] - list of prior precisions to use for grid search
%        w0 [p x 1] - initial estimate of regression weights (optional)
%
% Outputs:
%   wRidge [p x 1] - empirical bayes estimate of weight vector
%      rho [1 x 1] - estimate for prior precision (inverse variance)
%  SDebars [p x 1] - 1 SD error bars from posterior
% postHess [p x p] - Hessian (2nd derivs) of negative log-posterior at
%                    maximum ( inverse of posterior covariance)
%  logevid [1 x 1] - log-evidence for hyperparameters
%
% Requires MATLAB optimization toolbox.
%
% $Id$

if ~isempty(yy)
    zaso = encapsulateRaw(xx, yy);
else
    zaso = xx;
end

nw = zaso.dimx; % length of stimulus filter

% initialize filter estimate with MAP regression estimate, if necessary
if nargin == 6
    rho0 = 5;        % initial value of ridge parameter
    Lprior = speye(nw)*rho0;
    [rsum, ragg] = zasoFarray(zaso, {@(x,y) x' * x, @(x,y) x' * y}, {});
    XX = rsum{1};
    XY = rsum{2};
    w0 = (XX + Lprior) \ XY;
end

% --- set prior and log-likelihood function pointers ---
mstruct.neglogli = @neglogli_poissGLM;
mstruct.logprior = @logprior_AR1;
mstruct.liargs = {zaso,fnlin};
mstruct.priargs = {nfilt,rhoNull};

% --- Do grid search over ridge parameter -----
[hprsMax,wmapMax] = gridsearch_GLMevidence(w0,mstruct,rhovals,avals);
fprintf('best grid point: rho=%.0f, alpha=%.2f\n', hprsMax);

% --- Do gradient ascent on evidence ----
[wRidge,hprs,logevid,postHess] = findEBestimate_GLM(wmapMax,hprsMax,mstruct);

if nargout >2
    SDebars = sqrt(diag(inv(postHess)));
end

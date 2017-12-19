function [kest,stats] = autoRidgeRegress_gradprimal(dd,hprnge)
% Automatic ridge regression using gradient and Hessian in primal space
%
% [kest,stats] = autoRidgeRegress_gradprimal(sdat,hprnge,opts,prs0)
%
% Empirical Bayes estimate of regression coefficients under ridge prior,
% with maximum marginal likelihood estimate of prior variance and noise
% variance.
%
% INPUTS:
% -------
%  sdat - data structure with fields:
%          .xx - stimulus autocovariance matrix X'*X
%          .xy - stimulus-response cross-covariance X'*Y
%          .yy - response variance Y'*Y
%          .ny - number of samples 
%
%  hprnge - structure with range for each hyperparameter (OPTIONAL):
%          .alpha - maximum of marginal variance for prior
%          .nsevar - variance of observation noise
% 
%  prs0 - initial setting for hyperparams  (OPTIONAL)
%         (If not provided, then will initialize with a grid search)
%
% OUTPUTS:
% --------
%   kest [nk x 1] - ridge regression estimate of regression weights
%   stats - struct with relevant fields
%
% Note: doesn't include a DC term, so should be applied when response and
% regressors have been standardized to have mean zero.


%% ========= Initialize range for hyperparameters ====================================
% Note: these ranges are purely heuristic. Please feel free to suggestx improvements

% marginal prior variance rho  (will convert later to alpha)
rhomax = 2*(dd.yy./dd.ny)/mean(diag(dd.xx)); % ratio of variance of output to intput
rhomin = min(1,.01*rhomax);  % variance of ridge regression estimate
hprnge.rho = [rhomin,rhomax];
        
% noise variance sigma_n^2
nsevarmax = dd.yy/dd.ny; % marginal variance of y
nsevarmin = min(1,nsevarmax*.01); % var ridge regression residuals
hprnge.nsevar = [nsevarmin, nsevarmax];


%% ========= Grid search for initial hyperparameters =============================

ngrid = 5;  % search a 5 x 5 grid for initial value of hyperparams
rnges = reshape(struct2array(hprnge),[],2); % ranges for each variable

lfun = @(prs)neglogev_ridgePrimal(prs,dd);  % loss function to optimize
[nllvals,gridpts] = grideval(ngrid,rnges,lfun);  % evaluate it on a grid
[hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2)); % find minimum


%% ========== Optimize evidence using fmincon ======================================
LB = [1e-2, 1e-2]';
UB = inf(2,1);

% Check Derivative & Hessian  (for debugging purposes - comment out later!)
% -------------------------------------------------------------------------
% Check elements of the derivative:
% DerivCheck_Elts(lfun,1,hprs0);  % check first element
% DerivCheck_Elts(lfun,2,hprs0);  % check 2nd element
% DerivCheck(lfun,hprs0); % check entire deriv
% HessCheck_Elts(lfun,[1 1], hprs0);  % check [1,1] entry of Hessian
% HessCheck_Elts(lfun,[1 2], hprs0);  % check [1,2] entry of Hessian
% HessCheck_Elts(lfun,[2 2], hprs0);  % check [2,2] entry of Hessian - argh, bug???
% HessCheck(lfun, hprs0);  % now fixed!

% fminopts = optimset('gradobj','on','Hessian','on','display','iter','algorithm','trust-region-reflective');
fminopts = optimset('gradobj','on','Hessian','on','display','off','algorithm','trust-region-reflective');
hprshat = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);


%% ------ compute posterior mean and covariance at maximizer of hyperparams--------------
[neglogEv,~,H,kest,Lpost] = lfun(hprshat); 

% Assemble summary statistics
stats.rho = hprshat(1); % rho hyperparameter 
stats.nsevar = hprshat(2); % noise variance
stats.alpha = 1/hprshat(1);  % 1/rho
stats.H = H;  % Hessian of hyperparameters
stats.neglogEv = neglogEv; % negative log-evidence at solution
stats.Lpost = Lpost;  % posterior covariance over filter
%stats.kpostSD = sqrt(diag(Lpost)); % 1SD posterior CI for hyperparams


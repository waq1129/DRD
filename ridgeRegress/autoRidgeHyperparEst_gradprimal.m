function hprshat = autoRidgeHyperparEst_gradprimal(dd,hprnge)
% hprshat = autoRidgeHyperparEst_gradprimal(dd,hprnge,K)
%
% Maximum marginal likelihood estimate of prior variance and noise
% variance for ridge regression.
%
% INPUTS:
% -------
%  dd - data structure with fields:
%          .xx - stimulus autocovariance matrix X'*X
%          .xy - stimulus-response cross-covariance X'*Y
%          .yy - response variance Y'*Y
%          .ny - number of samples 
%
%  hprnge - structure with range for each hyperparameter (OPTIONAL):
%          .alpha - maximum of marginal variance for prior
%          .nsevar - variance of observation noise
% 
%  K - Number of data partitions
%
% OUTPUTS:
% --------
%   hprshat [2 x 1] - Aggregated estimate of hyperparameters
% 
% updated 2015.04.21 (mca)



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
rnges = struct2cell(hprnge); % ranges for each variable
rnges = reshape([rnges{:}],2,2)';

lfun = @(prs)neglogev_ridgePrimal(prs,dd);  % loss function to optimize
[nllvals,gridpts] = grideval(ngrid,rnges,lfun);  % evaluate it on a grid
[hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2)); % find minimum


%% ========== Optimize evidence using fmincon ======================================
LB = [1e-2, 1e-2]';
UB = [inf, inf]';
% fminopts = optimset('gradobj','on','Hessian','on','display','iter','algorithm','trust-region-reflective');
fminopts = optimset('gradobj','on','Hessian','on','display','off','algorithm','trust-region-reflective');
hprshat = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);
function [khat,stats] = autoRidgeRegress_graddual_DNC(dd,hprnge,K)
% Automatic ridge regression using gradient and Hessian in Dual space by
% aggregated estimator as described in
% Lin, N & Xi, R. Aggregated estimating equation estimation.
% Statistics and Its Interfac. Vol. 4 (2011) 73-83

% In this version, we get the aggregated estimator of the hyperparameters
% and use that as a plug-in estimator for aggregated estimator of
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
%  K    - Number of subsets
%
% OUTPUTS:
% --------
%   kest [nk x 1] - ridge regression estimate of regression weights
%   stats - struct with relevant fields
%
% updated 2015.28.03 (mca)



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
dd.xx = dd.x'*dd.x;
lfun = @(prs)neglogev_ridgePrimal(prs,dd);  % loss function. Use primal form for now because its faster.
[nllvals,gridpts] = grideval(ngrid,rnges,lfun);  % evaluate it on a grid
[hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2)); % find minimum
%% ========== Optimize evidence by subsets using fmincon ======================================
LB = [1e-2, 1e-2]';
UB = [inf, inf]';
nsamps = dd.ny;

% ===Check number of subsets=======
% From Lin & Xi, Thm 4.2, the number of subsets must satisfy the following condition
if K>nsamps^(1/3)
    error('Number of partitions exceeds allowable maximum')
end

% fminopts = optimset('gradobj','on','Hessian','on','display','iter','algorithm','trust-region-reflective');
fminopts = optimset('gradobj','on','Hessian','on','display','off','algorithm','trust-region-reflective');
sampind = 1:nsamps;
hprs = zeros(2,K);
hprsTerms = hprs;
Hk = zeros(2,2,K);

%% Optimize hyperparameters
for k = 1:K
    
    % Draw random sample of data without replacement
    if floor(nsamps/K)<numel(sampind)
        [indk, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
        sampind(indsmap) = [];%remove samples for next round
    else
        indk = sampind;%otherwise use what is left of the sample
    end
    
    d.y = dd.y(indk);
    d.x = dd.x(indk,:);
    d.xx = dd.x(indk,:)*dd.x(indk,:)';
    d.xy = (dd.x(indk,:)'*dd.y(indk,:));
    d.yy = dd.y(indk,:)'*dd.y(indk,:);
    d.nx = dd.nx;
    d.ny = numel(indk);
    lfun = @(prs)neglogev_ridgeDual(prs,d);  % loss function to optimize
    hprshatk = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);
    [~,~,Hk(:,:,k),~,~] = lfun(hprshatk); %get Hessian
    hprs(:,k) = hprshatk;
    % Terms of weighted estimator
    hprsTerms(:,k) = -squeeze(Hk(:,:,k))*hprshatk;
end
SumA = -squeeze(sum(Hk,3));
hprshat = SumA\squeeze(sum(hprsTerms,3));
% hprshat = mean(hprs,2);

%% ------ compute posterior mean and covariance at maximizer of hyperparams--------------
Ak = zeros(d.nx,d.nx,K);
kestTerms = zeros(d.nx,K);
khat = kestTerms;
for k = 1:K
    
    % Draw random sample of data without replacement
    if floor(nsamps/K)<numel(sampind)
        [indk, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
        sampind(indsmap) = [];%remove samples for next round
    else
        indk = sampind;%otherwise use what is left of the sample
    end
    
    d.y = dd.y(indk);
    d.x = dd.x(indk,:);
    d.xx = dd.x(indk,:)*dd.x(indk,:)';
    d.xy = (dd.x(indk,:)'*dd.y(indk,:));
    d.yy = dd.y(indk,:)'*dd.y(indk,:);
    d.nx = dd.nx;
    d.ny = numel(indk);
    nx = d.nx;
    lfun = @(prs)neglogev_ridgeDual(prs,d);  % loss function to optimize
    hprshatk = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);
    [~,~,~,kest,Lpost] = lfun(hprshatk); %get Hessian
    
    % Make weight matrices
    cdiag = hprshatk(1)*ones(nx,1);  % diagonal of prior covariance
    nsevar = hprshatk(2);
    Ak(:,:,k) = (d.x'*d.x)/nsevar+spdiags(cdiag,0,d.nx,d.nx);
    % Terms of weighted estimator
    kestTerms(:,k) = -Ak*kest;
    khat0(:,k) = kest;
end
SumA = -squeeze(sum(Ak,3));
khat = SumA\squeeze(sum(kestTerms,3));

% Assemble summary statistics
stats.alpha = 1/hprshat(1);  % rho hyperparameter
stats.nsevar = hprshat(2); % noise variance
stats.H = H;  % Hessian of hyperparameters
stats.neglogEv = neglogEv; % negative log-evidence at solution
stats.Lpost = Lpost;  % posterior covariance over filter

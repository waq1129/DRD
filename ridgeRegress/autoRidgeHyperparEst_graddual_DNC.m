function [hprshat, hprs, Hk]  = autoRidgeHyperparEst_graddual_DNC(dd,K)
% Hyperparameter estimation by gradient and Hessian in Dual space by
% aggregated estimator as described in
% Lin, N & Xi, R. Aggregated estimating equation estimation.
% Statistics and Its Interfac. Vol. 4 (2011) 73-83
%
% INPUTS:
% -------
%  sdat - data structure with fields:
%          .x - design matrix X'*X
%          .y - response vector
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
%   hprshat [nparams x 1] - D&C estimate of hyperparameters
%   hprs    [nparams x K]  - Hyperpar estimates from each subset 
%   Hk      [nparams x nparams x K] - Hessian from each subset
% updated 2015.21.04 (mca)


%% ========= Initialize range for hyperparameters ====================================
% Note: these ranges are purely heuristic. Please feel free to suggestx improvements
% marginal prior variance rho  (will convert later to alpha)
rhomax = 2*(dd.y'*dd.y./dd.ny)/mean(diag(dd.x'*dd.x)); % ratio of variance of output to intput
rhomin = min(1,.01*rhomax);  % variance of ridge regression estimate
hprnge.rho = [rhomin,rhomax];
% noise variance sigma_n^2
nsevarmax = dd.y'*dd.y/dd.ny; % marginal variance of y
nsevarmin = min(1,nsevarmax*.01); % var ridge regression residuals
hprnge.nsevar = [nsevarmin, nsevarmax];
ngrid = 5;  % search a 5 x 5 grid for initial value of hyperparams
rnges = struct2cell(hprnge); % ranges for each variable
rnges = reshape([rnges{:}],2,2)';


LB = [1e-2, 1e-2]';
UB = [inf, inf]';
nsamps = dd.ny;
%% ========== Optimize evidence by subsets using fmincon ======================================

% ===Check number of subsets=======
% From Lin & Xi, Thm 4.2, the number of subsets must satisfy the following condition
if K>nsamps^(1/3)
    warning('Number of partitions exceeds allowable maximum')
end

fminopts = optimset('gradobj','on','Hessian','on','display','off','algorithm','trust-region-reflective');
sampind = 1:nsamps;
hprs = zeros(2,K);
hprsTerms = hprs;
Hk = zeros(2,2,K);

% Randomly assign samples to subsets
for k = 1:K
    if floor(nsamps/K)<numel(sampind)
        [indk{k}, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
        sampind(indsmap) = [];%remove samples for next round
    else
        indk{k} = sampind;%otherwise use what is left of the sample
    end
    
end

parfor k = 1:K
    % Draw random sample of data without replacement
    d  = struct('y', dd.y(indk{k}),'x', dd.x(indk{k},:),...
        'xx',dd.x(indk{k},:)*dd.x(indk{k},:)','nx',dd.nx,...
        'ny',numel(indk{k}));

    % ========= Grid search for initial hyperparameters =============================
    lfun = @(prs)neglogev_ridgeDual(prs,d); % loss function. 
    % lfun = @(prs)neglogev_ridgeDual(prs,dd);  % loss function to optimize
    [nllvals,gridpts] = grideval(ngrid,rnges,lfun);  % evaluate it on a grid
    [hprs0,~] = argmin(nllvals,gridpts(:,1),gridpts(:,2)); % find minimum
    
    % find optimal
    lfun = @(prs)neglogev_ridgeDual(prs,d);  % loss function to optimize
%     fprintf(' Estimating for subsample %2.0f of %2.0f\n',k,K);
    hprs(:,k) = fmincon(lfun,hprs0,[],[],[],[],LB,UB,[],fminopts);
    [~,~,Hk(:,:,k)] = lfun(hprs(:,k)); %get Hessian
    % Terms of weighted estimator
    hprsTerms(:,k) = -squeeze(Hk(:,:,k))*squeeze(hprs(:,k));
end

SumA = -squeeze(sum(Hk,3));
hprshat = SumA\squeeze(sum(hprsTerms,2));

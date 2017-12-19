function [hprshat, weights]  = autoRidge_dualDaCdualReg(dd,K)
% Script to automatically do empirical Bayes for ridge parameters and then
% do dual-form ridge regression.


fprintf('Estimating hyperparameters.\n')
[hprshat, ~, ~]  = autoRidgeHyperparEst_graddual_DNC(dd,K);
%% ------ Compute posterior mean and covariance at maximizer of evidence by divide and conquer --------
% rho = hprshat(1);
% nsevar = hprshat(2);
% 
% % Randomly assign samples to subsets
% for k = 1:K
%     if floor(nsamps/K)<numel(sampind)
%         [indk{k}, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
%         sampind(indsmap) = [];%remove samples for next round
%     else
%         indk{k} = sampind;%otherwise use what is left of the sample
%     end
%     
% end
% 
% % Estimate weights for each subset by Lin & Xi method
% parfor k = 1:K
%     d  = struct('y', dd.y(indk{k}),'x', dd.x(indk{k},:),...
%         'xx',dd.x(indk{k},:)*dd.x(indk{k},:)','nx',dd.nx,...
%         'ny',numel(indk{k}));
%     Hkmuk(k) = -d.x'*d.y/nsevar;% Covariance disappears here because posterior is Gaussian => Hk = -inv(Lambda), muk = Hk*X'*X*Y/nsevar
%     Hk(:,:,k) = % Don't know what to do here. The point is that this is too big.  Now we need to store K of them and invert?
% end

%% --------------Alternative DaC---------------------
fprintf('Estimating filter parameters.\n')

rho = hprshat(1);
nsevar = hprshat(2);

nsamps = dd.ny;
sampind = 1:nsamps;

% Randomly assign samples to subsets
for k = 1:K
    if floor(nsamps/K)<numel(sampind)
        [indk{k}, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
        sampind(indsmap) = [];%remove samples for next round
    else
        indk{k} = sampind;%otherwise use what is left of the sample
    end
    
end
% Estimate weights for each subset and average
muhatk = zeros(K,dd.nx);
rho = 2; % prior variance
nsevar = 5*5;   % stdev of added noise

% Estimate weights for each subset by Lin & Xi method
parfor k = 1:K
    d  = struct('y', dd.y(indk{k}),'x', dd.x(indk{k},:),...
        'xx',dd.x(indk{k},:)*dd.x(indk{k},:)','nx',dd.nx,...
        'ny',numel(indk{k}));
    % Estimate from dual form
    muhatk(k,:) =rho*d.x'*((rho*d.xx + nsevar/K*speye(d.ny))\d.y);% Dual form using Duncan-Guttman identity
%     muhatk(k,:) = ...
%(rho*speye(d.nx)-rho*rho*d.x'*((rho*d.x*d.x'+nsevar*speye(d.ny))\d.x))*d.x'*d.y/nsevar;%
%Dual form using Woodbury identity
% muhatk(k,:) = (rho*speye(d.nx) - d.x'*((d.xx/nsevar + rho*speye(d.ny))\d.x) )*d.x'*d.y/nsevar;
 
    W(k,:) = var(d.x,0,1);
end

% weights = mean(muhatk,1);
weights = sum(W.*muhatk,1)./sum(W,1);
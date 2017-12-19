function [neglogev,grad,H] = neglogev_ridgeDual3(prs,dat,K)
% Neg log-evidence for ridge regression model
% Ideally this code will construct the summed log likelihood of data
% partitioned into K sets.
% [neglogev,grad,H,mupost,Lpost] = neglogev_ridge(prs,dat)
%
% Computes negative log-evidence:
%    -log P(Y|X,sig^2,C)
% under linear-Gaussian model:
%       y = x'*w + n,    % linear observation model
%       n ~ N(0,sig^2),  % observation noise
%       w ~ N(0,rho*I),  % prior on weights
% Traditional ridge parameter is sig^2/rho
%
% INPUTS:
% -------
%  prs [2 x 1] - ridge model parameters [rho (marginal var); nsevar].
%                (also accepts struct with these fields)
%
%  dat - data structure with fields:
%        .xx - stimulus autocovariance matrix X'*X in Fourier domain
%        .xy - stimulus-response cross-covariance X'*Y in Fourier domain
%        .yy - response variance Y'*Y
%        .ny - number of samples
%
% OUTPUT:
% -------
%  neglogev [1 x 1] - negative marginal likelihood
%      grad [n x 1] - gradient
%         H [n x n] - hessian
%    mupost [n x 1] - posterior mean
%     Lpost [n x n] - posterior covariance
%
% Updated 2015.03.30 (mca)

% Unpack parameters
if isstruct(prs)
    rho = prs.rho;
    nsevar = prs.nsevar;
else
    rho = prs(1);
    nsevar = prs(2);
end
nllk = cell(1,K);
gradk = nllk;
Hk = nllk;


%% %Subsample data and create partitioned data matrices
[nsamps, nx] = size(dat.x);
sampind = 1:nsamps;
indk = cell(1,3);
for k = 1:K
    if floor(nsamps/K)<numel(sampind)
        %Sample observations without replacement
        [indk{k}, indsmap] = datasample(sampind,floor(nsamps/K), 'Replace',false);
        sampind(indsmap) = [];%remove samples for next round
    else
        indk{k} = sampind;%otherwise use what is left of the sample
    end
end
nout = nargout;
%% Evaluate
parfor k = 1:K
    
    % Assign data for kth subset
    d  = struct('y', dat.y(indk{k}),'x', dat.x(indk{k},:),...
        'xx',dat.x(indk{k},:)*dat.x(indk{k},:)',...
        'xy',dat.x(indk{k},:)'*dat.y(indk{k}),'nx',dat.nx,'ny',numel(indk{k}));
    Y = d.y;
    X = d.x;
    XX = d.xx;
    nx = d.nx;
    ny = d.ny;
    
    %Initialize covariance
    sigdiag = nsevar*ones(ny,1);
    sig = spdiags(sigdiag,0,ny,ny);
    
    if nout == 1 % Compute neglogli
        % dual form
        M = rho*XX + sig;
        trm1 = .5*logdet(rho*XX + sig);%log-determinant term
        trm2 = .5*Y'*(M\Y);%Quadratic term
        nllk{k} = trm1 + trm2;

    elseif nout == 2 % compute neglogli and gradient
        sig = spdiags(nsevar*ones(ny,1),0,ny,ny);
        M = rho*XX + sig;
        MY = M\Y;
        
        %  --- Compute neg-logevidence ---- primal form
        trm1 = .5*logdet(rho*XX + sig);%log-determinant term
        trm2 = .5*Y'*MY;%Quadratic term
        nllk{k} = trm1 + trm2;
        
        % --- Compute gradient ------------
        
        % Deriv w.r.t noise variance 'nsevar'
        MX = M\X;
        dLdthet = -.5*trace(MX*X') +.5*Y'*MX*MX'*Y;
        
        % Deriv w.r.t noise variance 'nsevar'
        RR = .5*(MY'*MY);
        Traceterm = -.5*sum(1./eig(M));
        dLdnsevar = Traceterm+RR;
        
        
        % Combine them into gardient vector
        gradk{k} = -[dLdthet; dLdnsevar];
        
    elseif nout == 3 % compute neglogli, gradient, & Hessian
        sig = spdiags(nsevar*ones(ny,1),0,ny,ny);
        M = rho*XX + sig;
        MY = M\Y;
        
        %  --- Compute neg-logevidence ---- primal form
        trm1 = .5*logdet(rho*XX + sig);%log-determinant term
        trm2 = .5*Y'*MY;%Quadratic term
        nllk{k} = trm1 + trm2 + ny*log(2*pi)/2;
        
        % --- Compute gradient ------------
        % Derivs w.r.t hyperparams rho and len
        MX = M\X;
        dLdthet = -.5*trace(MX*X') +.5*Y'*MX*MX'*Y;
        
        % Deriv w.r.t noise variance 'nsevar'
        RR = .5*(MY'*MY);
        Traceterm = -.5*sum(1./eig(M));
        dLdnsevar = Traceterm+RR;
        
        % Combine them into gardient vector
        gradk{k} = -[dLdthet; dLdnsevar];
        
        % --- Compute Hessian ------------
        
        % theta terms (rho and len)
        MXX = M\XX;
%         MMY = M\(MY);
        trm1 = .5*trace(MXX*MXX);%trace term
        trm2 = -Y'*MXX*MXX*MY;%quadratic term
        ddLddthet = trm1+trm2;
        
        % nsevar term
        trm1 = .5*sum(1./eig(M).^2);%trace term
        trm2 = -MY'*(M\MY);
        ddLdv = trm1 + trm2;
        
        % Cross term theta - nsevar
        trm1 = .5*trace(M\MXX');%trace term
        trm2 = -MY'*MXX*MY;%Quadratic term 1
        ddLdthetav = trm1 + trm2;
        Hk{k} = -unvecSymMtxFromTriu([ddLddthet;ddLdthetav; ddLdv]);
        
    end
end
if nout==1
    neglogev  = sum(cat(1,nllk{:}));
    % elseif nargout==3
    %     neglogev  = sum(cat(1,nllk{:}));
    %     grad = sum(cat(2,gradk{:}),2);
    %     H = sum(cat(3,Hk{:}),3);
else
    neglogev  = sum(cat(1,nllk{:}));
    grad = sum(cat(2,gradk{:}),2);
    H = sum(cat(3,Hk{:}),3);
    
end

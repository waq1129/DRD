function [kArd,logEv,alpha,nsevar] = runARDfull_prior(lam0,x,y,lambda,priortype,opts);
% [kArd,logEv,alpha,nsevar] = runARDfull(lam0,x,y,opts);
%
% Runs full ARD.  (i.e., computes ML value for the covariance matrix of an
% independent Gaussian prior)
%
% Inputs:
%   lam0 = initial value of the ratio of noise variance to prior variance
%          (ie. lam0 = nsevar*alpha)
%   nsevar0 - initial value of noise variance
%   x - design matrix  (each row is a single data vector)
%   y - dependent variable  (column vector)
%   opts - (optional) options stucture:  'maxiter' and 'tol'
%
% Outputs:
%   kArd - ARD estimate of kernel
%   logEv - log-evidence at ARD solution
%   alpha - estimate for inverse prior variance
%   nsevar - estimate for noise variance

% Check that options field is passed in
if nargin <= 5
    opts.maxiter = 1e4;
    opts.tol = 1e-4;
    %     fprintf('Setting options to defaults\n');
end

% ----- Initialize some stuff -------
[xlen,nx] = size(x);
uu = ones(nx,1);  % vector of ones
jcount = 1;  % counter
dparams = inf;  % Change in params from previous step
xx = x'*x;
xy = x'*y;

% ------ Initialize alpha & nsevar using MAP estimate around lam0 ------
kmap0 = (xx + lam0*eye(nx))\xy;
nsevar = var(y-x*kmap0);
alpha = lam0*uu/nsevar;

alpha2 = zeros(nx,1);

% ------ Run fixed-point algorithm for ARD  ------------
jjkeep = 1:nx; % keep track of eliminated dimensions
athresh = 1e8; % threshold for alpha, above which we remove
dds = [];
while (jcount <= opts.maxiter) & (dparams>opts.tol) & (~isempty(alpha))
    
    % Eliminate dimension j if alpha(j) grows too big
    jjtoobig = find(alpha>athresh);
    if ~isempty(jjtoobig)
        jjkeep(jjtoobig) = [];
        uu(jjtoobig) = [];
        x(:,jjtoobig) = [];
        xx(:,jjtoobig) = [];
        xx(jjtoobig,:) = [];
        xy(jjtoobig) = [];
        alpha(jjtoobig) = [];
        alpha2(jjtoobig) = [];
    end
    if isempty(alpha)
        break;
    end
    % Do ARD updates to remaining alphas
    CpriorInv = diag(alpha);
    [mu,Cprior] = compPostMeanVar([],[],nsevar,CpriorInv,xx,xy);
    switch priortype
        case 'exp'
            alpha2 = (uu + 2*lambda./alpha - alpha.*diag(Cprior))./(mu.^2+eps);
        case 'gamma'
            alpha2 = (uu + 2*lambda - alpha.*diag(Cprior))./(mu.^2+eps);
    end
    %     alpha2 = (uu +2*lambda)./(mu.^2+diag(Cprior)+eps);
    
    nsevar2 = sum((y-x*mu).^2)./(xlen-sum(uu-alpha.*diag(Cprior))+eps);
    
    % update counter, alpha & nsevar
    dparams = norm([alpha2;nsevar2]-[alpha;nsevar])/norm([alpha;nsevar]);
    dds = [dds; dparams];
    cc = determine_continue_value(dds);
    if cc>10
        break;
    end
    jcount = jcount+1;
    alpha = alpha2;
    nsevar = nsevar2;
end

if jcount < opts.maxiter
    fprintf('runARDfull: Finished in #%d steps\n', jcount)
elseif ~isempty(alpha)
    fprintf('runARDfull: MAXITER steps (%d) reached\n', jcount);
else
    fprintf('runARDfull: All variables eliminated!');
end

if isempty(alpha)
    kArd = zeros(nx,1);
    logEv = 0;
    return;
end
kArd = compPostMeanVar([],[],nsevar,diag(alpha),xx,xy);
logEv = compLogEv(CpriorInv,nsevar,xx,xy,y'*y,length(y));

% Reconstitute full (sparse) k, if necessary
if length(alpha) < nx
    k0 = kArd;
    a0 = alpha;
    kArd = zeros(nx,1);
    alpha = inf(nx,1);
    kArd(jjkeep) = k0;
    alpha(jjkeep) = a0;
end

fprintf('Number of zero-coeffs: %d (of %d) ll=logEv%.2f\n', length(find(kArd==0)),nx, logEv);



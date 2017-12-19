function [neglogev,grad,mupost,ii,cdiag] = neglogev_ASDspectral_dual(prs,dd,wwnrm,condthresh)
% Neg log-evidence for ASD regression model in Fourier domain
%
% [neglogev,grad,H,mupost,Lpost,ii] = neglogev_ASDspectral(prs,dd,wvecsq,opts)
%
% Computes negative log-evidence:
%    -log P(Y|X,sig^2,C)
% under linear-Gaussian model:
%       y = x'*w + n,  n ~ N(0,sig^2)
%       w ~ N(0,C),
% where C is ASD (or RBF or "squared exponential") covariance matrix
%
% INPUT:
% -------
%        prs [3 x 1] - ASD parameters [len; rho; nsevar]
%        dd [struct] - sufficient statistics for regression:
%                      .xx - Fourier domain stimulus autocovariance matrix X'*X
%                      .xy - Fourier domain stimulus-response cross-covariance X'*Y
%                      .yy - response variance Y'*Y
%                      .ny - number of samples
%      wwnrm [m x 1] - vector of normalized DFT frequencies, along each dimension
% condthresh [1 x 1] - threshold for condition number of K (Default = 1e8).
%
% OUTPUT:
% -------
%   neglogev - negative marginal likelihood
%   grad - gradient
%   H - Hessian
%   mupost - mean of posterior over regression weights
%   Lpost - posterior covariance over regression weights
%   ii - logical vector indicating which DFT frequencies are not pruned


% Unpack parameters
len = prs(1);
trho = prs(2);
nsevar = prs(3);

% Compute diagonal representation of prior covariance (Fourier domain)
ii = wwnrm < 2*log(condthresh)/len^2;
ni = sum(ii); % rank of covariance after pruning

% Build prior covariance matrix from parameters
% (Compute derivatives if gradient and Hessian are requested)
switch nargout
    case {0,1} % compute just diagonal of C
        cdiag = mkcovdiag_ASDstd(len,trho,wwnrm(ii)); % compute diagonal and Fourier freqs
        
    case 2
        % compute diagonal of C and deriv of C^-1
        [cdiag,dcinv] = mkcovdiag_ASDstd(len,trho,wwnrm(ii)); % compute diagonal and Fourier freqs
        
    otherwise
        % compute diagonal of C, 1st and 2nd derivs of C^-1 and C
        [cdiag,dcinv,dc,ddcinv] = mkcovdiag_ASDstd(len,trho,wwnrm(ii)); % compute diagonal and Fourier freqs
end

% Prune XX and XY Fourier coefficients and divide by nsevar
x = dd.x(ii,:)/sqrt(nsevar);
y = dd.y/sqrt(nsevar);
XY = x*y;

% Compute neglogli
XC = bsxfun(@times,x',cdiag');
XCX = XC*x;
In = eye(size(XCX));
logterm = real(logdetns(In+XCX))-sum(log(cdiag));
trm1 = -.5*(logterm + sum(log(cdiag)) + (dd.nsamps)*log(2*pi*nsevar)); % Log-determinant term
invXCXIn = (XCX+In)\In;
invterm = y'*(invXCXIn*(XCX*y));
trm2 = .5*(-dd.yy/nsevar + invterm);   % Quadratic term
neglogev = -trm1-trm2;  % negative log evidence
% Compute neglogli and Gradient
if nargout >= 2
    % Compute matrices we need
    %     Lpostinv = (XX+Cinv);
    %     Lpost = inv(Lpostinv);
    Lpdiag = cdiag-sum((XC'*invXCXIn).*XC',2);
    mupost = cdiag.*(x*(invXCXIn*y));
    
    % --- Compute gradient ------------
    % Derivs w.r.t hyperparams rho and len
    dLdthet = -.5*dcinv'*(cdiag - (Lpdiag + mupost.^2));
    % Deriv w.r.t noise variance 'nsevar'
    Xmu = x'*mupost;
    RR = .5*(dd.yy/nsevar - 2*mupost'*XY + Xmu'*Xmu)/nsevar; % Squared Residuals / 2*nsevar^2
    Tracetrm = .5*(ni-dd.nsamps-sum((Lpdiag./cdiag)))/nsevar;
    dLdnsevar = -Tracetrm-RR;
    % Combine them into gardient vector
    grad = [dLdthet; dLdnsevar];
end

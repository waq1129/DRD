function [neglogev,grad,H,mupost,Lpost] = neglogev_ridge(prs,dat)
% Neg log-evidence for ridge regression model
%
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
% Updated 2015.03.25 (jwp)

% Unpack parameters
if isstruct(prs)
    rho = prs.rho;
    nsevar = prs.nsevar;
else
    rho = prs(1);
    nsevar = prs(2);
end

nx = length(dat.xy);  % stimulus dimensionality
ny = dat.ny;  % number of samples
XX = dat.xx/nsevar;
XY = dat.xy/nsevar;

Cinv = spdiags(ones(nx,1)/rho,0,nx,nx); % inverse cov in diagonalized space

if nargout == 1 % Compute neglogli

    trm1 = -.5*(logdet(XX+Cinv) + nx*(log(rho)) + ny*log(2*pi*nsevar)); % Log-determinant term
    trm2 = .5*(-dat.yy/nsevar + XY'*((XX+Cinv)\XY));   % Quadratic term
    neglogev = -trm1-trm2;  % negative log evidence
    
elseif nargout == 2 % compute neglogli and gradient
    mdcinv = -(1/rho); % multiplier for deriv of C^-1 w.r.t theta (simplify!)
    Lpostinv = (XX+Cinv);
    Lpost = inv(Lpostinv);
    Lpdiag = diag(Lpost);
    mupost = Lpost*XY;
    
    % --- Compute neg-logevidence ----
    trm1 = -.5*(logdet(Lpostinv) + nx*(log(rho)) + (ny)*log(2*pi*nsevar));
    trm2 = .5*(-dat.yy/nsevar + XY'*Lpost*XY);    % Quadratic term    
    neglogev = -trm1-trm2;  % negative log evidence
    
    % --- Compute gradient ------------
    % Derivs w.r.t hyperparams rho and len
    dLdthet = -.5*mdcinv*(nx - sum(Lpdiag + mupost.^2)/rho);
    
    % Deriv w.r.t noise variance 'nsevar'
    RR = .5*(dat.yy/nsevar - 2*mupost'*XY + mupost'*XX*mupost)/nsevar; % Squared Residuals / 2*nsevar^2
    Tracetrm = .5*(nx-ny-sum(Lpdiag)/rho)/nsevar;
    dLdnsevar = -Tracetrm-RR;
    % Combine them into gardient vector
    grad = [dLdthet; dLdnsevar];

elseif nargout >= 3 % compute neglogli, gradient, & Hessian
    mdcinv = -(1/rho);  % multiplier for deriv of C^-1 w.r.t theta 
    mdc = (1/rho); % multiplier for deriv of C w.r.t theta 
    mddcinv = (2/rho^2); % multiplier for 2nd deriv of C^-1 w.r.t theta 
    
    Lpostinv = (XX+Cinv);
    Lpost = inv(Lpostinv);
    Lpdiag = diag(Lpost);
    mupost = Lpost*XY;
    
    % --- Compute neg-logevidence ----
    trm1 = -.5*(logdet(Lpostinv) + nx*(log(rho)) + (ny)*log(2*pi*nsevar));
    trm2 = .5*(-dat.yy/nsevar + XY'*Lpost*XY);    % Quadratic term    
    neglogev = -trm1-trm2;  % negative log evidence
    
    % --- Compute gradient ------------
    % Derivs w.r.t hyperparams rho and len
    dLdthet = -.5*mdcinv*(nx - sum(Lpdiag + mupost.^2)/rho);
    
    % Deriv w.r.t noise variance 'nsevar'
    RR = .5*(dat.yy/nsevar - 2*mupost'*XY + mupost'*XX*mupost)/nsevar;  % Squared Residuals / 2*nsevar^2
    Tracetrm = .5*(nx-ny-sum(Lpdiag)/rho)/nsevar;
    dLdnsevar = -Tracetrm-RR;

    % Combine them into gardient vector
    grad = [dLdthet; dLdnsevar];

    % --- Compute Hessian ------------
    % theta terms (rho and len)
    dLpdiag = -sum(Lpost.^2,2)*(mdcinv/rho); % Deriv of diag(Lpost) w.r.t thetas
    dmupost = -(Lpost*mupost)*(mdcinv/rho); % Deriv of mupost w.r.t thetas
    trm1stuff = -.5*(mdc - (dLpdiag + 2*dmupost.*mupost)/rho);
    ddLddthet_trm1 = sum(trm1stuff,1)*mdcinv;
    ddLddthet_trm2 = -.5*mddcinv*(nx - sum(Lpdiag + mupost.^2)/rho);
    ddLddthet = ddLddthet_trm1+ddLddthet_trm2;

    % nsevar term
    dLpdiagv = sum(Lpost.*(Lpost*XX),2)/nsevar; % Deriv of diag(Lpost) wr.t nsevar
    dmupostv = -(Lpost*mupost)/(rho*nsevar); % Deriv of mupost w.r.t nsevar
    ddLdv = -(dLdnsevar/nsevar - RR/(nsevar) ...
        - sum(dLpdiagv)/(2*rho*nsevar) ...
        + ((-XY+XX*mupost)'*dmupostv)/nsevar);  % 2nd deriv w.r.t. nsevar
        
    % Cross term theta - nsevar
    ddLdthetav = .5*mdcinv*sum((dLpdiagv+2*dmupostv.*mupost)/rho);
    
    % assemble Hessian 
    H = unvecSymMtxFromTriu([ddLddthet;ddLdthetav; ddLdv]);
    
end

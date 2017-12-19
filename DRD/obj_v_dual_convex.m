function [f, df] = obj_v_dual_convex(vv, eta, datastruct, bp, kdiag, G, opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [f, df, ddf] = obj_v_dual(u, datastruct, bp, kdiag, G ,opt)
%
% Objective function returns function value, gradient and hessian
%
% INPUTS:
%     vv [ld x 1] - input v vector
%     datastruct (structure)
%     bp [ld x 1] - projected vector from b
%     kdiag [ld x 1] - diagonal of covariance matrix in the frequency
%     domain
%     G - kronecker structure of Fourier basis
%     opt (structure) - contains various options
%
% OUTPUTS:
%     f - objective
%     df [ld x 1] - gradient with respect to vv
%     ddf [ld x ld] - hessian matrix of vv
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% unpack data and variable
X = datastruct.x;
% y = datastruct.y;

ld = length(vv); % frequency domain dimension
n = size(X,1); % sample size
d = length(opt.iikeep); % real domain dimension
nsevar = exp(datastruct.log_nsevar);

%% get cdiag
ufreq = vv.*sqrt(kdiag)+bp;
ureal = kronmulttrp(G,ufreq); % freq -> real
cdiag = nonlinear_u(ureal, opt, -opt.b);
cdiag_inf = nonlinear_u(ureal, opt, inf);
% plot(cdiag), drawnow

% truncate if need
cdiag(~opt.iikeep) = [];
ureal(~opt.iikeep) = [];
cdiag_inf(~opt.iikeep) = [];

%% --- Compute function --- %%
S = bsxfun(@times,X,cdiag')*X'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = inf;
    df = sparse(ld,1);
    return;
end
f = sum(eta./cdiag_inf)+logdetns(S)+vv'*vv; %loss function

%% --- Compute gradient --- %%
if nargout > 1
    % gradient for u
    gcdiag = grad_nonlinear_u(cdiag, ureal, opt, -opt.b);
    
    % compute quadratic term
    tmp = -eta./cdiag_inf;
    tmp1 = zeros(length(opt.iikeep),1);
    tmp1(opt.iikeep) = tmp;
    qtrm = kronmult(G,tmp1);
    
    % compute log-det term
    invS = S\eye(size(S));
    SinvX = invS*X;
    tmp = sum(SinvX.*X)'.*gcdiag;
    tmp1 = zeros(d,1);
    tmp1(opt.iikeep) = tmp;
    logdettrm = kronmult(G,tmp1);
    df_u = qtrm+logdettrm;
    
    df = df_u.*sqrt(kdiag)+vv*2;
end

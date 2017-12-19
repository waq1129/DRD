function [f, df, ddf] = obj_v_dual(vv, datastruct, bp, kdiag, G, opt)
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
%                      domain
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
y = datastruct.y;

ld = length(vv); % frequency domain dimension
n = size(X,1); % sample size
d = length(opt.iikeep); % real domain dimension
nsevar = exp(datastruct.log_nsevar); % noise variance

%% get cdiag
ufreq = vv.*sqrt(kdiag)+bp;
ureal = kronmulttrp(G,ufreq); % freq -> real
cdiag = nonlinear_u(ureal, opt, -opt.b);
% plot(cdiag), pause(0.1), drawnow

% truncate if need
cdiag(~opt.iikeep) = [];
ureal(~opt.iikeep) = [];

%% --- Compute function --- %%
S = bsxfun(@times,X,cdiag')*X'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = inf;
    df = sparse(ld,1);
    ddf = sparse(ld,ld);
    return;
end
invS = S\eye(size(S));
q = invS*y;
f = y'*q + logdetns(S)+vv'*vv; %loss function

%% --- Compute gradient --- %%
if nargout > 1
    % gradient for u
    gcdiag = grad_nonlinear_u(cdiag, ureal, opt, -opt.b);
    
    % compute quadratic term
    qX = q'*X;
    tmp = qX'.^2.*gcdiag;
    tmp1 = zeros(d,1);
    tmp1(opt.iikeep==1) = tmp;
    qtrm = kronmult(G,tmp1)';
    
    % compute log-det term
    SinvX = invS*X;
    tmp = sum(SinvX.*X)'.*gcdiag;
    tmp1 = zeros(d,1);
    tmp1(opt.iikeep==1) = tmp;
    logdettrm = kronmult(G,tmp1)';
    df_u = (-qtrm+logdettrm)';
    
    df = df_u.*sqrt(kdiag)+vv*2;
end

%% --- Compute Hessian ---- %%
% not using this part currently
% if nargout > 2
%     iikeep = opt.iikeep;
%     G = expand_kron(G)';
%     G = G(iikeep, :);
%     
%     % gradient for u
%     gcdiag = grad_nonlinear_u(cdiag, ureal, opt, -opt.b);
%     SinvX = invS*X;
%     
%     XS = SinvX';
%     XSy = XS*y;
%     
%     XSyXSy = XSy*XSy';
%     XSX = XS*X;
%     cc = gcdiag*gcdiag';
%     hcdiag = hess_nonlinear_u(cdiag, ureal, opt, -opt.b);
%     ddf1 = - XSX.*(cc).*(XSyXSy) + diag(hcdiag.*diag(XSyXSy));
%     ddf2 = XSX.*(cc).*(XSX') - diag(hcdiag.*diag(XSX));
%     ddf12 = G'*(ddf1+ddf2)*G;
%     ddf_u_u = ddf12;
%     ddf = -real(ddf_u_u).*(sqrt(kdiag)*sqrt(kdiag)')+eye(size(ddf_u_u));
% end

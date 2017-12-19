function [f, df, ddf] = obj_v_dual_sdrd(vv, datastruct, bp, kdiag, G, cfdiag, Gf, opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [f, df, ddf] = obj_v_dual_sdrd(u, datastruct, bp, kdiag, G, cfdiag, Gf, opt)
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
%     cfdiag [ldf x 1] - diagonal of covariance matrix in the frequency
%                        domain for smoothing kernel
%     Gf - kronecker structure of Fourier basis for smoothing kernel
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
% plot(cdiag),drawnow

% truncate if need
ureal(~opt.iikeep) = [];
cdiag(~opt.iikeep) = [];

%% --- Compute function --- %%
cdiag_half = sqrt(cdiag);
cfdiag_half = sqrt(cfdiag);

XCs = bsxfun(@times, X, cdiag_half');
XCs1 = zeros(size(XCs,1),length(opt.iikeep));
XCs1(:,opt.iikeep) = XCs;
XCs = XCs1;
XCsB = kronmult(Gf,XCs')';
XCsBCf = bsxfun(@times, XCsB, cfdiag_half');

S = XCsBCf*XCsBCf'+ nsevar*speye(n); % S matrix

if isinf(sum(S(:))) || isnan(sum(S(:))) 
    f = inf;
    df = sparse(ld,1);
    ddf = sparse(ld,ld);
    return;
end
invS = S\eye(size(S));
q = invS*y;
f = 0.5*y'*q + 0.5*logdetns(S)+0.5*vv'*vv; %loss function

%% --- Compute gradient --- %%
if nargout > 1
    % gradient for u
    gcdiag = grad_nonlinear_u(cdiag, ureal, opt, -opt.b);
    gcdiag_half = 0.5*gcdiag./cdiag_half;
    gcdiag_half(isnan(gcdiag_half)) = 0;
    X1 = kronmulttrp(Gf,bsxfun(@times,XCsBCf,cfdiag_half')')';
    X1 = X1(:,opt.iikeep);
    
    % compute quadratic term
    qX = q'*X;
    qX1 = q'*X1;
    tmp = qX'.*qX1'.*gcdiag_half;
    tmp1 = zeros(d,1);
    tmp1(opt.iikeep==1) = tmp;
    qtrm = kronmult(G,tmp1)'*2;
    
    % compute log-det term
    SinvX = invS*X;
    tmp = sum(SinvX.*X1)'.*gcdiag_half;
    tmp1 = zeros(d,1);
    tmp1(opt.iikeep==1) = tmp;
    logdettrm = kronmult(G,tmp1)'*2;
    
    df_u = (-qtrm+logdettrm)';
    df = 0.5*df_u.*sqrt(kdiag)+vv;
end

%% --- Compute Hessian ---- %%
% not using this part currently
if nargout>2
    Gnew = expand_kron(G)';
    Gnew = Gnew(opt.iikeep,:);
    cdiagnew = cdiag;
    urealnew = ureal;
    
    %% --- Compute function --- %%
    In = speye(n);
    S = XCsBCf*XCsBCf'+ nsevar*In; % S matrix
    invS = S\eye(size(S));
    
    %% --- Compute gradient --- %%
    % gradient for ufreq
    XS = X'*invS;
    XSy = XS*y;
    X1S = X1'*invS;
    X1Sy = X1S*y;
    
    gcdiag = grad_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    gcdiag_half = 0.5*gcdiag./cdiag_half;
    gcdiag_half(isnan(gcdiag_half)) = 0;
    
    %% --- Compute Hessian ---- %%
    % hessian for u_u
    hcdiag = hess_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    hcdiag_half = 0.5*hcdiag./cdiag_half-0.5*gcdiag.*gcdiag_half./cdiag;
    hcdiag_half(isnan(hcdiag_half)) = 0;
    
    tmp = gcdiag_half.*XSy;
    tmp1 = gcdiag_half.*X1Sy;
    Xtmp1 = bsxfun(@times, X, tmp1');
    X1tmp = bsxfun(@times, X1, tmp');
    Xtmp = Xtmp1+X1tmp;
    tmpX1S = bsxfun(@times, tmp, X1S);
    tmp1XS = bsxfun(@times, tmp1, XS);
    tmpXS = tmpX1S+tmp1XS;
    tmpa = hcdiag_half.*XSy.*X1Sy;
    
    tmp1 = zeros(length(opt.iikeep),1);
    tmp1(opt.iikeep) = tmp;
    gg = bsxfun(@times,kronmult(Gf,diag(tmp1)),cfdiag_half);
    tmpb = gg'*gg;
    tmpb = tmpb(opt.iikeep,opt.iikeep);
    
    ddf1 = -tmpXS*Xtmp + diag(tmpa) + tmpb;
    
    cXS = bsxfun(@times, gcdiag_half, XS);
    cX1S = bsxfun(@times, gcdiag_half, X1S);
    Xc = bsxfun(@times, X, gcdiag_half');
    X1c = bsxfun(@times, X1, gcdiag_half');
    
    cXSXc = cXS*Xc; % this step is slow
    cXSX1c = cXS*X1c; % this step is slow
    cX1SXc = cX1S*Xc; % this step is slow
    cX1SX1c = cX1S*X1c; % this step is slow
    
    XSX = XS*X; % this step is slow
    X1SX = X1S*X; % this step is slow
    XSX1 = XS*X1; % this step is slow
    X1SX1 = X1S*X1; % this step is slow
    
    tmp2 = hcdiag_half.*sum(X1S.*X',2);
    tmp1 = zeros(length(opt.iikeep),1);
    tmp1(opt.iikeep) = gcdiag_half;
    hh = bsxfun(@times,kronmult(Gf,diag(tmp1)),cfdiag_half);
    hh = hh(:,opt.iikeep);
    tmp3 = (hh'*hh).*XSX;
    
    ddf2 = 0.5*(cX1SX1c.*XSX+cX1SXc.*XSX1+cXSX1c.*X1SX+cXSXc.*X1SX1) - diag(tmp2) - tmp3; % this step is slow
    
    ddf = -ddf1-ddf2;
    
    %%
    L = Gnew'*ddf*Gnew;
    ddf = L.*(sqrt(kdiag)*sqrt(kdiag)')+eye(size(vv,1));
end
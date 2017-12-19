function L = DataLikelihood_Hessian_sdrd(ufreq, datastruct, G, cfdiag, Gf, opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% L = -\partial^2 Likelihood
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% unpack data and variables
X = datastruct.x;
X1 = zeros(size(X,1),length(opt.iikeep));
X1(:,opt.iikeep) = X;
X = X1;
y = datastruct.y;
n = size(X,1);
ld = length(ufreq);
nsevar = exp(datastruct.log_nsevar);

%% get cdiag
ureal = kronmulttrp(G,ufreq);
cdiag = nonlinear_u(ureal, opt, -opt.b);

%% --- Compute function --- %%
cdiag_half = sqrt(cdiag);
cfdiag_half = sqrt(cfdiag);

XCs = bsxfun(@times, X, cdiag_half');
XCsB = kronmult(Gf,XCs')';
XCsBCf = bsxfun(@times, XCsB, cfdiag_half');
X1 = kronmulttrp(Gf,bsxfun(@times,XCsBCf,cfdiag_half')')';

%% --- Compute function --- %%
In = speye(n);
S = XCsBCf*XCsBCf'+ nsevar*In; % S matrix
invS = S\eye(size(S));

%% prune dimensions given cdiag and opt.iikeep
iikeep = logical(opt.iikeep);
cdiagnew = cdiag(iikeep);
iikeepnew = truncateC(cdiagnew,opt);
iikeep0 = iikeep;
iikeep0(iikeep) = iikeepnew;
iikeep = iikeep0;

Xnew = X(:,iikeepnew);
X1new = X1(:,iikeepnew);
cdiagnew = cdiag(iikeep);
cdiag_halfnew = cdiag_half(iikeep);
urealnew = ureal(iikeep);

if sum(iikeep)<ld
    Gnew = expand_kron(G);
    Gnew = Gnew(:, iikeep)';
    Gfnew = expand_kron(Gf);
    Gfnew = Gfnew(:, iikeep)';
    
    %% --- Compute gradient --- %%
    % gradient for ufreq
    XS = Xnew'*invS;
    XSy = XS*y;
    X1S = X1new'*invS;
    X1Sy = X1S*y;
    
    gcdiag = grad_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    gcdiag_half = 0.5*gcdiag./cdiag_halfnew;
    gcdiag_half(isnan(gcdiag_half)) = 0;
    
    %% --- Compute Hessian ---- %%
    % hessian for u_u
    hcdiag = hess_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    hcdiag_half = 0.5*hcdiag./cdiag_halfnew-0.5*gcdiag.*gcdiag_half./cdiagnew;
    hcdiag_half(isnan(hcdiag_half)) = 0;
    
    tmp = gcdiag_half.*XSy;
    tmp1 = gcdiag_half.*X1Sy;
    Xtmp1 = bsxfun(@times, Xnew, tmp1');
    X1tmp = bsxfun(@times, X1new, tmp');
    Xtmp = Xtmp1+X1tmp;
    tmpX1S = bsxfun(@times, tmp, X1S);
    tmp1XS = bsxfun(@times, tmp1, XS);
    tmpXS = tmpX1S+tmp1XS;
    tmpa = hcdiag_half.*XSy.*X1Sy;
    
    gg = bsxfun(@times,Gfnew'*diag(tmp),cfdiag_half);
    tmpb = gg'*gg;
    
    ddf1 = -tmpXS*Xtmp + diag(tmpa) + tmpb;
    
    cXS = bsxfun(@times, gcdiag_half, XS);
    cX1S = bsxfun(@times, gcdiag_half, X1S);
    Xc = bsxfun(@times, Xnew, gcdiag_half');
    X1c = bsxfun(@times, X1new, gcdiag_half');
    
    cXSXc = cXS*Xc; % this step is slow
    cXSX1c = cXS*X1c; % this step is slow
    cX1SXc = cX1S*Xc; % this step is slow
    cX1SX1c = cX1S*X1c; % this step is slow
    
    XSX = XS*Xnew; % this step is slow
    X1SX = X1S*Xnew; % this step is slow
    XSX1 = XS*X1new; % this step is slow
    X1SX1 = X1S*X1new; % this step is slow
    
    tmp2 = hcdiag_half.*sum(X1S.*Xnew',2);
    hh = bsxfun(@times,Gfnew'*diag(gcdiag_half),cfdiag_half);
    tmp3 = (hh'*hh).*XSX;
    
    ddf2 = 0.5*(cX1SX1c.*XSX+cX1SXc.*XSX1+cXSX1c.*X1SX+cXSXc.*X1SX1) - diag(tmp2) - tmp3; % this step is slow
    
    ddf = -ddf1-ddf2;
    
    L = Gnew'*ddf*Gnew;
    
else
    Gnew = expand_kron(G);
    Gnew = Gnew(:, iikeep)';
    Gfnew = expand_kron(Gf);
    Gfnew = Gfnew(:, iikeep)';
    
    %% --- Compute gradient --- %%
    % gradient for ufreq
    XS = Xnew'*invS;
    XSy = XS*y;
    X1S = X1new'*invS;
    X1Sy = X1S*y;
    
    gcdiag = grad_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    gcdiag_half = 0.5*gcdiag./cdiag_halfnew;
    gcdiag_half(isnan(gcdiag_half)) = 0;
    
    %% --- Compute Hessian ---- %%
    % hessian for u_u
    hcdiag = hess_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    hcdiag_half = 0.5*hcdiag./cdiag_halfnew-0.5*gcdiag.*gcdiag_half./cdiagnew;
    hcdiag_half(isnan(hcdiag_half)) = 0;
    
    tmp = gcdiag_half.*XSy;
    tmp1 = gcdiag_half.*X1Sy;
    Xtmp1 = bsxfun(@times, Xnew, tmp1');
    X1tmp = bsxfun(@times, X1new, tmp');
    Xtmp = Xtmp1+X1tmp;
    tmpX1S = bsxfun(@times, tmp, X1S);
    tmp1XS = bsxfun(@times, tmp1, XS);
    tmpXS = tmpX1S+tmp1XS;
    tmpa = hcdiag_half.*XSy.*X1Sy;
    
    gg = bsxfun(@times,Gfnew'*diag(tmp),cfdiag_half);
    
    ddf1 = - (Gnew'*tmpXS)*(Xtmp*Gnew) + bsxfun(@times, Gnew', tmpa')*Gnew + (Gnew'*gg')*(gg*Gnew);
    
    cXS = bsxfun(@times, gcdiag_half, XS);
    cX1S = bsxfun(@times, gcdiag_half, X1S);
    Xc = bsxfun(@times, Xnew, gcdiag_half');
    X1c = bsxfun(@times, X1new, gcdiag_half');
    
    cXSXc = cXS*Xc; % this step is slow
    cXSX1c = cXS*X1c; % this step is slow
    cX1SXc = cX1S*Xc; % this step is slow
    cX1SX1c = cX1S*X1c; % this step is slow
    
    XSX = XS*Xnew; % this step is slow
    X1SX = X1S*Xnew; % this step is slow
    XSX1 = XS*X1new; % this step is slow
    X1SX1 = X1S*X1new; % this step is slow
    
    tmp2 = hcdiag_half.*sum(X1S.*Xnew',2);
    hh = bsxfun(@times,Gfnew'*diag(gcdiag_half),cfdiag_half);
    tmp3 = (hh'*hh).*XSX;
    
    ddf2 = 0.5*Gnew'*(cX1SX1c.*XSX+cX1SXc.*XSX1+cXSX1c.*X1SX+cXSXc.*X1SX1)*Gnew...
        - bsxfun(@times, Gnew', tmp2')*Gnew - Gnew'*tmp3*Gnew; % this step is slow
    
    ddf = -ddf1-ddf2;
    
    L = ddf;
    
end
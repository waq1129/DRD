function L = DataLikelihood_Hessian(ufreq, datastruct, G, opt)
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

%% prune dimensions given cdiag and opt.iikeep
iikeep = logical(opt.iikeep);
cdiagnew = cdiag(iikeep);
iikeepnew = truncateC(cdiagnew,opt);
iikeep0 = iikeep;
iikeep0(iikeep) = iikeepnew;
iikeep = iikeep0;

Xnew = X(:,iikeepnew);
cdiagnew = cdiag(iikeep);
urealnew = ureal(iikeep);

if sum(iikeep)<ld
    Gnew = expand_kron(G);
    Gnew = Gnew(:, iikeep)';
    
    %% --- Compute function --- %%
    C_mat = repmat(cdiagnew,1,n);
    In = eye(n);
    S = Xnew*(C_mat.*Xnew') + nsevar*In;
    invS = S\eye(size(S));
    
    %% --- Compute gradient --- %%
    % gradient for ufreq
    XS = Xnew'*invS;
    XSy = XS*y;
    gcdiag = grad_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    
    %% --- Compute Hessian ---- %%
    % hessian for u_u
    hcdiag = hess_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    
    tmp = gcdiag.*XSy;
    Xnewtmp = bsxfun(@times, Xnew, tmp');
    tmpXS = bsxfun(@times, tmp, XS);
    tmp1 = hcdiag.*XSy.*XSy;
    ddf1 = - tmpXS*Xnewtmp + 0.5*diag(tmp1);
    
    cXS = bsxfun(@times, gcdiag, XS);
    Xnewc = bsxfun(@times, Xnew, gcdiag');
    cXSXc = cXS*Xnewc; % this step is slow
    XSX = XS*Xnew; % this step is slow
    tmp2 = hcdiag.*sum(XS.*Xnew',2);
    ddf2 = 0.5*(cXSXc.*XSX) - 0.5*diag(tmp2); % this step is slow
    
    ddf12 = ddf1+ddf2;
    ddf_u_u = ddf12;
    ddf = -real(ddf_u_u);
    
    %%
    L = Gnew'*ddf*Gnew;
else
    Gnew = expand_kron(G);
    Gnew = Gnew(:, iikeep)';
    
    %% --- Compute function --- %%
    C_mat = repmat(cdiagnew,1,n);
    In = eye(n);
    S = Xnew*(C_mat.*Xnew') + nsevar*In;
    invS = S\eye(size(S));
    
    %% --- Compute gradient --- %%
    % gradient for ufreq
    XS = Xnew'*invS;
    XSy = XS*y;
    gcdiag = grad_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    
    %% --- Compute Hessian ---- %%
    % hessian for u_u
    hcdiag = hess_nonlinear_u(cdiagnew, urealnew, opt, -opt.b);
    
    tmp = gcdiag.*XSy;
    Xnewtmp = bsxfun(@times, Xnew, tmp');
    tmpXS = bsxfun(@times, tmp, XS);
    tmp1 = hcdiag.*XSy.*XSy;
    ddf1 = - (Gnew'*tmpXS)*(Xnewtmp*Gnew) + 0.5*bsxfun(@times, Gnew', tmp1')*Gnew;
    
    cXS = bsxfun(@times, gcdiag, XS);
    Xnewc = bsxfun(@times, Xnew, gcdiag');
    cXSXc = cXS*Xnewc; % this step is slow
    XSX = XS*Xnew; % this step is slow
    tmp2 = hcdiag.*sum(XS.*Xnew',2);
    ddf2 = 0.5*Gnew'*(cXSXc.*XSX)*Gnew - 0.5*bsxfun(@times, Gnew', tmp2')*Gnew; % this step is slow
    
    ddf12 = ddf1+ddf2;
    ddf_u_u = ddf12;
    
    %%
    L = -real(ddf_u_u);
end
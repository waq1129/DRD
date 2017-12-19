function f = obj_hyp(prs, nonprs, prsind, ureal, v, logkdiag_old, L, datastruct, fracdelta, opt)

%% unpack hypers
prss = [prs(:); nonprs(:)];
prssa = [prsind(:) prss(:)];
[~, ii] = sort(prssa(:,1));
prssb = prssa(ii,:);
prss = prssb(:,2);

rho = prss(1);
delta = prss(2);
log_nsevar = prss(3);
nsevar = exp(log_nsevar);

%% unpack data
X = datastruct.x;
y = datastruct.y;
n = size(X,1);

%% kernel covariance
logkdiag = mkcov_logASDfactored_nD(rho,delta,datastruct.nd,fracdelta,datastruct.nd(:),opt.cond);
kdiag = exp(logkdiag);

%% cdiag
cdiag = nonlinear_u(ureal,opt,-opt.b);
cdiag(~opt.iikeep) = [];

%% Data likelihood
S = bsxfun(@times,X,cdiag')*X'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    % quit if S has inf or nan
    f = 1e50;
    fprintf('flag hit1: inf or nan\n');
    return;
else
    % quit if S is ill-conditioned
    condS = cond(S);
    if condS > 1e10 
        f = 1e50;
        fprintf('flag hit2: ill-conditioned\n');
        return;
    end
end
invS = S\eye(size(S));
q = invS*y; % = inv(S)*y;
f0 = 0.5*(y'*q + logdetns(S));

%% GP prior
A = bsxfun(@times, L, kdiag')+eye(length(kdiag));
f1 = 0.5*logdetns(A);
if ~isreal(f1)
    % if f1 is complex, use SVD instead
    [~, ee, ~] = svdecon(A);
    f1 = 0.5*sum(log(diag(ee)));
end

f2 = 0.5*sum(a_exp_b(v.^2, logkdiag_old-logkdiag));

%%
f = f0+f1+f2;
if isinf(abs(f)) || isnan(abs(f)) || ~isreal(f)
    f = 1e50;
    return;
end


function f = obj_v_dual_mcmc(vv, log_nsevar, datastruct, b, kdiag, G, DCterm, DCmult, opt)
%% unpack data and variable
X = datastruct.x;
y = datastruct.y;
nd = datastruct.nd;

n = size(X,1); % sample size
nsevar = exp(log_nsevar);

%% get cdiag
bp = sparse(length(kdiag),1); bp(DCterm) = b*DCmult;
ufreq = vv.*sqrt(kdiag)+bp;
ureal = kronmulttrp(G,ufreq); % freq -> real
cdiag = nonlinear_u(ureal, opt, -opt.b);

% truncate if need
cdiag(~opt.iikeep) = [];

%% --- Compute function --- %%
S = bsxfun(@times,X,cdiag')*X'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = -inf;
    return;
end
invS = S\eye(size(S));
q = invS*y; % = inv(S)*y;
f = y'*q + logdetns(S); %loss function
f = -0.5*f;

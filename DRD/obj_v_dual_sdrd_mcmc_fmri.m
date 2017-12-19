function f = obj_v_dual_sdrd_mcmc_fmri(vv, log_nsevar, datastruct, b, kdiag, G, Gf, DCterm, DCmult, opt)
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

%% --- Compute function --- %%
cdiag_half = sqrt(cdiag);

% truncate if need
cdiag_half(~opt.iikeep) = [];

XCs = bsxfun(@times, X, cdiag_half');
XCsBCf = XCs*Gf';

S = XCsBCf*XCsBCf'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = -inf;
    return;
end

invS = S\eye(size(S));
q = invS*y; % = inv(S)*y;
f = 0.5*y'*q + 0.5*(logdetns(S))+0.5*vv'*vv; %loss function
f = -f;
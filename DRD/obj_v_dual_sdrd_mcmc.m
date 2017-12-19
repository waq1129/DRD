function f = obj_v_dual_sdrd_mcmc(vv, log_nsevar, datastruct, b, kdiag, G, len, logcfdiag_fun, DCterm, DCmult, opt)
%% unpack data and variable
X = datastruct.x;
y = datastruct.y;
nd = datastruct.nd;
n = size(X,1); % sample size
nsevar = exp(log_nsevar);

%% get cdiag
bp = sparse(prod(nd),1); bp(DCterm) = b*DCmult;
ufreq = vv.*sqrt(kdiag)+bp;
ureal = kronmulttrp(G,ufreq); % freq -> real
cdiag = nonlinear_u(ureal, opt, -opt.b);
[logcfdiag, ~, Gf] = logcfdiag_fun(len);
cfdiag = exp(logcfdiag);

%% --- Compute function --- %%
cdiag_half = sqrt(cdiag);
cfdiag_half = sqrt(cfdiag);

% truncate if need
cdiag_half(~opt.iikeep) = [];

XCs = bsxfun(@times, X, cdiag_half');
XCs1 = zeros(size(XCs,1),length(opt.iikeep));
XCs1(:,opt.iikeep) = XCs;
XCs = XCs1;
XCsB = kronmult(Gf,XCs')';
XCsBCf = bsxfun(@times, XCsB, cfdiag_half');

S = XCsBCf*XCsBCf'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = -inf;
    return;
end

invS = S\eye(size(S));
q = invS*y; % = inv(S)*y;
f = 0.5*y'*q + 0.5*(logdetns(S))+0.5*vv'*vv; %loss function
f = -f;
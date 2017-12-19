function f = obj_v_dual_sdrd_mcmc_hyp(vv, log_nsevar, datastruct, b, kdiag, G, Gf, DCterm, DCmult, opt, ssid)
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
if ssid==5 % doing slice sampling for len
    XCs = bsxfun(@times, X, cdiag_half');
    XCs1 = zeros(size(XCs,1),length(opt.iikeep));
    XCs1(:,opt.iikeep) = XCs;
    XCs = XCs1;
    XCsBCf = kronmult(Gf,XCs')';
else
    XCsBCf = XCs*Gf';
end
S = XCsBCf*XCsBCf'+ nsevar*speye(n); % S matrix
if isinf(sum(S(:))) || isnan(sum(S(:)))
    f = -inf;
    return;
end

invS = S\eye(size(S));
q = invS*y; % = inv(S)*y;
f = 0.5*y'*q + 0.5*(logdetns(S))+0.5*vv'*vv; %loss function
f = -f;
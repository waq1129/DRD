function cdiag = nonlinear_u(u, opt, m)
if nargin<3
    m = inf;
end
if strcmp(opt.nonlinearity,'exp')
    cdiag = exp(u)-exp(-m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
end
if strcmp(opt.nonlinearity,'rec') || strcmp(opt.nonlinearity,'rectifier')
    cdiag = loglogexp1(u,m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
end

if strcmp(opt.nonlinearity,'sqrt')
    cdiag = u.^2;
end

if strcmp(opt.nonlinearity,'hard')
    cdiag = u;
    cdiag(cdiag<=0) = 0;
end
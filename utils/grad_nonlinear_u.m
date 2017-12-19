function gcdiag = grad_nonlinear_u(cdiag, ureal, opt, m)
%% grad C_s
if nargin<4
    m = inf;
end

if strcmp(opt.nonlinearity,'exp')
    %     gcdiag = cdiag+exp(-m);
    cdiag = exp(ureal)-exp(-m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
    gcdiag = exp(ureal);
    gcdiag(ii) = 0;
end
if strcmp(opt.nonlinearity,'rec') || strcmp(opt.nonlinearity,'rectifier')
    %     gcdiag = 1-exp(-cdiag);
    [cdiag, tmp] = loglogexp1(ureal,m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
    gcdiag = 1-tmp;
    gcdiag(ii) = 0;
end

if strcmp(opt.nonlinearity,'sqrt')
    gcdiag = 2*ureal;
end
function hcdiag = hess_nonlinear_u(cdiag, ureal, opt, m)
%% hess C_s
if nargin<4
    m = inf;
end
if strcmp(opt.nonlinearity,'exp')
    %     hcdiag = cdiag+exp(-m);
    cdiag = exp(ureal)-exp(-m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
    hcdiag = exp(ureal);
    hcdiag(ii) = 0;
end
if strcmp(opt.nonlinearity,'rec')
    %     probit = 1-exp(-cdiag);
    [cdiag, tmp] = loglogexp1(ureal,m);
    [cdiag,ii] = hard_thresh(cdiag,opt);
    probit = 1-tmp;
    hcdiag = probit-probit.^2;
    hcdiag(ii) = 0;
end


if strcmp(opt.nonlinearity,'sqrt')
    hcdiag = 2*ones(length(cdiag),1);
end
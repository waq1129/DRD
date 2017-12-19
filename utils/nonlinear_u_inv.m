function u = nonlinear_u_inv(c, opt, m)
if nargin<3
    m = inf;
end
if strcmp(opt.nonlinearity,'exp')
    u = log(c+exp(-m));
end
if strcmp(opt.nonlinearity,'rec')
    u = log(exp(c)-1+exp(-m));
end

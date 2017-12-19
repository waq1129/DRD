function vnew = opt_convex(vv, datastruct, bp, kdiag, G, opt)
X = datastruct.x;
y = datastruct.y;
n = size(X,1); % sample size
nsevar = exp(datastruct.log_nsevar);
m = -opt.b;
P = nsevar*eye(n)-exp(-m)*X*X';
invP = P\eye(size(P));
maxit = 100;
vnew = vv;

options.Method='lbfgs';
options.TolFun=1e-4;
options.MaxIter = 1e2;
options.maxFunEvals = 1e2;
options.Display = 'off';

dd = [];
for it = 1:maxit
    cu = @(v,m) nonlinear_u(kronmulttrp(G,v.*sqrt(kdiag)+bp), opt, m);
    cdiag_inf = cu(vnew,inf);
    cdiag_inf(~opt.iikeep) = [];
    
    Sigma = bsxfun(@times,X,cdiag_inf')*X'*invP+eye(n);
    Sigma = Sigma\eye(size(Sigma));
    invPSigma = invP*Sigma;
    tmp = cdiag_inf.*(X'*(invPSigma*y));
    eta = tmp.^2;
    
    f_convex = @(vv) obj_v_dual_convex(vv, eta, datastruct, bp, kdiag, G, opt);
    vnew1 = minFunc(f_convex, vnew, options);
    dd = [dd; norm(cu(vnew1,m)-cu(vnew,m))/norm(cu(vnew,m))];
    vnew = vnew1;
    
    if dd(end)<1e-3
        break;
    end
end



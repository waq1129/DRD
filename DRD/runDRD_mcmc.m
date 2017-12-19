function [wsamps, usamps, hypers_estimation, w_dif, sq_er] = runDRD_mcmc(prs0,datastruct,lb,ub,iters,init_bound,truth)
% Initialization for hyperparameters
% 1 rho, 2 delta, 3 b, 4 log_nsevar, 5 len
lb([1,2,4,5]) = lb; lb(3) = -30;
ub([1,2,4,5]) = ub; ub(3) = 5;

nd = datastruct.nd;
if isempty(prs0)
    kmle = datastruct.x\datastruct.y; % get mle estimator
    rho0 = norm(kmle); % initial marginal variance
    delta0 = nd/10; % initial delta
    b0 = -12; % mean
    log_nsevar0 = 0; % initial log of noise variance
    hypers_init = [rho0, delta0, b0, log_nsevar0];
else
    b0 = -12; % mean
    hypers_init = prs0;
end
ub(1) = 1000;

% Set up for optimization
ind = [1,2,3,4]; % index of hypers for estimation: 1 rho, 2 delta, 3 b, 4 log_nsevar
nind = setdiff([1:4],ind); % index of hypers not for estimation
opt.cond = 1e12^(1/numel(nd)); % condthresh for optimization
opt.svMin = 1e-6; % threshold for cutting off cdiag
opt.nonlinearity = 'rec'; % choose the nonlinearity for transforming u
opt.iikeep = true(prod(nd),1); % elements to keep apriori
opt.th_c = 0; % flag for thresholding cdiag
opt.b = -inf;
DCmult = sqrt(prod(nd)); % factor to multiply by dc term
mindelta = lb(ind==2); % minimal delta
slice_width = 10; % for slice sampling

% Initialize with random value within bounds
if init_bound
    hypers_init(ind) = (ub(ind)-lb(ind)).*rand(length(lb(ind)),1) + lb(ind);
end
% Or initialize with true hypers
hypers_estimation = zeros(iters,4);
hypers_estimation(1,:) = hypers_init;

w_hat_old = inf*ones(sum(opt.iikeep),1);
sq_er = inf*ones(iters,1);
w_dif = inf*ones(iters,1);
vsamp = randn(prod(nd),1);
wsamps = zeros(iters,prod(nd));
usamps = zeros(iters,prod(nd));

%% priors
% prior for rho
subplot(221)
priorfun_r = @(x,pf) gen_gammaprior(x,20,10,lb(1),ub(1),pf); % dgrid,dmean,dstd,lb,ub,plotfig
rgrid = 0.01:0.5:50;
logrprior = priorfun_r(rgrid,1);

% prior for delta
subplot(222)
priorfun_d = @(x,pf) gen_gammaprior(x,100,50,lb(2),ub(2),pf);
dgrid = 0.01:1:200;
logdprior = priorfun_d(dgrid,1);

% prior for b
subplot(223)
priorfun_b = @(x,pf) gen_gaussprior(x,-10,8,lb(3),ub(3),pf);
bgrid = -30:0.5:5;
logbprior = priorfun_b(bgrid,1);

% prior for log_nsevar
subplot(224)
priorfun_n = @(x,pf) gen_gaussprior(x,-2,5,lb(4),ub(4),pf); % log nsevar
ngrid = -10:0.5:5;
lognprior = priorfun_n(ngrid,1);

%% The main loop
time = cputime;
for iter = 2:iters
    display([ 'iter: ' num2str(iter)])
    
    % Get hypers from the previous iteration
    rho = hypers_estimation(iter-1,1);
    delta = hypers_estimation(iter-1,2);
    b = hypers_estimation(iter-1,3);
    log_nsevar = hypers_estimation(iter-1,4);
    
    % Generate diagonal of Fourier-defined SE covariance (both K and b are generated in the frequency domain)
    [logkdiag, wnrm, G] = mkcov_logASDfactored_nD(rho,delta,nd,mindelta,nd(:),opt.cond); % G is Bfft
    kdiag = exp(logkdiag);
    DCterm = logical(prod(wnrm==0,2));
    bp = sparse(prod(nd),1); bp(DCterm) = b*DCmult; % b in the frequency domain
    
    % sample u in v space (ufreq=sqrt(kdiag).*v+bp, then transform ufreq to ureal)
    nu = randn(prod(nd),1);
    nLL = @(vv) obj_v_dual_mcmc(vv, log_nsevar, datastruct, b, kdiag, G, DCterm, DCmult, opt);
    vsamp_new = gppu_elliptical(vsamp, nu, nLL);
    ufreq_new = vsamp_new.*sqrt(kdiag)+bp; % get new ufreq
    ureal_new = kronmulttrp(G,ufreq_new); % get new ureal
    display(['old nLL: ' num2str(nLL(vsamp)) ' new nLL: ' num2str(nLL(vsamp_new))])
    vsamp = vsamp_new;
    usamps(iter,:) = ureal_new;
    
    % Calculate w_hat from ureal_new, w_hat is the MAP estimate
    X = datastruct.x;
    y = datastruct.y;
    n = size(X,1);
    cdiag = nonlinear_u(ureal_new(opt.iikeep==1),opt,-opt.b); % get new cdiag
    cdiag = abs(cdiag);
    nsevar = exp(hypers_estimation(iter-1,4)); % nsevar from the previous iter
    S = bsxfun(@times,X,cdiag')*X'+ nsevar*speye(n); % S matrix, S = XCX'+nsevar*I
    invS = S\eye(size(S));
    w_hat = bsxfun(@times,X,cdiag')'*invS*y; % derive w_hat in dual form
    w_hat1 = zeros(size(opt.iikeep)); w_hat1(opt.iikeep==1) = w_hat; % w_hat with the orignal length if truncated
    wsamps(iter,:) = w_hat1;
    
    %% Initialize hypers
    % hypers0 = (optub-optlb).*rand(length(optlb),1) + optlb;
    hypers0 = hypers_estimation(iter-1,:); % hypers for estimation
    hypers_new = update_theta(hypers0, vsamp, ...
        @(vv, kdiag, b, log_nsevar) obj_v_dual_mcmc(vv, log_nsevar, datastruct, b, kdiag, G, DCterm, DCmult, opt), ...
        @(rho, delta) mkcov_logASDfactored_nD(rho,delta,nd,mindelta,nd(:),opt.cond), ...
        @(theta) theta_log_prior(theta, priorfun_r, priorfun_d, priorfun_b, priorfun_n),...
        slice_width, -inf, ind);
    
    % collect new hypers
    hypers_estimation(iter,:) = hypers_new;
    sq_er(iter) = norm(datastruct.y-datastruct.x*w_hat)/norm(datastruct.y);
    w_dif(iter) = norm(w_hat-w_hat_old);
    w_hat_old = w_hat;
    
    display(['rho: ' num2str(hypers_estimation(iter,1)) ...
        ' delta: ' num2str(hypers_estimation(iter,2)) ...
        ' b: ' num2str(hypers_estimation(iter,3)) ...
        ' nsevar: ' num2str(exp(hypers_estimation(iter,4))) ...
        ' w dif: ' num2str(w_dif(iter)) ' sq_er: ' num2str(sq_er(iter))])
    
    %% Plot w_hat, ureal and sqrt(cdiag), compared with true values, as well
    % as posterior distributions for hypers
    subplot(321);
    plot_post(1,hypers_estimation,logrprior,rgrid,'rho',iter)
    
    subplot(322);
    plot_post(2,hypers_estimation,logdprior,dgrid,'delta',iter)
    
    subplot(323);
    plot_post(3,hypers_estimation,logbprior,bgrid,'b',iter)
    
    subplot(324);
    plot_post(4,hypers_estimation,lognprior,ngrid,'lognsevar',iter)
    
    subplot(325);
    t = 1:prod(nd);
    plot(t,truth.c_true,'b-',t,nonlinear_u(ureal_new,opt,-opt.b),'r-','linewidth',3);
    title(sprintf('current GP sample c (iter %d)',iter))
    xlabel('c'); drawnow
    
    subplot(326);
    plot(t,truth.w_true,'b-',t,w_hat1,'r-');
    title(sprintf('current GP sample w (iter %d)',iter))
    xlabel('w'); drawnow
    
end
time = cputime-time;

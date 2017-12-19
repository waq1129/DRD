function [wsamps, usamps, hypers_estimation, w_dif, sq_er] = runsDRD_fmri(prs0,datastruct,iters,init_bound)
% Initialization for hyperparameters
% 1 rho, 2 delta, 3 b, 4 log_nsevar, 5 len
nd = datastruct.nd;
mindelta = 1; maxdelta = min(nd); minl = 2; maxl = min(nd);
lb = [0.001;mindelta;-20;-5;minl]; ub = [1e5;maxdelta;5;10;maxl]; % bounds for [rho, delta, log_nsevar len]

if isempty(prs0)
    kmle = datastruct.x\datastruct.y; % get mle estimator
    rho0 = norm(kmle); % initial marginal variance
    delta0 = 10; % initial delta
    b0 = -10; % mean
    log_nsevar0 = 0; % initial log of noise variance
    len0 = 2; % initial length scale for smoothing kernel
    hypers_init = [rho0, delta0, b0, log_nsevar0, len0];
else
    b0 = -10; % mean
    hypers_init = prs0;
end
ub(1) = 1000;

% Set up for optimization
ind = [1,2,3,4]; % index of hypers for estimation: 1 rho, 2 delta, 3 b, 4 log_nsevar, 5 len
nind = setdiff([1:5],ind); % index of hypers not for estimation
opt.cond = 1e12^(1/numel(nd)); % condthresh for optimization
opt.cond_smooth = 1e3^(1/numel(nd)); % condthresh for optimization
opt.svMin = 1e-6; % threshold for cutting off cdiag
opt.nonlinearity = 'rec'; % choose the nonlinearity for transforming u
opt.iikeep = logical(datastruct.maskdata(:));
opt.th_c = 0; % flag for thresholding cdiag
opt.b = -inf;
DCmult = sqrt(prod(nd)); % factor to multiply by dc term
mindelta = lb(ind==2); % minimal delta
slice_width = 10; % for slice sampling
frac = 0.99;

% Initialize with random value within bounds
if init_bound
    hypers_init(ind) = (ub(ind)-lb(ind)).*rand(length(lb(ind)),1) + lb(ind);
end
% Or initialize with true hypers
hypers_estimation = zeros(iters,5);
hypers_estimation(1,:) = hypers_init;

w_hat_old = inf*ones(sum(opt.iikeep),1);
sq_er = inf*ones(iters,1);
w_dif = inf*ones(iters,1);
vsamp = randn(prod(nd),1);
wsamps = zeros(iters,prod(nd));
usamps = zeros(iters,prod(nd));

%% priors
% prior for rho
subplot(321)
priorfun_r = @(x,pf) gen_gammaprior(x,20,10,lb(1),ub(1),pf); % dgrid,dmean,dstd,lb,ub,plotfig
rgrid = 0.01:0.5:50;
logrprior = priorfun_r(rgrid,1);

% prior for delta
subplot(322)
priorfun_d = @(x,pf) gen_gammaprior(x,100,50,lb(2),ub(2),pf);
dgrid = 0.01:1:200;
logdprior = priorfun_d(dgrid,1);

% prior for b
subplot(323)
priorfun_b = @(x,pf) gen_gaussprior(x,-10,8,lb(3),ub(3),pf);
bgrid = -30:0.5:5;
logbprior = priorfun_b(bgrid,1);

% prior for log_nsevar
subplot(324)
priorfun_n = @(x,pf) gen_gaussprior(x,-2,5,lb(4),ub(4),pf);
ngrid = -10:0.5:5;
lognprior = priorfun_n(ngrid,1);

% prior for len
subplot(325)
priorfun_l = @(x,pf) gen_gammaprior(x,100,50,lb(5),ub(5),pf);
lgrid = 0.01:0.5:200;
loglprior = priorfun_l(lgrid,1);

%% The main loop
time = cputime;
for iter = 2:iters
    display([ 'iter: ' num2str(iter)])
    
    % Get hypers from the previous iteration
    rho = hypers_estimation(iter-1,1);
    delta = hypers_estimation(iter-1,2);
    b = hypers_estimation(iter-1,3);
    log_nsevar = hypers_estimation(iter-1,4);
    len = hypers_estimation(iter-1,5);
    
    % Generate diagonal of Fourier-defined SE covariance (both K and b are generated in the frequency domain)
    [logkdiag, wnrm, G] = mkcov_logASDfactored_nD(rho,delta,nd,mindelta,nd(:),opt.cond); % G is Bfft
    kdiag = exp(logkdiag); ld = length(kdiag);
    DCterm = logical(prod(wnrm==0,2));
    bp = sparse(ld,1); bp(DCterm) = b*DCmult; % b in the frequency domain
    
    % Generate frequency covariance
    logcfdiag_fun = @(len) mkcov_logASDfactored_nD(1,len,nd,frac*len,nd(:),opt.cond_smooth);
    if iter == 2 || sum(ismember(ind,5))>0 % optimize len
        display('opt len')
        [logcfdiag, ~, Gf] = logcfdiag_fun(len);
        cfdiag = exp(logcfdiag);
        Gf = expand_kron(Gf); Gf = bsxfun(@times, Gf(:,opt.iikeep), cfdiag.^0.5);
    end
    
    % sample u in v space (ufreq=sqrt(kdiag).*v+bp, then transform ufreq to ureal)
    nu = randn(ld,1);
    nLL = @(vv) obj_v_dual_sdrd_mcmc_fmri(vv, log_nsevar, datastruct, b, kdiag, G, Gf, DCterm, DCmult, opt);
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
    nsevar = exp(hypers_estimation(iter-1,4)); % nsevar from the previous iter
    cdiag = nonlinear_u(ureal_new(opt.iikeep==1),opt,-opt.b); % get new cdiag
    cdiag_half = sqrt(abs(cdiag));
    XCs = bsxfun(@times, X, cdiag_half');
    XCsBCf = XCs*Gf';
    S = XCsBCf*XCsBCf'+ nsevar*speye(n); % S matrix
    invS = S\eye(size(S));
    X1 = XCsBCf*Gf;
    w_hat = bsxfun(@times,X1,cdiag_half')'*invS*y; % derive w_hat in dual form
    w_hat1 = zeros(size(opt.iikeep)); w_hat1(opt.iikeep==1) = w_hat; % w_hat with the orignal length if truncated
    wsamps(iter,:) = w_hat1;
    
    %% Initialize hypers
    % hypers0 = (optub-optlb).*rand(length(optlb),1) + optlb;
    hypers0 = hypers_estimation(iter-1,:); % hypers for estimation
    hypers_new = update_theta_sdrd_fmri(hypers0, vsamp, Gf, logcfdiag_fun, ...
        @(vv, kdiag, b, log_nsevar, Gf, ssid) obj_v_dual_sdrd_mcmc_hyp(vv, ...
        log_nsevar, datastruct, b, kdiag, G, Gf, DCterm, DCmult, opt, ssid), ...
        @(rho, delta) mkcov_logASDfactored_nD(rho,delta,nd,mindelta,nd(:),opt.cond), ...
        @(theta) theta_log_prior_sdrd(theta, priorfun_r, priorfun_d, priorfun_b, priorfun_n, priorfun_l),...
        slice_width, -inf, opt, ind);
    
    % collect new hypers
    hypers_estimation(iter,:) = hypers_new;
    sq_er(iter) = norm(datastruct.y-datastruct.x*w_hat)/norm(datastruct.y);
    w_dif(iter) = norm(w_hat-w_hat_old);
    w_hat_old = w_hat;
    
    display(['rho: ' num2str(hypers_estimation(iter,1)) ...
        ' delta: ' num2str(hypers_estimation(iter,2)) ...
        ' b: ' num2str(hypers_estimation(iter,3)) ...
        ' nsevar: ' num2str(exp(hypers_estimation(iter,4))) ...
        ' len: ' num2str(hypers_estimation(iter,5)) ...
        ' w dif: ' num2str(w_dif(iter)) ' sq_er: ' num2str(sq_er(iter))])
    
    %% Plot w_hat, ureal and sqrt(cdiag), compared with true values, as well
    % as posterior distributions for hypers
    subplot(421);
    plot_post(1,hypers_estimation,logrprior,rgrid,'rho',iter)
    
    subplot(422);
    plot_post(2,hypers_estimation,logdprior,dgrid,'delta',iter)
    
    subplot(423);
    plot_post(3,hypers_estimation,logbprior,bgrid,'b',iter)
    
    subplot(424);
    plot_post(4,hypers_estimation,lognprior,ngrid,'lognsevar',iter)
    
    subplot(425);
    plot_post(5,hypers_estimation,loglprior,lgrid,'len',iter)
    
    subplot(427);
    plot(cdiag,'r-');
    title(sprintf('current GP sample c (iter %d)',iter))
    xlabel('c'); drawnow
    
    subplot(428);
    plot(w_hat,'r-');
    title(sprintf('current GP sample w (iter %d)',iter))
    xlabel('w'); drawnow
    
end
time = cputime-time;

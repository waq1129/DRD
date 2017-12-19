%% test_fastASD_1D.m
%
% Short script to illustrate fast automatic smoothness determination (ASD)
% for vector of filter coefficients

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk = 1000;  % number of filter coeffs (1D vector)
rho = 2; % marginal variance
len = 25;  % ASD length scale

C0 = mkcov_ASD(len,rho,nk); % prior covariance matrix 
k = mvnrnd(zeros(1,nk),C0)'; % sample k from mvnormal with this covariance

%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 10;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(211); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');

subplot(212); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');

%% Compute ridge regression estimate 
fprintf('\n...Running ridge regression with fixed-point updates...\n');

% Sufficient statistics (old way of doing it, not used for ASD)
dd.xx = x'*x;   % stimulus auto-covariance
dd.xy = (x'*y); % stimulus-response cross-covariance
dd.yy = y'*y;   % marginal response variance
dd.nx = nk;     % number of dimensions in stimulus
dd.ny = nsamps;  % total number of samples

% Run ridge regression using fixed-point update of hyperparameters
maxiter = 100;
tic;
kridge = autoRidgeRegress_fp(dd,100);
toc;

%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');

% Set lower bound on length scale.  (Larger -> faster inference).
% If it's set too high, the function will warn that optimal length scale is
% at this lower bound, and you should re-run with a smaller value since
% this may be too high
minlen = 20; 

% Run ASD
tic;
[kasd,asdstats] = fastASD(x,y,nk,minlen);
toc;

%%  ---- Make Plots ----
subplot(211);
h = plot(t,k,'k-',t,kridge);
set(h(1),'linewidth',2.5); 
title('ridge estimate');
legend('true', 'ridge');

subplot(212);
kasdSD = sqrt(asdstats.Lpostdiag); % posterior stdev for asd estimate
plot(t,k,'k-',t,kasd,'r'); hold on;
errorbarFill(t,kasd,2*kasdSD); % plot posterior marginal confidence intervals
hold off;
legend('true','ASD')
title('ASD estimate (+/- 2SD)');

% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n----------------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len,asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho,asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n------\n  Ridge = %7.2f\n  ASD = %9.2f\n\n', [err(kridge) err(kasd)]);

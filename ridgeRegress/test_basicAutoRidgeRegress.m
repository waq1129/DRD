%% test_basicAutoRidgeRegress.m
%
% Script to illustrate two versions of medium-scale empirical bayes ridge
% regression 
% - medium scale:  x'x fits in memory
% - empirical bayes: means evidence optimization for hyperparameters (prior
% precision and noise variance)

setpaths; % make sure we have paths set

% Set hyperparameters
nk = 1200;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

% Make filter
k = randn(nk,1)*sqrt(rho);

%  Make stimulus and response
nsamps = 500; % number of stimulus sample
signse = 5;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(211); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');

subplot(212); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');

%% Compute sufficient statistics

dd.xx = x'*x;  
dd.xy = (x'*y);
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;

% Use fixed-point algorithm
maxiter = 100;
tic;
[kridge1,hprs1] = autoRidgeRegress_fp(dd,maxiter);
toc; fprintf('\n\n');

% Use gradient-Hessian optimization in primal form
tic;
[kridge2,hprs2] = autoRidgeRegress_gradprimal(dd);
toc;
dd.y = y;
dd.x = x;  

dd.xx = x*x';  

tic;
[kridge3,hprs3] = autoRidgeRegress_graddual(dd);
toc;

%%  ---- Make Plots ----

h = plot(t,k,'k-',t,kridge1,t,kridge2,'r',t,kridge3,'g--');
set(h(1),'linewidth',2.5);
title('estimates');
legend('true', 'ridgefp', 'ridgegrad');

fprintf('\nHyerparam estimates\n------------------\n');
fprintf(' alpha:  %3.1f  %5.2f %5.2f %5.2f\n',alpha,hprs1.alpha,hprs2.alpha,hprs3.alpha);
fprintf('signse:  %3.1f  %5.2f %5.2f %5.2f\n',signse,sqrt(hprs1.nsevar),sqrt(hprs2.nsevar),sqrt(hprs3.nsevar));

err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n   RidgeFP = %7.2f\n RidgeGradPrimal = %7.2f\n RidgeGradDual = %7.2f\n', [err(kridge1) err(kridge2) err(kridge3)]);


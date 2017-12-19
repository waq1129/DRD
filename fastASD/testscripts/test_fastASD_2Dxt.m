%% test_fastASD_2D.m
%
% Test automatic smoothness determination (ASD) for a 2D spatiotemporal RF
% (1D space, 1D of time)

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nkt = 1;  % number of time lags in filter
nkx = 40;  % number of spatial pixels in filter
nks = [nkt nkx];
nktot = prod(nks); % total number of filter coeffs

% Make Gabor filter
[tg,xg] = ndgrid(1:nkt,1:nkx);
kim = makeGabor(-pi/12, (3/nkt), 0, [nkt/2, nkx/2], [nkt,nkx]/4, tg,xg)*.5;
k = kim(:);


%%  Make stimulus and response
nsamps = 5000; % number of stimulus sample
signse = 1;   % stdev of added noise

% make correlated stimuli
xf = exp(-(0:6))'*exp(-(-3:3).^2);
x = conv2(randn(nsamps,nkx),xf, 'same'); % stimuli

% generate response
filterresp = sameconv(x,kim); % convolve filter with stimulus
y = filterresp + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nktot;
subplot(121); % ------
imagesc(kim); xlabel('index'); ylabel('filter coeff'); title('true filter');
subplot(122); % ------
plot(filterresp, filterresp, 'k.', filterresp, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');

%% Compute ridge regression estimate 
% fprintf('\n...Running ridge regression with fixed-point updates...\n');
% 
% % Sufficient statistics (old way of doing it, not used for ASD)
% dd.xx = x'*x;   % stimulus auto-covariance
% dd.xy = (x'*y); % stimulus-response cross-covariance
% dd.yy = y'*y;   % marginal response variance
% dd.nx = nk;     % number of dimensions in stimulus
% dd.ny = nsamps;  % total number of samples
% 
% % Run ridge regression using fixed-point update of hyperparameters
% maxiter = 100;
% tic;
% kridge = autoRidgeRegress_fp(dd,maxiter);
% toc;


%% Compute ASD estimate
fprintf('\n\n...Running ASD_2D...\n');

minlens = [2;2];  % minimum length scale along each dimension

tic; 
dd = compSuffStats_ASD(x,y,nks,minlens);
toc;


[kasd,asdstats] = fastASD(x,y,nks,minlens);

%%  ---- Make Plots ----

subplot(222);
imagesc(reshape(kridge,nks))
title('ridge');

subplot(224);
imagesc(reshape(kasd,nks))
title('ASD');

% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f)\n',len(1),asdstats.len,ci(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f)\n',rho(1),asdstats.rho,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors
err = @(khat)(sum((k-khat(:)).^2)); % Define error function
fprintf('\nErrors:\n------\n  Ridge = %7.2f\n  ASD2D = %7.2f\n\n', ...
     [err(kridge) err(kasd)]);
% 

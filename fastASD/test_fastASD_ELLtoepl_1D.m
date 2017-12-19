%% test_fastASD_ELLtoepl_1D.m
%
% Short script to illustrate fast automatic smoothness determination (ASD)
% for vector of filter coefficients using ELL approximation in 1D

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk = 5000;  % number of filter coeffs (1D vector)
rho = 1; % marginal variance of filter coeffs
len = 20;  % ASD length scale

% Generate factored ASD prior covariance matrix in Fourier domain
opts.nxcirc = nk;
opts.condthresh = 1e6;
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len,rho],nk,opts); % columns
nf1 = length(cdiag1); % number of frequencies needed
fprintf('Number of frequencies needed for true filter: %d\n', nf1);

% Draw true regression coeffs 'k' by sampling from ASD prior 
k = U1*(sqrt(cdiag1).*randn(nf1,1));

% push edges towards zero, so filter is localized
nrm = min(max(len*10,floor(nk/5)),nk/2);
kmlt = exp(-(0:(nrm-1)).^2/(.25*nrm.^2))'; % for squashing the edges
k(1:nrm) = k(1:nrm).*flipud(kmlt);
k(end-nrm+1:end) = k(end-nrm+1:end).*kmlt;

clf; plot(1:nk,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');

%%  Make stimulus and simulate response
nsamps = 10000; % number of stimulus sample
signse = 200;   % stdev of added noise

% Make stimulus power spectrum
nxcirc = nk;  % ceil(nk+4*len);  % For now, this is the only version implemented!
ncos = ceil((nxcirc+1)/2); nsin = floor((nxcirc-1)/2);
ww = [0:(ncos-1),(-nsin:1:-1)]'; % fourier freqs
Spow = 10./(1+0.*(.1*ww.^2));  % 1/F power spectrum

% Generate stimuli 
x = realifft(bsxfun(@times,randn(nxcirc,nsamps),sqrt(Spow)));
x = x(1:nk,:)';

% % ------------------------------------------------------------
% % OPTIONAL: make stimulus cov explicitly
% % (just for visualization purposes: will run out of memory if nk too big!)
% % ------------------------------------------------------------
% B = realfftbasis(nk,nxcirc); % real Fourier basis 
% Cstim = B'*diag(Spow)*B; % stimulus prior covariance
% xcov = cov(x); % stimulus sample covariance
% 
% subplot(231);  imagesc(Cstim); axis image; 
% title('stimulus prior cov');
% subplot(232);  imagesc(xcov); axis image;4K
% title('stimulus sample cov'); 
% subplot(233);
% plot(1:nk,Cstim(nk/2,:),1:nk,xcov(nk/2,:)); 
% title('cov slices'); axis square;
% legend('prior', 'sample');
% % -------------------------------------------------------------

% simulate response
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(223); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');
subplot(224); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');


%% Compute ASD estimate 
fprintf('\n\n...Running ASD...\n');

% Set lower bound on length scale.  (Larger -> faster inference).
% If it's set too high, the function will warn that optimal length scale is
% at this lower bound, and you should re-run with a smaller value since
% this may be too high
minlen = len*.5; 

% Run ASD
tic;
[kasd,asdstats] = fastASD(x,y,nk,minlen);
toc;

%% Now test out ELL variants

% First, try computing stuff and passing in to new function that does
% things more efficiently

dd = compLSsuffstats_ELLtoepl(x,y,nk,minlen,nxcirc);


%% Compute ELLtoplitz estimate
tic;
[kasd2,asdstats2] = fastASD_ELLtoepl(dd);
toc;

%% Compute ELL estimate
ncov = length(dd.xy);
Cstmdiag = nsamps*spdiags(Spow([1:ceil(ncov/2),end-floor(ncov/2)+1:end]),0,ncov,ncov);


dd2 = dd;
dd2.xx = Cstmdiag;

tic;
[kasd3,asdstats3] = fastASD_ELLtoepl(dd2);
toc;


%%  ---- Make Plots ----
subplot(211);
h = plot(t,k,'k-',t,kasd); box off;
set(h(1),'linewidth',2.5); 
title('ridge estimate');
legend('true', 'ridge');

subplot(211);
kasdSD = sqrt(asdstats.Lpostdiag); % posterior stdev for asd estimate
plot(t,k,'k-',t,kasd,'r'); hold on; 
errorbarFill(t,kasd,2*kasdSD); % plot posterior marginal confidence intervals
hold off; box off;
legend('true','ASD')
title('ASD estimate (+/- 2SD)');
xlabel('filter coeff');

subplot(212);
kasdSD2 = sqrt(asdstats.Lpostdiag); % posterior stdev for asd estimate
plot(t,k,'k-',t,kasd2,'r', t,kasd3,'g--'); hold on; 
errorbarFill(t,kasd2,2*kasdSD2); % plot posterior marginal confidence intervals
hold off; box off;
legend('true','ELLtoepl','ELL')
title('ASD-toepl estimate (+/- 2SD)');
xlabel('filter coeff');

% Display facts about estimate
ci = asdstats.ci;
ci2 = asdstats2.ci;
ci3 = asdstats2.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n----------------------------\n');
fprintf('     l: %5.1f  %5.1f (+/-%.1f) %5.1f (+/-%.1f) %5.1f (+/-%.1f)\n',...
    len,asdstats.len,ci(1),asdstats2.len,ci2(1),asdstats3.len,ci3(1));
fprintf('   rho: %5.1f  %5.1f (+/-%.1f) %5.1f (+/-%.1f) %5.1f (+/-%.1f)\n',...
    rho,asdstats.rho,ci(2),asdstats2.rho,ci2(2),asdstats3.rho,ci3(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f) %5.1f (+/-%.1f) %5.1f (+/-%.1f)\n',...
    signse.^2,asdstats.nsevar,ci(3),asdstats2.nsevar,ci2(3),asdstats3.nsevar,ci3(3));

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n------\n       ASD = %5.2f\n  ASDtoepl = %5.2f\n    ASDELL = %5.2f\n\n', ...
    [err(kasd) err(kasd2), err(kasd3)]);

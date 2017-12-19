%% test_fastASD_2D_Spencer.m
%
% Simulated example designed to see whether we can run on problems about the size of Spencer's data. 
% Note: this is for a single lag only!  Need 3D to incorporate time

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nks = [128 128]; % number of pixels along each dimension
nk = prod(nks);  % total # of regression coefficients
len = 8*[1 1];  % length scale along each dimension
rho = .5;  % marginal prior variance

% Generate factored ASD prior covariance matrix in Fourier domain
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nks(1)); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nks(2)); % rows
nf1 = length(cdiag1); % number of frequencies needed
nf2 = length(cdiag2); 

% Draw true regression coeffs 'k' by sampling from ASD prior 
kh = sqrt(rho)*randn(nf1,nf2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nk,nf1*nf2);

% Inverse Fourier transform
kim = U1*(U2*kh')'; % convert to space domain (as 2D image )
k = kim(:);  % as vector

%%  Generate stimulus and simulate responses

nsamps = 2000; % number of stimulus samples
signse = 20;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),2)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(221);  % ------
imagesc(U1*diag(cdiag1)*U1');  title('column prior covariance');
subplot(223); % ------
imagesc(kim); xlabel('index'); ylabel('filter coeff'); title('true filter');
subplot(224); % ------
plot(x*k, x*k, 'k.', x*k, y, 'r.'); xlabel('noiseless y'); ylabel('observed y');


%% Compute isotropic ASD estimate
fprintf('\n\n...Running isotropic ASD_2D...\n');

minlens = [7;7];  % minimum length scale along each dimension (must be <= true len)
tic; 
[kasd,asdstats] = fastASD(x,y,nks,minlens);
toc;


%%  ---- Make Plots ----

% Plot sorted coeffs
subplot(211);
[ksrt,iisrt] = sort(k); 
h = plot(1:nk, kasd(iisrt), 'r--',1:nk, ksrt, 'k');
set(h(2), 'linewidth', 3);
legend('ASD', 'true k', 'location', 'northwest');
title('sorted coeffs');

% Show image of ASD estimate
subplot(224);
imagesc(reshape(kasd,nks))
title('isotropic ASD estimate');

% Display facts about estimate
ci = asdstats.ci;
fprintf('\nHyerparam estimates (+/-1SD)\n-----------------------\n');
fprintf('        True    isotropic         full  \n');
fprintf('   rho: %5.1f  %5.2f (+/-%.2f)\n',rho(1),asdstats.rho,ci(1));
fprintf('   len: %5.1f  %5.2f (+/-%.2f)\n',len(1),asdstats.len,ci(2));
fprintf('nsevar: %5.1f  %5.1f (+/-%.1f)\n\n',signse.^2,asdstats.nsevar,ci(3));

% Compute errors
err = @(khat)(1-(norm(k-khat(:))/norm(k))^2); % Define error function
fprintf(['R^2:\n------\n  ASD  = %6.3f\n'], [err(kasd)]);
% 

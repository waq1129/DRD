%% test_ELLtoeplitzCov_estim.m
%
% Examine whether we can do better if we find least-squares estimate of toeplitz covariance 

%% SETUP

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk = 200;  % number of filter coeffs (1D vector)
rho = 2; % marginal variance
len = 15;  % ASD length scale

% Set params governing circular boundary
nxx = nk*2-1; % number of elements in full autocovariance
nxc = ceil(nk+3*len); % circular boundary
ncos = ceil((nxc+1)/2); % number of cosine terms
nsin = floor((nxc-1)/2); % number of sine terms

% Make ASD covariance for prior over weights
opts.nxcirc = nxc;
opts.condthresh = 1e8;
[cdiag,Uc,wvec] = mkcov_ASDfactored([len,rho], nk,opts);
nw = length(wvec);

Cprior = Uc*diag(cdiag)*Uc'; % stimulus covariance
Cpriorinv = Uc*diag(1./cdiag)*Uc';
k = mvnrnd(zeros(1,nk),Cprior)'; % sample k from mvnormal with this covariance

nrm = min(50,floor(nk/5));
k(1:nrm) = k(1:nrm)./(sqrt(nrm:-1:1))';
k(end-nrm+1:end) = k(end-nrm+1:end)./(sqrt(1:nrm))';
plot(k);
% Set additional hyperparam
signse = 20;

%% MAKE STIMULUS

Cstimdiag = exp(-abs(wvec)/5);
Cstim = Uc*diag(Cstimdiag)*Uc'; % stimulus covariance
L = Uc*diag(sqrt(Cstimdiag)); % for creating correlated stimuli

%  Make stimulus 
nsamps = 100; % number of stimulus sample
x = (L*randn(length(Cstimdiag),nsamps))'; % stimuli
xcov = cov(x);

%% 1. Compute stimulus autocovariance (using FFT)

% Smarter way to do it (using FFT);
xh = ifft(mean(abs(fft(x',nk+ncos).^2),2));
nnrm = min(ncos,nk); % number of coeffs to worry about normalizing
xh(1:nnrm) = xh(1:nnrm)./(nk:-1:(nk-nnrm+1))';
xc = ([xh(1:ncos);flipud(xh(2:nsin+1))])*nsamps/(nsamps-1);


%% 2. Compute Toeplitz approximation to full covariance

% Make Fourier domain cov
Chatdiag = real(fft(xc));  % fourier transform of row

% Make cov with realFFT 
Br = realfftbasis(nxc,nxc);
Ctoepl = Br'*diag(Chatdiag)*Br;
xcovtoepl = Ctoepl(1:nk,1:nk); % truncate to just the part we need

%% 3. Alternative estimate (least-squares).

ii = vec(circulant(1:nk));
M = sparse(1:nk^2,ii,1);
xc2 = (M'*M)\(M'*xcov(:));
xc2 = ([xc2(1:ncos);flipud(xc2(2:nsin+1))]);
Ctoepl2 = Br'*diag(real(fft(xc2)))*Br;
xcovtoepl2 = Ctoepl2(1:nk,1:nk); % truncate to just the part we need

subplot(221); imagesc(Cstim);
subplot(222); imagesc(xcov);
subplot(223); imagesc(xcovtoepl);
subplot(224); imagesc(xcovtoepl2);


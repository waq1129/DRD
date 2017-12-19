%% test_ELLtoeplitzCov2.m
%
% Examine ELL-toeplitz approximation to stimulus covariance and ELL map
% estimate

%% SETUP

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk = 200;  % number of filter coeffs (1D vector)
rho = 2; % marginal variance
len = 10;  % ASD length scale

% Set params governing circular boundary
nxc = nk; % ceil(nk+len*3); % use this for easy handling of circular boundary

% Make ASD covariance for prior over weights
opts.nxcirc = nxc;
opts.condthresh = 1e8;
[cdiag,Uc,wvec] = mkcov_ASDfactored([len,rho], nk,opts);
nw = length(wvec);

Cprior = Uc*diag(cdiag)*Uc'; % stimulus covariance
Cpriorinv = Uc*diag(1./cdiag)*Uc';
k = mvnrnd(zeros(1,nk),Cprior)'; % sample k from mvnormal with this covariance

% push edges towards zero, so filter is localized
nrm = max(len*8,floor(nk/5)); 
kmlt = exp(-(0:(nrm-1)).^2/(8*len.^2))'; % for squashing the edges
k(1:nrm) = k(1:nrm).*flipud(kmlt);
k(end-nrm+1:end) = k(end-nrm+1:end).*kmlt;
plot(k);


%%  Make stimulus and simulate response
nsamps = 1000; % number of stimulus sample
signse = 10;   % stdev of added noise

% Make stimulus power spectrum
ncos = ceil((nxc+1)/2); % number of cosine terms
nsin = floor((nxc-1)/2); % number of sine terms
ww = [0:(ncos-1),(-nsin:1:-1)]'; % fourier freqs
Spow = 1./(1+.01*ww.^2);  % 1/F power spectrum

% Generate stimuli 
x = realifft(bsxfun(@times,randn(nxc,nsamps),sqrt(Spow)));
x = x(1:nk,:)';  % truncate

% ------------------------------------------------------------
% OPTIONAL: make stimulus cov explicitly
% (just for visualization purposes: will run out of memory if nk too big!)
% ------------------------------------------------------------
B = realfftbasis(nk,nxc); % real Fourier basis 
Cstim = B'*diag(Spow)*B; % stimulus prior covariance
xx = x'*x; % stimulus sample covariance


subplot(231);  imagesc(Cstim); axis image; 
title('stimulus prior cov');
subplot(232);  imagesc(xx); axis image;
title('stimulus sample cov'); 
subplot(233);
plot(1:nk,Cstim(nk/2,:),1:nk,xx(nk/2,:)); 
title('cov slices'); axis square;
legend('prior', 'sample');
% -------------------------------------------------------------
% -------------------------------------------------------------

% --- simulate response ---
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(223); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');
subplot(224); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');


%% 1. Compute stimulus autocovariance

% Smart way to do it (using FFT);
xh = ifft(mean(abs(fft(x',nk+ncos).^2),2));
nnrm = min(ncos,nk); % number of coeffs to worry about normalizing
xh(1:nnrm) = xh(1:nnrm)./(nk:-1:(nk-nnrm+1))';
xc = ([xh(1:ncos);flipud(xh(2:nsin+1))]); % *nsamps/(nsamps-1);

% plot it
plot(1:nxc,xc);
title('stimulus autocovariance'); xlabel('lag (circular)');


%% 2. Compute Toeplitz approximation to full covariance

% Make Fourier domain cov
Chatdiag = real(fft(xc));  % fourier transform of row

% Make cov with realFFT 
Br = realfftbasis(nxc,nxc);
Ctoepl = Br'*diag(Chatdiag)*Br;
xxtoepl = nsamps*Ctoepl(1:nk,1:nk); % truncate to just the part we need

% Check covariances (optional)
imagesc([nsamps*Cstim, xx, xxtoepl])


%% 3. Compute the MAP estimate using each version of X'X
vn = signse^2;  % noise variance
Cinv = diag(1./cdiag);  % inverse of prior covariance 
xyh = Uc'*(x'*y);  

%kml = (x'*x)\(x'*y);  % But this is terrible.  Don't plot
kmap = Uc * ((Uc'*(x'*x)*Uc + vn*Cinv)\xyh);
kell = Uc * ((Uc'*(nsamps*Cstim)*Uc + vn*Cinv)\xyh);
kelltoepl = Uc * ((Uc'*(xxtoepl)*Uc + vn*Cinv)\xyh);


%% 4. Now do it using circular assumption in the fourier domain
iiw1 = 1:(nw+1)/2;
iiw2 = nxc-(nw-3)/2:nxc;
iiw = [iiw1 iiw2];
XXrfft = nsamps*diag(Chatdiag([iiw1 iiw2]));  % This is the fourier-domain covariance (keep just the freqs in from wvec)

% Test it
subplot(211);
imagesc([xx, xxtoepl, Uc*XXrfft*Uc']);  % should match


% Compute extrapolated X'*Y
xy = x'*y;
iix1 = 1:nk;
iix2 = nk+1:nxc;
xy2 = Ctoepl(iix2,iix1)*(Ctoepl(iix1,iix1)\xy);

xyfull = [xy; xy2];
subplot(223);
plot(1:nxc, xyfull, 1:nk, xy,'r--');
legend('original xy', 'extrapolated xy');
title('X''*Y');

xyfh = Br*xyfull;
xyfh = xyfh([iiw1 iiw2]);
subplot(224);
plot(1:nw, xyh, 1:nw, xyfh, 'r--')
title('fft(X''*Y)');

% ELL estimate under circular assumption
kelltoepl2 = Uc* ((XXrfft + vn*Cinv) \xyh);

%% Now re-do 4, but more compactly / smarter

% Compute stimulus autocovariance
xh = ifft(mean(abs(fft(x',nk+ncos).^2),2));
nnrm = min(ncos,nk); % number of coeffs to worry about normalizing
xh(1:nnrm) = xh(1:nnrm)./(nk:-1:(nk-nnrm+1))';
xc = ([xh(1:ncos);flipud(xh(2:nsin+1))]); % *nsamps/(nsamps-1);

% Compute matrices necessary for extending X'Y.
iix1 = 1:nk;
iix2 = nk+1:nxc;
M = toeplitz(xc,xc((iix1)));
Mii = M(iix1,iix1);
Mij = M(iix2,iix1);


%% 6. Compare the posteriors we get from the two approaches
ii = [iiw1, iiw2]; % indices to keep 
Cp = Br(ii,:)'*diag(cdiag)*Br(ii,:); % prior covariance
L1 = (1/vn*Cp*(nsamps*Ctoepl) + eye(nxc))\Cp;
L1prune = L1(1:nk,1:nk);
L2 = Uc*inv(1/vn*Uc'*(x'*x)*Uc + Cinv)*Uc';
L3 = (1/vn*Cprior*(x'*x) + eye(nk))\Cprior;

subplot(211); imagesc([L2 L1prune]); title('true and approx posterior cov ');
%subplot(222); imagesc(L1prune); title('toepl-approx posterior cov');
subplot(425); plot(1:nk,L2(:,1),1:nk,L1prune(:,1),'r--'); title('1st slice');
subplot(427); plot(1:nk,L2(:,nk/2),1:nk, L1prune(:,nk/2),'r--'); title('middle slice');

%% Make plots
subplot(2,2,4)
t = 1:nk;
plot(t,k,t,kmap,t,kell,t,kelltoepl,t,kelltoepl2,'--');
legend('true','map', 'ell', 'toepl','toepl2');
title('filter estimates');

% Compute errors
err = @(khat)(norm(k-khat).^2);
errs = [err(kmap), err(kell), err(kelltoepl) err(kelltoepl2)]
fprintf('\nErrs:\n map=%7.2f\n ell=%7.2f\n toepl= %.2f\n toepl2=%.2f\n', errs);


% %% See how well vanilla ASD estimate does
% 
% minlen = len*.75;
% [kasd,asdstats] = fastASD(x,y,nk,minlen);
% errs = [err(kmap), err(kasd)]

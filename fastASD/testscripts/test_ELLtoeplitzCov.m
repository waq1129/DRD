%% test_ELLtoeplitzCov.m
%
% Examine ELL-toeplitz approximation to stimulus covariance and ELL map
% estimate

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
nsamps = 200; % number of stimulus sample
x = (L*randn(length(Cstimdiag),nsamps))'; % stimuli


%% 1. Compute stimulus autocovariance

xc = zeros(nxc,1); % stimulus autocovariance (row vec)
for jj = 1:ncos
    xc(jj) = sum(sum(x(:,jj:end).*x(:,1:end-jj+1),2))/(nk+1-jj);
end
xc(ncos+1:end) = flipud(xc(2:nsin+1));
xc = xc/(nsamps-1);

% Smarter way to do it (using FFT);
xh = ifft(mean(abs(fft(x',nk+ncos).^2),2));
nnrm = min(ncos,nk); % number of coeffs to worry about normalizing
xh(1:nnrm) = xh(1:nnrm)./(nk:-1:(nk-nnrm+1))';
xc2 = ([xh(1:ncos);flipud(xh(2:nsin+1))])*nsamps/(nsamps-1);

% Show they're the same
clf;
plot(1:nxc,xc,1:nxc,xc2,'r--');
title('stimulus autocovariance');
xlabel('lag (circular)');


%% 2. Compute Toeplitz approximation to full covariance


% % METHOD 1:  average diagonal of covariance
% % =========================================
% 
% xcov = (x'*x)/(nsamps-1); % Compute covariance matrix explicitly
% xcovrow = zeros(1,nxx);
% for jj = 1:nk
%     xcovrow(end-nk-jj+2:end-jj+1) = xcovrow(end-nk-jj+2:end-jj+1)+ xcov(jj,:);
% end
% ncovobs = [1:nk,nk-1:-1:1];
% xcovrow = xcovrow./ ncovobs;
% xcovrow = circshift(xcovrow',nk);
% xcovrow = xcovrow([(1:ncos),(nxx-nsin+1:nxx)]);
% plot(xcovrow);


% METHOD 2:  use Fourier domain
% =========================================

% Make Fourier domain cov
Chatdiag = real(fft(xc));  % fourier transform of row

% Make cov with realFFT 
Br = realfftbasis(nxc,nxc);
Ctoepl = Br'*diag(Chatdiag)*Br;
xcovtoepl = Ctoepl(1:nk,1:nk); % truncate to just the part we need

% Check covariances (optional)
imagesc([Cstim, cov(x), xcovtoepl])

%% 3. simulate response 

y = x*k + randn(nsamps,1)*signse;  % generate response


%% 4. Compute the MAP estimate using each version of X'X
vn = signse^2;  % noise variance
Cinv = diag(1./cdiag);  % inverse of prior covariance 
xyh = Uc'*(x'*y);  

%kml = (x'*x)\(x'*y);  % But this is terrible.  Don't plot
kmap = Uc * ((Uc'*(x'*x)*Uc + vn*Cinv)\xyh);
kell = Uc * ((Uc'*(nsamps*Cstim)*Uc + vn*Cinv)\xyh);
kelltoepl = Uc * ((Uc'*(nsamps*xcovtoepl)*Uc + vn*Cinv)\xyh);

%% 5. Now do it using circular assumption in the fourier domain
iiw1 = 1:(nw+1)/2;
iiw2 = nxc-(nw-3)/2:nxc;
XXrfft = diag(Chatdiag([iiw1 iiw2]));  % This is the fourier-domain covariance (keep just the freqs in from wvec)

% Test it
xcov = (x'*x)/(nsamps-1);
subplot(211);
imagesc([xcov, xcovtoepl, Uc*XXrfft*Uc']);  % should match


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
kelltoepl2 = Uc* ((nsamps*XXrfft + vn*Cinv) \xyh);

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

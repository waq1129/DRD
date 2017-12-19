%% test_ELLtoeplitzCov2.m
%
% Examine ELL-toeplitz approximation to stimulus covariance and ELL map
% estimate

%% SETUP

% add directory with DFT tools 
%setpaths;


% Generate true filter vector k
nk1 = 20; nk2 = nk1;  % number of filter pixels along [cols, rows]
nktot = nk1*nk2; % total number of filter coeffs
len = [5 5];  % length scale along each dimension
rho = 1;  % marginal prior variance

% Generate factored ASD prior covariance matrix in Fourier domain
nxc = nk1;
opts.nxcirc = nxc;
opts.condthresh = 1e6;
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nk1,opts); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nk2,opts); % rows
nw1 = length(cdiag1); % number of frequencies needed
nw2 = length(cdiag2); 
nw = nw1*nw2; % total number of frequencies 

% Draw true regression coeffs 'k' by sampling from ASD prior 
kh = sqrt(rho)*randn(nw1,nw2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nktot,nw1*nw2);

% Inverse Fourier transform
kim0 = U1*(U2*kh')'; % convert to space domain (as 2D image )

%% push edges towards zero, so filter is localized to middle of region
[xg,yg] = meshgrid(1:nk1,1:nk2); xg = (xg(:)-nk1/2); yg = (yg(:)-nk2/2); % gridded x and y locations
ksig = nk1*.4; p = 6;
kmlt = exp(-.5*((abs(xg)/ksig).^p+(abs(yg)/ksig).^p));
kmlt = reshape(kmlt,nk1,nk2);
subplot(221); imagesc(kmlt);
subplot(222); plot(kmlt);
kim = kim0.*kmlt;
k = kim(:);  % as vector
subplot(223); imagesc(kim);
subplot(224); plot(kim);

% Optional: make full covariance matrix (for inspection purposes only; will cause
% out-of-memory error if filter dimensions too big!)
C1 = U1*diag(cdiag1)*U1';
C2 = U2*diag(cdiag2)*U2';
Cprior = rho*kron(C2,C1);


%%  Make stimulus and simulate response
nsamps = 2000; % number of stimulus sample
signse = 10;   % stdev of added noise

% Make stimulus power spectrum
ncos = ceil((nxc+1)/2); % number of cosine terms
nsin = floor((nxc-1)/2); % number of sine terms
ww = [0:(ncos-1),(-nsin:1:-1)]'; % fourier freqs
Spow1 = 1./(1+.01*ww.^2);  % 1/F power spectrum
Spow2 = 1./(1+.05*abs(ww).^1);  % 1/F power spectrum
subplot(211); plot([Spow1 Spow2]);
Spow = kron(Spow2,Spow1);

% Generate stimuli 
B1 = realfftbasis(nxc);
x = kronmult({diag(sqrt(Spow2)),diag(sqrt(Spow1))},randn(nxc.^2,nsamps));
x = kronmult({B1',B1'},x)';

%%

% ------------------------------------------------------------
% OPTIONAL: make stimulus cov explicitly
% (just for visualization purposes: will run out of memory if nk too big!)
% ------------------------------------------------------------

% % NO IDEA WHY THIS DOESN'T WORK
% B = realfftbasis(nk1,nxc); % real Fourier basis 
% Cstim1 = B'*diag(Spow1)*B; % stimulus prior covariance
% Cstim2 = B'*diag(Spow2)*B; % stimulus prior covariance
% Cstim = kron(Cstim2,Cstim1);
%
% x2 = (chol(Cstim)'*randn(nxc.^2,nsamps))'; xx2 = x2'*x2; % AND THIS TOO

% BUT THIS DOES:
Cs0 = kronmult({diag(sqrt(Spow2)),diag(sqrt(Spow1))},speye(nxc.^2));
Cs = kronmult({B1',B1'},Cs0);
Cstim = Cs*Cs';

% compute stimulus sample covariance
xx = x'*x;

clf;
subplot(231);  imagesc(Cstim); axis image; 
title('stimulus prior cov');
subplot(232);  imagesc(xx); axis image;
title('stimulus sample cov'); 
subplot(233);
plot(1:nktot,xx(nktot/2,:),1:nktot,nsamps*Cstim(nktot/2,:),'--');
title('cov slices'); axis square;
legend('sample', 'stim prior');

%% --- simulate response ---
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nktot;
subplot(223); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');
subplot(224); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');


%% 1. Compute stimulus autocovariance

% Smart way to do it (using FFT);
xtens = reshape(x',nk1,nk2,nsamps);  % tensor x
xh = ifft2(mean(abs(fft2(xtens,nk1+ncos,nk2+ncos).^2),3));
nnrm = min(ncos,nk1); % number of coeffs to worry about normalizing
knrm = (nk1:-1:(nk1-ncos+1))'*(nk2:-1:(nk2-ncos+1));  % normalizer (for unbiased estimate of cov)
xh(1:nnrm,1:nnrm) = xh(1:nnrm,1:nnrm)./knrm;

%%
xcfull = ([xh(1:ncos,:);flipud(xh(2:nsin+1,:))]); % symmetrize vertically
xcfull = ([xcfull(:,1:ncos),fliplr(xcfull(:,2:nsin+1))]); % symmetrize horizontally

%%% plot it
plot(1:nxc,xcfull);
title('stimulus autocovariance'); xlabel('lag (circular)');


%% 2. Compute Toeplitz approximation to full covariance

% Make Fourier domain cov
Chatdiag = vec(real(fft2(xcfull)));  % fourier transform of row

% Make cov with realFFT 
Br = realfftbasis(nxc,nxc);
Ctoepl = kron(Br',Br')*diag(Chatdiag)*kron(Br,Br);
xxtoepl = nsamps*Ctoepl(1:nktot,1:nktot); % truncate to just the part we need

% Check covariances (optional)
subplot(211);
imagesc([xx, nsamps*Cstim, xxtoepl])
subplot(212);
h = plot(1:nktot,xx(:,nktot/2), ...
    1:nktot,nsamps*Cstim(:,nktot/2),... 
    1:nktot, xxtoepl(:,nktot/2), '--');
set(h(1:2),'linewidth', 2);
legend('xx', 'ell', 'elltoepl');
set(gca,'xlim', [150 250]);

dfun = @(x1)(norm(xx(:)-x1(:)));
XXdiscrepancy = [dfun(nsamps*Cstim), dfun(xxtoepl)];
XXdiscrepancy-min(XXdiscrepancy)



% fuck yeah, it works!  (this is a major surprise)

% 3. Compute the MAP estimate using each version of X'X
vn = signse^2;  % noise variance
Cinv1 = diag(1./cdiag1);  % inverse of prior covariance 
Cinv2 = diag(1./cdiag2);  % inverse of prior covariance 
Cinv = kron(Cinv2,Cinv1);

%%
Uc = kron(U2,U1);
xyh = Uc'*(x'*y);  

kmap = Uc * ((Uc'*(x'*x)*Uc + vn*Cinv)\xyh);
kell = Uc * ((Uc'*(nsamps*Cstim)*Uc + vn*Cinv)\xyh);
kelltoepl = Uc * ((Uc'*(xxtoepl)*Uc + vn*Cinv)\xyh);

% STILL TO DO: extend x boundary so that nxcirc > nx

%% 6.  Make plots

subplot(231);  imagesc(kim);
subplot(232); imagesc(reshape(kmap,[nk1,nk2]));
subplot(233); imagesc(reshape(kell,[nk1,nk2]));
subplot(234); imagesc(reshape(kelltoepl,[nk1,nk2]));

subplot(2,2,4)
t = 1:nktot;
plot(t,k,t,kmap,t,kell,t,kelltoepl);
legend('true','map', 'ell', 'toepl');
title('filter estimates');

% Compute errors
err = @(khat)(norm(k-khat).^2);
errs = [err(kmap), err(kell), err(kelltoepl)]
fprintf('\nErrs:\n map=%7.2f\n ell=%7.2f\n toepl= %.2f\n', errs);


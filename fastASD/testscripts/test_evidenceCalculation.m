% test_EvidenceCalculation
%
% Examine effects of full rank vs. reduced-rank prior covariance on negative log evidence

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk = 1000;  % number of filter coeffs (assumed 1D)
rho = 1; % marginal variance
len = 50; % length scale

C0 = mkcov_ASD(len,rho,nk); % prior covariance matrix 
k = mvnrnd(zeros(1,nk),C0)'; % sample k from prior with this covariance

% -- plot ---
subplot(221);
plot(1:nk, k);
xlabel('index'); 
ylabel('filter coeff'); 
title('true filter');

%%  Make stimulus and response
nsamps = 100; % number of stimulus sample
signse = 5;   % stdev of added noise
nsevar = signse.^2; % observation noise variance
x = gsmooth(randn(nk,nsamps),.5)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% -- plot to examine noise level ---
rnge = [min(y),max(y)];
subplot(223);
h = plot(rnge,rnge,'k', x*k, y, 'r.'); set(h, 'markersize',12);
xlabel('noiseless y'); ylabel('observed y');


%% Precompute sufficient statistics
sdat.xx = x'*x;  
sdat.xy = (x'*y);
sdat.yy = y'*y;
sdat.nx = nk;
sdat.nsamps = nsamps;

% set up grid over d (smoothness) parameter
lgrid = linspace(len/10,len*2, 50);
nd = length(lgrid);
nle = zeros(nd,4);

for jj = 1:nd
    
    % set length-scale (smoothness) param
    lval = lgrid(jj);

    % Compute evidence for original ASD covariance
    C1 = mkcov_ASD(lval,rho,nk);
    nle(jj,1) = neglogev_dualform(x,y,nsevar,C1);
    
    % Compute evidence with pruned ASD covariance (but evaluated in dual)
    [Cdiag,U1,wvec] = mkcov_ASDfactored([lval;rho],nk);
    C2 = U1*diag(Cdiag)*U1';
    nle(jj,2) = neglogev_dualform(x,y,nsevar,C2);
    % report rank of Fourier-domain covariance matrix
    fprintf('d=%.2f, rank(C)=%d\n',lval,length(Cdiag));
    
    % Compute primal form in Fourier domain with Fourier-transformed data
    % (just for debugging)
    nxcirc = nk+ceil(6*lval);  % number of points for FFT (sets circular boundary)
    condthresh = 1e10; % cutoff for condition number
    B = realfftbasis(nk,nxcirc,wvec);  % change of basis matrix
    sdatFourier = sdat; 
    sdatFourier.xx = B*sdat.xx*B';
    sdatFourier.xy = B*sdat.xy;
    trho = sqrt(2*pi)*rho*lval; % transformed rho param
    ww = (2*pi/nxcirc)^2*(wvec.^2); % normalized squared Fourier freqs
    tic;
    prs = [lval;trho;  nsevar];  % ASD params
    nle(jj,3) = neglogev_ASDspectral(prs,sdatFourier,ww,condthresh);
    toc;
end

%% -- Make plot -----
subplot(222);
plot(lgrid,nle(:,1), '-',lgrid,nle(:,2),'g--',lgrid,nle(:,3),'ro');
legend('dual','factored','spectral');
ylabel('neg log evidence');
title('neg log evidence');
xlabel('d');
subplot(224); % ----
plot(lgrid,exp(min(nle(:,1))-nle(:,1)), '-', ...
    lgrid,exp(min(nle(:,2))-nle(:,2)), 'g--',...
    lgrid,exp(min(nle(:,3))-nle(:,3)), 'ro');
title('evidence');
ylabel('evidence');
xlabel('d');



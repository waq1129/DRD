%% test_fastASD_ELLtoepl_2D.m
%
% Script to illustrate fast automatic smoothness determination (ASD)
% for vector of filter coefficients using ELL toeplitz approximation in 2D

% add directory with DFT tools 
setpaths;

% Generate true filter vector k
nk1 = 150; nk2 = nk1;  % number of filter pixels along [cols, rows]
nks = [nk1 nk2];
nktot = nk1*nk2; % total number of filter coeffs
len = [nk1/10 nk2/10];  % length scale along each dimension
rho = 100;  % marginal prior variance
nxcirc = nks;
fprintf('\n********\nFilter dimensionality: (%d x %d) = %d total coeffs\n',nk1,nk2,nktot);

% Params governing samples and reponse noise
nsamps = 5000; % number of stimulus sample
signse = 125;   % stdev of added noise

% Generate factored ASD prior covariance matrix in Fourier domain
nxc = nk1;
opts.nxcirc = nxc;
opts.condthresh = 1e6;
[cdiag1,U1,wvec1] = mkcov_ASDfactored([len(1),1],nk1,opts); % columns
[cdiag2,U2,wvec2] = mkcov_ASDfactored([len(2),1],nk2,opts); % rows
nw1 = length(cdiag1); % number of frequencies needed
nw2 = length(cdiag2); 
nw = nw1*nw2; % total number of frequencies 

useGabor=1;
if ~useGabor % --------------------------------------
% ====================================================================
% OPTION 1:  Draw true regression coeffs 'k' by sampling from ASD prior  
% ====================================================================
kh = sqrt(rho)*randn(nw1,nw2).*(sqrt(cdiag1)*sqrt(cdiag2)'); % Fourier-domain kernel
fprintf('Filter has: %d pixels, %d significant Fourier coeffs\n',nktot,nw1*nw2);

% Inverse Fourier transform
kim0 = U1*(U2*kh')'; % convert to space domain (as 2D image )

% push edges towards zero, so filter is localized to middle of region
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

% % Optional: make full covariance matrix (for inspection purposes only; will cause
% % out-of-memory error if filter dimensions too big!)
% C1 = U1*diag(cdiag1)*U1';
% C2 = U2*diag(cdiag2)*U2';
% Cprior = rho*kron(C2,C1);

else % --------------------------------------
    
% ====================================================================
% OPTION 2: Use Gabor
% ====================================================================
[xg,yg] = meshgrid(1:nk1,1:nk2);
kim = makeGabor(-pi/3, (3/nk1), 0,nk1/2*[1 1], nk1/6*[1.5 1], xg,yg)*2;
k = kim(:);

% plot it
clf; subplot(121); imagesc(kim); axis image;
title('true filter');
subplot(122); plot(k);

end % --------------------------------------


%%  Make stimulus and simulate response

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
Sd1 = spdiags(sqrt(Spow1),0,nxc,nxc); % stimulus power spectrum, dim 1
Sd2 = spdiags(sqrt(Spow1),0,nxc,nxc); % stimulus power spectrum, dim 2
x = kronmult({Sd2,Sd1},randn(nxc.^2,nsamps));
x = kronmult({B1',B1'},x)';

%% --- simulate response ---
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nktot;
subplot(223); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');
subplot(224); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');
drawnow;


%% Compute ASD estimate 
fprintf('\n\n...Running 2D ELL-toeplitz ASD...\n');
% 
% Set lower bound on length scale.  (Larger -> faster inference).
% If it's set too high, the function will warn that optimal length scale is
% at this lower bound, and you should re-run with a smaller value since
% this may be too high
minlen = [2 2]; 

% Run ASD
tic;
[kasd,asdstats] = fastASD(x,y,nks,minlen,nxcirc);
toc;
kasdim = reshape(kasd,nks);


%% Now test out ELL variants

% First, try computing stuff and passing in to new function that does
% things more efficiently

dd = compLSsuffstats_ELLtoepl2D(x,y,nks,minlen,nxcirc);



%% Compute ELLtoplitz estimate
tic;
[kasd2im,asdstats2] = fastASD_ELLtoepl2D(dd);
toc;
kasd2 = kasd2im(:);

% %% Compute full ELL estimate 
% ncov = length(dd.xy);
% Cstmdiag = nsamps*spdiags(Spow([1:ceil(ncov/2),end-floor(ncov/2)+1:end]),0,ncov,ncov);
% dd2 = dd;
% dd2.xx = Cstmdiag;
% tic;
% [kasd3,asdstats3] = fastASD_ELLtoepl(dd2);
% toc;


%%  ---- Make Plots ----

subplot(231); imagesc(kim); title('true');
subplot(232); imagesc(kasdim); title('ASD');
subplot(233); imagesc(kasd2im); title('ELL-toeplitz');

subplot(235);
kasdSD = sqrt(asdstats.Lpostdiag); % posterior stdev for asd estimate
kasdSDim = reshape(kasdSD,nks);
tp = 1:nk1;
plot(tp,kim(nk1/2,:),tp,kasdim(nk1/2,:),tp,kasd2im(nk1/2,:), '--'); hold on;
errorbarFill(tp',kasdim(nk1/2,:)',2*kasdSDim(nk1/2,:)'); % plot posterior marginal confidence intervals
hold off; box off; axis tight;
legend('true','ASD', 'ell-toepl')
title('Estimate (+/- 2SD): middle slice');
xlabel('filter coeff');

% % Display facts about estimate
ci = asdstats.ci; ci2 = asdstats2.ci;  % conf intervals
fprintf('\n\nHyerparam estimates (+/-1SD)\n=============================\n');
fprintf('          len            rho           nsevar\n');
fprintf('---------------------------------------------\n');
if useGabor
    fprintf('    true:    ?           %5.1f            %5.1f \n',var(k),signse^2);
else
    fprintf('    true: %5.1f          %5.1f           %5.1f \n',len(1),rho,signse^2);
end
fprintf('     asd: %5.1f (+/-%.1f) %5.1f (+/-%.1f)  %5.1f (+/-%.1f)\n',...
    asdstats.len,ci(1),asdstats.rho,ci(2),asdstats.nsevar,ci(3));
fprintf('elltoepl: %5.1f (+/-%.1f) %5.1f (+/-%.1f)  %5.1f (+/-%.1f)\n',...
    asdstats2.len,ci2(1),asdstats2.rho,ci2(2),asdstats2.nsevar,ci2(3));

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n------\n       ASD = %5.2f\n  ASDtoepl = %5.2f\n\n', ...
    [err(kasd) err(kasd2)]);


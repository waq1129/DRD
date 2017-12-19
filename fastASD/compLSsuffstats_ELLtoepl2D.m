function dd = compLSsuffstats_ELLtoepl2D(x,y,nxperdim,minlens,nxcirc)
% Compute sufficient statistics with diagonalized Fourier stimulus covariance
%
% dd = compLSsuffstats_fourierELLdiag(x,y,dims,minlens,nxcirc)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%        dims [m x 1] - number of coefficients along each stimulus dimension
%     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%
% OUTPUT:
% ------
%     dd (struct) - carries sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft  {1 x p} - cell array with DFT bases for each dimension

condthresh = 1e8; % default value (condition number on prior covariance)

% Determine number of freqs and make Fourier basis for each dimension
[wwnrm,inds,wwvecs,wwinds] = compASDfreqs(nxperdim,minlens,nxcirc,condthresh);
fprintf('\n Total # Fourier coeffs represented: %d\n\n', size(wwnrm,1));
nsamps = size(y,1);
nw1 = length(wwinds{1});
nw2 = length(wwinds{2});
nwtot = length(wwnrm);

% Compute toeplitz version of stim covariance
xcov = compToeplStimCov2D(x,nxcirc);

% Make Fourier domain cov
Chatdiag = (real(fft2(xcov)));  % fourier transform of row

iiw1 = 1:ceil((nw1+1)/2); % initial freqs
iiw2 = ceil(nxcirc-(nw1-3)/2):nxcirc(1); % final freqs
iiw = [iiw1 iiw2]; % frequencies to keep

% prune out along each axis
Chatdiag2 = vec(Chatdiag(iiw,iiw));

% finally,prune out extras
Chatdiag3 = Chatdiag2(inds);

% Make Fourier-domain cov(keep just the freqs in from wvec)
XXfftdiag = nsamps*spdiags(Chatdiag3,0,nwtot,nwtot);
dd.xx = XXfftdiag;

% Compute projected mean X'*Y
xyfft = realfft(realfft(reshape(x'*y,nxperdim))')';
xyfft = vec(xyfft(iiw,iiw));
dd.xy = xyfft(inds);

% Fill in other statistics
dd.yy = y'*y; % marginal response variance
dd.nsamps = nsamps; % total number of samples
dd.wwnrm = wwnrm; % normalized squared frequencies
dd.inds = inds;
dd.wwvecs = wwvecs;
dd.wwinds = wwinds;
dd.condthresh = condthresh;
dd.nxcirc = nxcirc;
dd.nxperdim = nxperdim;
dd.minlen = minlens;
dd.nwsupport = length(iiw)*[1 1];
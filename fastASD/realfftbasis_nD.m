function Bfft = realfftbasis_nD(dims,nxcirc,wvec)
% Compute least-squares regression sufficient statistics in DFT basis
%
% [dd,wwnrm,Bfft] = compLSsuffstats_fourier(x,y,dims,minlens,nxcirc,condthresh)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%        dims [m x 1] - number of coefficients along each stimulus dimension
%     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
%
% OUTPUT:
% ------
%     dd (struct) - carries sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft  {1 x p} - cell array with DFT bases for each dimension

% Set circular bounardy (for n-point fft) to avoid edge effects, if needed
if (nargin < 2) || isempty(nxcirc)
    nxcirc = dims(:);
end
if nargin < 3
    % Make frequency vector
    ncos = ceil((nxcirc+1)/2); % number of cosine terms (positive freqs)
    nsin = floor((nxcirc-1)/2); % number of sine terms (negative freqs)
    wvec = [0:(ncos-1), -nsin:-1]'; % vector of frequencies
end
nd = length(dims); % number of filter dimensions

% Determine number of freqs and make Fourier basis for each dimension
Bfft = cell(nd,1); % Fourier basis matrix for each filter dimension

% Loop through dimensions
for jj = 1:nd
    Bfft{jj} = realfftbasis(dims(jj),nxcirc(jj),wvec);
end

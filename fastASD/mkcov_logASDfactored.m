function [logcdiag, wvec, U] = mkcov_logASDfactored(rho,len,nx,opts)
% Factored representation of ASD covariance matrix in Fourier domain
%
% [Cdiag,U,wvec] = mkcov_ASDfactored(prs,nx,opts)
%
% Covariance represented as C = U*sdiag*U'
% where U is unitary (in some larger basis) and sdiag is diagonal
%
%  C_ij = rho*exp(((i-j)^2/(2*l^2))
%
% INPUT:
% ------
%   prs [2 x 1]  - ASD parameters [len = length scale; rho - maximal variance; ]:
%    nx [1 x 1]  - number of regression coeffs
% opts [struct] - options structure with fields:
%                 .nxcirc - # of coefficients for circular boundary
%                 .condthresh - threshold for condition number of K (Default = 1e8).
% 
% Note: nxcirc = nx gives circular boundary
%
% OUTPUT:
% -------
%   cdiag [ni x 1] - vector with thresholded eigenvalues of C
%       U [ni x nxcirc] - column vectors define orthogonal basis for C (on Reals)
%    wvec [nxcirc x 1] - vector of Fourier frequencies
%      ii [nxcirc x 1] - binary vector indicating which frequencies included in U, sdiag

% len = prs(1);
% rho = prs(2);

% Parse inputs
if nargin < 4
    opts.nxcirc = nx; % extends support by 4 stdevs of ASD kernel width
    opts.condthresh = 1e8;  % threshold for small eigenvalues 
end

% Check that nxcirc isn't bigger than nx
if opts.nxcirc < nx
    warning('mkcov_ASDfactored: nxcirc < nx. Some columns of x will be ignored');
end

% compute vector of Fourier frequencies
maxfreq = floor(opts.nxcirc/(pi*opts.minlen)*sqrt(.5*log(opts.condthresh))); % max
% maxfreq = opts.nxcirc; % max
if maxfreq < opts.nxcirc/2
    wvec = [(0:maxfreq)';(-maxfreq:-1)'];
else
    % in case cutoff is above max number of frequencies
    ncos = ceil((opts.nxcirc-1)/2); % # neg frequenceis
    nsin = floor((opts.nxcirc-1)/2); % # pos frequencies
    wvec = [0:ncos, -nsin:-1]'; % vector of frequencies
end

% Compute diagonal of ASD covariance matrix
const = (2*pi/opts.nxcirc)^2; % constant 
ww = (wvec.^2)*const;  % effective frequency vector
% cdiag = sqrt(2*pi)*rho*len*exp(-.5*ww*len^2);
logcdiag = log(sqrt(2*pi)*rho*len)-.5*ww*len^2;

% Compute real-valued discrete Fourier basis U
if nargout > 2
    U = realfftbasis(nx,opts.nxcirc,wvec);
end
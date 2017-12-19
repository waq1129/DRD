function [logCdiag, wwnrm, Bfft, cdiag_half] = mkcov_logASDfactored_nD(rhos,lens,dims,minlen,nxcirc,condthresh)
% Compute least-squares regression sufficient statistics in DFT basis
%
% [dd,wwnrm,Bfft] = compLSsuffstats_fourier(x,y,dims,lens,nxcirc,condthresh)
%
% INPUT:
% -----
%           x [n x p] - stimulus, where each row vector is the spatial stim at a single time
%           y [n x 1] - response vector
%        dims [m x 1] - number of coefficients along each stimulus dimension
%     lens [m x 1] - minimum length scale for each dimension (can be scalar)
%      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
%  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
%
% OUTPUT:
% ------
%     dd (struct) - carries sufficient statistics for linear regresion
%  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim
%   Bfft  {1 x p} - cell array with DFT bases for each dimension


% Check if optional inputs passed in

if nargin < 4
    minlen = lens(1); % default value (condition number on prior covariance)
end
% Set circular bounardy (for n-point fft) to avoid edge effects, if needed
if (nargin < 5) || isempty(nxcirc)
    nxcirc = dims(:);
end
if nargin < 6
    condthresh = 1e8^(1/numel(dims)); % default value (condition number on prior covariance)
end
nd = length(dims); % number of filter dimensions
if length(lens) == 1 % make vector out of lens, if necessary
    lens = repmat(lens,nd,1);
end
if length(rhos) == 1 % make vector out of lens, if necessary
    rhos = repmat(rhos^(1/nd),nd,1);
end
if length(minlen) == 1 % make vector out of lens, if necessary
    minlen = repmat(minlen,nd,1);
end
% Determine number of freqs and make Fourier basis for each dimension
logcdiagvecs = cell(nd,1); % eigenvalues for each dimension
cdiag_half = cell(nd,1); % eigenvalues for each dimension
wvecs = cell(nd,1); % eigenvalues for each dimension
Bfft = cell(nd,1); % eigenvalues for each dimension
opt1.condthresh = condthresh;

% fprintf('\ncompLSsuffstats_fourier:\n # filter freqs per stimulus dim:');
% Loop through dimensions
if nargout>2
    for jj = 1:nd
        opt1.minlen = minlen(jj);
        opt1.nxcirc = nxcirc(jj);
        [logcdiagvecs{jj}, wvecs{jj}, Bfft{jj}] = mkcov_logASDfactored(rhos(jj),lens(jj),dims(jj),opt1);
        cdiag_half{jj} = diag(exp(logcdiagvecs{jj}/2));
    end
else
    for jj = 1:nd
        opt1.minlen = minlen(jj);
        opt1.nxcirc = nxcirc(jj);
        [logcdiagvecs{jj}, wvecs{jj}] = mkcov_logASDfactored(rhos(jj),lens(jj),dims(jj),opt1);
        cdiag_half{jj} = diag(exp(logcdiagvecs{jj}/2));
    end
end
switch nd
    % switch based on stimulus dimension
    
    case 1, % 1 dimensional stimulus
        logCdiag = logcdiagvecs{1};
        wwnrm = (2*pi/nxcirc(1))^2*(wvecs{1}.^2); % normalized freqs squared
        
    case 2, % 2 dimensional stimulus
        
        % Form full frequency vector and see which to cut
        logCdiag = tensorsum(logcdiagvecs{2},logcdiagvecs{1});
        ii = (logCdiag/max(logCdiag))>-inf; % indices to keep
        
        % compute vector of normalized frequencies squared
        [ww1,ww2] = ndgrid(wvecs{1},wvecs{2});
        wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2 ...
            (ww2(ii)*(2*pi/nxcirc(2))).^2];
        
    case 3, % 3 dimensional stimulus
        
        logCdiag = tensorsum(logcdiagvecs{3},(tensorsum(logcdiagvecs{2},logcdiagvecs{1})));
        ii = (logCdiag/max(logCdiag))>-inf; % indices to keep
        
        % compute vector of normalized frequencies squared
        [ww1,ww2,ww3] = ndgrid(wvecs{1},wvecs{2},wvecs{3});
        wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2, ...
            (ww2(ii)*(2*pi/nxcirc(2))).^2, ....,
            (ww3(ii)*(2*pi/nxcirc(3))).^2];
        
    otherwise
        error('compLSsuffstats_fourier.m : doesn''t yet handle %d dimensional filters\n',nd);
        
end

wwnrm = sum(wwnrm,2);


function xcov = compToeplStimCov(x,nn)
% Compute stimulus autocovariance
%
% xacov = compToeplStimCov(x,nn)

nx = size(x,2); % number of stim samples and stim pixels

% Compute number of terms needed in autocovariance
ncos = ceil((nn+1)/2); % number of cosine terms
nsin = floor((nn-1)/2); % number of sine terms

% Compute autocovariance (using FFT)
xh = ifft(mean(abs(fft(x',nx+ncos).^2),2));
nnrm = min(ncos,nx); % number of coeffs to worry about normalizing
xh(1:nnrm) = xh(1:nnrm)./(nx:-1:(nx-nnrm+1))'; % normalize some coeffs so 'unbiased'
xcov = [xh(1:ncos);flipud(xh(2:nsin+1))];
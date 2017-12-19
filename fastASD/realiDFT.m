function [x, wvec] = realiDFT(xhat,nx)
% x = realiDFT(xhat,nxcirc)
%
% Compute real-valued version of Inverse Discrete Fourier Transform (DFT) using ifft,  
% but storing cosine and sine terms separately so that vector is real-valued.  
%
% Input should have structure;  [DC(0), Cos(1:(nxcirc-1)/2), Sin(-(nxcirc-1)/2:-1)]';
%
% INPUT:
%      xhat - vector or matrix of Fourier transform
%        nx - # points in original signal (optional), must be <= length(nxcirc)
%
% OUTPUT:
%         x - inverse DFT of x


% convert x to column if a row vector
if (size(xhat,1) == 1)
    ROWVEC = true; 
    xhat = xhat'; 
else
    ROWVEC = false;
end

if (nargin == 1)
    nx = size(xhat,1); % number of coeffs in signal
end

nxcirc = size(xhat,1); % number of coefficients in DFT

% Make frequency vector
ncos = ceil((nxcirc-1)/2); % number of cosine terms
nsin = floor((nxcirc-1)/2); % number of sin terms

% Fix DC term
xhat(1,:) = xhat(1,:)*sqrt(2);
% Fix highest cosine term if nx is even
if mod(nx,2) == 0; 
    xhat(ncos+1,:) = xhat(ncos+1,:)*sqrt(2);
end

% Put in real parts
xfft = xhat;
xfft(ncos+2:end,:) = flipud(xhat(2:nsin+1,:));
% Put in imaginary parts
xfft(2:nsin+1,:) = xfft(2:nsin+1,:) + 1i*flipud(xhat(ncos+2:end,:));
xfft(ncos+2:end,:) = xfft(ncos+2:end,:) - 1i*xhat(ncos+2:end,:);

% Take inverse fourier transform
x = real(ifft(xfft));
x = x(1:nx,:)*sqrt(nxcirc/2);

% Convert back to row vec, if necessary
if ROWVEC,  x = x';  
end

% Compute frequency vector, if desired
if nargout > 1
    ncos = ceil((nxcirc-1)/2); % number of negative frequencies;
    nsin = floor((nxcirc-1)/2); % number of positive frequencies;
    wvec = [0:ncos, -nsin:-1]'; % vector of frequencies
end
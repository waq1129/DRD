function xr = fft2real(x, Bfft)
% transform from freq to real by using kronecker structure of Bfft

if 1
    % use kronmulttrp function
    xr = kronmulttrp(Bfft,x);
else
    if length(Bfft)==1
        nx = size(Bfft{1},2);
        xr = realiDFT(x,nx);
    end
    if length(Bfft)==2
        cellsz = cellfun(@size,Bfft,'uni',false);
        ss = cell2mat(cellsz);
        nx = ss(:,2); nx = nx(:)';
        xr = realiDFT2(reshape(x,nx));
        xr = xr(:);
    end
end



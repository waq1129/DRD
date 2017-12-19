function xh = real2fft(x, Bfft)

if 1
    xh = kronmult(Bfft,x);
else
    if length(Bfft)==1
        nx = size(Bfft{1},1);
        xh = realDFT(x,nx);
    end
    if length(Bfft)==2
        cellsz = cellfun(@size,Bfft,'uni',false);
        ss = cell2mat(cellsz);
        nx = ss(:,1); nx = nx(:)';
        xh = realDFT2(reshape(x,nx));
        xh = xh(:);
    end
end



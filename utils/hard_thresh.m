function [c0,ii] = hard_thresh(c,opt)
if ~isfield(opt,'th_c')
    opt.th_c = 0;
end

if opt.th_c
%     display('cut off')
    ii = c<0;
    c0 = c;
    c0(ii) = 0;
else
    c0 = c;
    ii = false(size(c));
end
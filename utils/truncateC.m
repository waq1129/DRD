function iikeep = truncateC(cdiag,opt)

d = length(cdiag);
svthresh = max(cdiag)*opt.svMin;
% svthresh = opt.svMin;

if min(cdiag)>svthresh
    iikeep = true(d,1);
else
    iikeep = (abs(cdiag)>=svthresh); % pixels to keep
end

% if sum(iikeep)>sum(opt.iikeep)
%     % prune if still less sparse than initial status
%     %     display('pruning happening');
%     iikeep = opt.iikeep;
% end
function p = closest_factors(x)

% ff = factor(x);
% flip_flag = 0;
% for ii=1:length(ff)-1
%     a = prod(ff(1:ii));
%     b = prod(ff(ii+1:end));
%     if flip_flag
%         break;
%     end
%     if a>=b
%         flip_flag = 1;
%     end
% end
%
% before_flip = ii-1;
% flip = ii;
%
% p1 = [prod(ff(1:before_flip)) prod(ff(before_flip+1:end))];
% p2 = [prod(ff(1:flip)) prod(ff(flip+1:end))];
%
% if abs(diff(p1))<abs(diff(p2))
%     p = p1;
% else
%     p = p2;
% end
%
% p = sort(p,'ascend');

for ii=1:x
    if ii*(ii+1)>=x
        break;
    end
end
p = [ii,ii+1];
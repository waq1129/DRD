function cc = determine_continue_value(x)
x = x(:);
cc = 1;
ll = x(end);
for ii=1:length(x)
    dd = x-circshift(x,ii);
    dda = abs(dd(end))/abs(ll);
    if dda<0.5
        %         keyboard
        cc = cc+1;
        if cc>10
            break;
        end
    else
        break;
    end
end

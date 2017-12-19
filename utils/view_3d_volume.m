function view_3d_volume(w3d,view)
clf
nd = size(w3d);
r = {1:nd(1),1:nd(2),1:nd(3)};
p = closest_factors(round(nd(view)/2));
cc = 1;
for ii=round(nd(view)/4)+1:round(nd(view)/4)+round(nd(view)/2)
    r{view} = ii;
    ww = w3d(r{1},r{2},r{3});
    ss = size(ww);
    if length(ss)>2
        ss(view) = [];
    end
    ww = reshape(ww,ss);
    subplot(p(1),p(2),cc),imagesc(ww)
    cc = cc+1;
end


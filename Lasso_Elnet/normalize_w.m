function w = normalize_w(w)
if norm(w)~=0
    w = w/norm(w);
end
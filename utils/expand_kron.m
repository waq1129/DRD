function AB = expand_kron(A)
nd = length(A);
AB = 1;
for ii=1:nd
    AB = kron(A{ii},AB);
end
function AB = kron_prod_mult_cell(A,B,transpa,transpb)
if nargin<3
    transpa = 0;
    transpb = 0;
end

nd = length(A);
if transpa==1 && transpb==0
    AB = cell(nd,1);
    for ii=1:nd
        AB{ii} = A{ii}'*B{ii};
    end
end
if transpb==1 && transpa==0
    AB = cell(nd,1);
    for ii=1:nd
        AB{ii} = A{ii}*B{ii}';
    end
end

if transpa==0 && transpb==0
    AB = cell(nd,1);
    for ii=1:nd
        AB{ii} = A{ii}*B{ii};
    end
end
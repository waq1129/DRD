function anat1 = put_on_anat(data,anat,iikeep)

[nx,ny,nz] = size(data);
anat2 = imresize3D(double(anat),[nx,ny,nz]);

anatv = vec(anat2); anatv = anatv/norm(anatv);
datav = vec(data);
anatv(iikeep) = datav(iikeep);
anat1 = reshape(anatv,[nx,ny,nz]);
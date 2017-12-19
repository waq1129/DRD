function view_nii_ui(data)
nii = make_nii(data,[],[round(size(data,1)/2),round(size(data,2)/2),round(size(data,3)/2)],64);
view_nii(nii);
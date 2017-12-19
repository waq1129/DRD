function s = a_exp_b(a,b,mm)
if nargin<3
    mm = 0;
end
azero = a==0;
aa = abs(a);
as = sign(a);
cc = log(aa)+b;
% maxcc = max(cc);
cc1 = cc-mm;
s = as.*exp(cc1);
s(azero) = 0;

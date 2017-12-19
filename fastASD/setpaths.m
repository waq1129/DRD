% setpaths.m
%
% Script to set paths for ASD code

if (exist('realfft')~=2)  % check if tools_dft is in path
    warning('need to add tools_dft (available in ncclabcode) to path');
end

if (exist('kronmult')~=2)  % check if tools_dft is in path
    warning('need to add tools_kron (available in ncclabcode) to path');
end

if (exist('grideval')~=2)  % check if tools_optim is in path
    addpath ../tools_optim
end

if (exist('autoRidgeRegress_fp')~=2)  % check if tools_optim is in path
    addpath ../code_ridgeRegress
end

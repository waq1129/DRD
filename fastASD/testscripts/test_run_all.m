% test_run_all.m

% Script to run all test scripts, just to make sure all pieces are still working


ssrunall = {'test_inspectASDcov', ...      % examine Fourier representaiton of ASD cov
    'test_evidenceCalculation', ... % evidence approximation
    'test_fastASD_1D', ... % 1D filter
    'test_fastASD_1DgroupedCoeffs', ...  % iid groups of coefficients
    'test_fastASD_2D', ...    % 2D filter
    'test_fastASD_1Dnu', ...  % 1D non-uniform
    'test_fastASD_2Dnu', ...  % 2D non-uniform  
    'test_fastASD_2D_aniso', ... % 2D w/ multiple length scales
    'test_fastASD_2D_forSpencer', ... % 2D w/ multiple length scales    
    'test_fastASD_3D'}; % 3D filter

% run them all in turn
for jtrunall=1:length(ssrunall)
    clearvars -except jtrunall ssrunall
    clf;
    eval(ssrunall{jtrunall});
    fprintf('\n\n=========\n Finished:\n %s \n=======\n',ssrunall{jtrunall});
    fprintf('(Press key to continue)\n');
    pause;
end

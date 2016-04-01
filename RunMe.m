% Please cite the following paper if your are using this code:
% @inproceedings{faraki2015approximate,
%   title={APPROXIMATE INFINITE-DIMENSIONAL REGION COVARIANCE DESCRIPTORS FOR IMAGE CLASSIFICATION},
%   author={Faraki, Masoud and Harandi, Mehrtash T and Porikli, Fatih},
%   booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
%   pages={1364--1368},
%   year={2015},
%   organization={IEEE}
% }

% Due to the utilized randomization the results differ each time you run.

clear; clc;
load('./DataSet_Color_Derivatives.mat');
load('./TrnTstInds');
gamma = 1e-5;
tmpNProj = 21;

DataSet.RawSamples = double(DataSet.RawSamples);
TrainSet.C = DataSet.C(:,:,allTrnInds);
TestSet.C = DataSet.C(:,:,allTstInds);
TrainSet.y = DataSet.y(allTrnInds);
TestSet.y = DataSet.y(allTstInds);
NTest = length(TestSet.y);
NTrain = length(TrainSet.y);

acc_PLS = classify_PLS(DataSet.C  , DataSet.y, allTrnInds, allTstInds);
Dist = AIRM(TestSet.C , TrainSet.C);
[~ , b] = min(Dist, [], 2);
pred_y = TrainSet.y(b);
acc_NN = numel(find(pred_y == TestSet.y)) / NTest;


proj_C_RFF = RandomFourierFeatures(DataSet.RawSamples , tmpNProj, gamma );
TrainSet.proj_C_RFF = proj_C_RFF(:,:,allTrnInds);
TestSet.proj_C_RFF = proj_C_RFF(:,:,allTstInds);
acc_PLS_RFF = classify_PLS(proj_C_RFF, DataSet.y, allTrnInds, allTstInds);
Dist_proj = AIRM(TestSet.proj_C_RFF , TrainSet.proj_C_RFF);
[~ , b] = min(Dist_proj, [], 2);
pred_y = TrainSet.y(b);
acc_NN_RFF = numel(find(pred_y == TestSet.y)) / NTest;

proj_C_Nyst = NystEmbedding(DataSet.RawSamples , tmpNProj, gamma);
TrainSet.proj_C_Nyst = proj_C_Nyst(:,:,allTrnInds);
TestSet.proj_C_Nyst = proj_C_Nyst(:,:,allTstInds);
acc_PLS_Nyst = classify_PLS(proj_C_Nyst, DataSet.y, allTrnInds, allTstInds);
Dist_proj = AIRM(TestSet.proj_C_Nyst , TrainSet.proj_C_Nyst);
[~ , b] = min(Dist_proj, [], 2);
pred_y = TrainSet.y(b);
acc_NN_Nyst = numel(find(pred_y == TestSet.y)) / NTest;

disp('Classification accuracies using nearest neighbour classifier on the Conventional RCovDs and  approximate infinite dimensional RCovDs obtained by the random Fourier features and the Nystrom method, respectively.');
acc_NN
acc_NN_RFF
acc_NN_Nyst
disp('Classification accuracies using kernel partial least squares classifier on the Conventional RCovDs and  approximate infinite dimensional RCovDs obtained by the random Fourier features and the Nystrom method, respectively.');
acc_PLS
acc_PLS_RFF
acc_PLS_Nyst

% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

% Please cite the following paper if your are using this code:

% "Approximate Infinite-Dimensional Region Covariance Descriptors for Image
% Classification", Masoud Faraki, Mehrtash Harandi, and Fatih Porikli, 40th
% IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brisbane, Australia, April 19-24, 2015.

% @inproceedings{faraki2015approximate,
%   title={APPROXIMATE INFINITE-DIMENSIONAL REGION COVARIANCE DESCRIPTORS FOR IMAGE CLASSIFICATION},
%   author={Faraki, Masoud and Harandi, Mehrtash T and Porikli, Fatih},
%   booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
%   pages={1364--1368},
%   year={2015},
%   organization={IEEE}
% }

function acc_PLS = classify_PLS(X, y, allTrnInds, allTstInds)
numFactor = 50;
%Creating so called kernel data
nPoints = length(y);
nClasses = max(y);
Y = zeros(nClasses,nPoints);
Y(sub2ind(size(Y),y,1:nPoints)) = 1;
y_tst = y(allTstInds);

for tmpC1 = 1:nPoints
    lox_X(:,:,tmpC1) = logm(X(:,:,tmpC1));
end
vec_log_X = SPD2Euclidean(lox_X);
%Kernel PLS
K = vec_log_X'*vec_log_X;
Ktr = K(allTrnInds,allTrnInds);
Kts = K(allTrnInds,allTstInds);

[B,T,U] = KerNIPALS(Ktr,Y(:,allTrnInds)',numFactor);
temp = T'*Ktr*U;
tmpMat = (U/(temp + 1e-5 * eye(size(temp))))*T'*Y(:,allTrnInds)';
PLStrn = Ktr'*tmpMat;
PLStst = Kts'*tmpMat;

[~,ypred] = max(PLStst,[],2);
acc_PLS = sum(ypred == y_tst')/length(allTstInds);
%%

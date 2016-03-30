function outPoints = SPD2Euclidean(inPoints)
%This function maps a set of symmetric matrices to vectors.
%inPoints is a 3D array of size NxNxP which stores P NxN symmetric matrices.
%outPoints is a 2D array of size 0.5N(N+1)xP

[nFeatures,~,nPoints] = size(inPoints);
outPoints = zeros(0.5*nFeatures*(nFeatures+1),nPoints);
tmpSPD = ones(nFeatures);
tmpSPD(tril(tmpSPD) == 1) = 0;
tmpIdx = tmpSPD > 0;
for tmpC1 = 1:nPoints
    tmpSPD = inPoints(:,:,tmpC1);
    outPoints(1:nFeatures,tmpC1) = diag(tmpSPD);
    outPoints(1+nFeatures:end,tmpC1) = sqrt(2)*tmpSPD(tmpIdx);
end 

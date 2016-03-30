function proj_C = RandomFourierFeatures(X, d, gamma)
[D, NPixels, NImages] = size(X);
W = sqrt(2*gamma)*randn(d,D);
bias = 2*pi*rand(d,1);
Bias = repmat(bias, 1, NPixels);

proj_C = zeros(d,d,NImages);

for tmpImage=1:NImages
    proj_covSamples = cos(W*X(:,:,tmpImage) + Bias) / sqrt(d);
    proj_outCov = cov(proj_covSamples');
    proj_outCov = proj_outCov + eps * eye(size(proj_outCov));    
    proj_C(:,:,tmpImage) = proj_outCov;
end


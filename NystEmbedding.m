function proj_C = NystEmbedding(X, d, b)
[D, NPixels, NImages] = size(X);
data = reshape(X, D, NPixels*NImages);
data = data';
[n,dim] = size(data);
dex = randperm(n);
center = data(dex(1:2*d),:);

W = exp(-sqdist(center', center') * b);
E = exp(-sqdist(data', center') * b);
[Ve, Va] = eigs(W,d);
va = diag(Va);
%pidx = find(va > 1e-6);
pidx = 1:d;
inVa = sparse(diag(va(pidx).^(-0.5)));
G = E * Ve(:,pidx) * inVa;
G = G';

Xnew = reshape(G, d, NPixels, NImages);
proj_C = zeros(d,d,NImages);

for tmpImage=1:NImages
    proj_outCov = cov(Xnew(:,:,tmpImage)');
    proj_outCov = proj_outCov + eps * eye(size(proj_outCov));
    proj_C(:,:,tmpImage) = proj_outCov;
end

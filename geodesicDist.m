function d = geodesicDist(X,Y,varargin)
X05 = real(X^-0.5);
temp = X05 * Y * X05;
[V,D] = eig(temp);
logTemp = V * diag(diag(log(D))) * V';
d = trace(logTemp * logTemp);
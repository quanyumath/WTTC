function [X,X0] = prox_ctnn(Y,rho,theta)


[n1,n2,n3] = size(Y);
max12 = max(n1,n2);
X = zeros(n1,n2,n3);X0 = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
S = max(S-rho,0);
tol = max12*eps(max(S));
r = sum(S > tol);
S = S(1:r);
X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
% compute \nabla H(X)
nabla=S/theta;nabla(S>theta)=1;
X0(:,:,1) = U(:,1:r)*diag(nabla)*V(:,1:r)';
tnn = tnn+sum(S);
trank = max(trank,r);

% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = max(S-rho,0);
    tol = max12*eps(max(S));
    r = sum(S > tol);
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
    % compute \nabla H(X)
    nabla = S/theta;nabla(S>theta) = 1;
    X0(:,:,i) = U(:,1:r)*diag(nabla)*V(:,1:r)';
    X(:,:,n3+2-i) = conj(X(:,:,i));
    X0(:,:,n3+2-i) = conj(X0(:,:,i));
    tnn = tnn+sum(S)*2;
    trank = max(trank,r);
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = max(S-rho,0);
    tol = max12*eps(max(S));
    r = sum(S > tol);
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
    % compute \nabla H(X)
    nabla = S/theta;nabla(S>theta) = 1;
    X0(:,:,i) = U(:,1:r)*diag(nabla)*V(:,1:r)';
    tnn = tnn+sum(S);
    trank = max(trank,r);
end
tnn = tnn/n3;
X = ifft(X,[],3);X0 = ifft(X0,[],3);

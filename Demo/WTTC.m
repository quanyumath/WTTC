function [X, A, H, V, D] = WTTC(XX, Omega, opts)
% known: index set of observed entries
% XX: original image
% opts.
%      tol: tolerance for relative change of function value, default:1e-3
%      maxiter: max number of iterations, default: 100

%% Parameters and defaults
if isfield(opts,'maxiter')  maxiter = opts.maxiter;   else maxiter = 100;    end
if isfield(opts,'tol')      tol = opts.tol;           else tol = 1e-3;       end
if isfield(opts,'beta')     beta = opts.beta;         else beta = 1;         end
if isfield(opts,'lambdaA')  lambdaA = opts.lambdaA;   else lambdaA = 0.8;    end
if isfield(opts,'lambdaH')  lambdaH = opts.lambdaH;   else lambdaH = 1;      end
if isfield(opts,'lambdaV')  lambdaV = opts.lambdaV;   else lambdaV = 1;      end
if isfield(opts,'lambdaD')  lambdaD = opts.lambdaD;   else lambdaD = 0.1;    end
if isfield(opts,'gamma')    gamma = opts.gamma;       else gamma = 1e-2;     end
if isfield(opts,'theta')    theta = opts.theta;       else theta = 30;       end
if isfield(opts,'rho')      rho = opts.rho;           else rho = 1;          end
if isfield(opts,'beta_max') beta_max = opts.beta_max; else beta_max = 1;     end

%% Data preprocessing and initialization
X = zeros(size(XX)) + 0.5; X(Omega) = XX(Omega);
t0 = 1; % used for extrapolation weight update
[n1, n2, n3] = size(X);
for k = 1:n3
    [wA(:, :, k), wH(:, :, k), wV(:, :, k), wD(:, :, k)] = dwt2(X(:, :, k), 'haar'); %使用haar小波
end

%% save old variable
X0 = X; A0 = wA; H0 = wH; V0 = wV; D0 = wD;
wA0 = wA; wH0 = wH; wV0 = wV; wD0 = wD;
A = A0; H = H0; V = V0; D = D0;
for iter = 1:maxiter
    bg = beta + gamma;
    %--Updata A
    if iter == 1
        [A, nablaA] = prox_ctnn((beta * wA + gamma * A)/bg, lambdaA/bg, theta(1));
    else
        [A, nablaA] = prox_ctnn((beta * wA + gamma * A + lambdaA * nablaA)/bg, lambdaA/bg, theta(1));
    end
    %--Updata H
    if iter == 1
        [H, nablaH] = prox_ctnn((beta * wH + gamma * H)/bg, lambdaH/bg, theta(1));
    else
        [H, nablaH] = prox_ctnn((beta * wH + gamma * H + lambdaH * nablaH)/bg, lambdaH/bg, theta(1));
    end
    %--Updata V
    if iter == 1
        [V, nablaV] = prox_ctnn((beta * wV + gamma * V)/bg, lambdaV/bg, theta(1));
    else
        [V, nablaV] = prox_ctnn((beta * wV + gamma * V + lambdaV * nablaV)/bg, lambdaV/bg, theta(1));
    end
    %--Updata D
    nabla = D / theta(2); nabla(D > theta(2)) = 1; nabla(D < -theta(2)) = -1;
    G = lambdaD * nabla + beta * wD + gamma * D; G = G / bg;
    D = sign(G) .* max(abs(G)-lambdaD/bg, 0);

    %--Updata X
    for k = 1:n3
        X_i(:, :, k) = idwt2(A(:, :, k), H(:, :, k), V(:, :, k), D(:, :, k), 'haar');
    end
    X_in = beta * X_i(1:n1, 1:n2, 1:n3) + gamma * X0; X = X_in / bg;
    X(Omega) = XX(Omega);

    for k = 1:n3
        [wA(:, :, k), wH(:, :, k), wV(:, :, k), wD(:, :, k)] = dwt2(X(:, :, k), 'haar');
    end
    % check stopping criterion
    if norm(X0(:)-X(:)) / norm(X0(:)) < tol / 10 && iter > 5
        % xi_A
        NA = lambdaA / beta * nablaA; PNA = prox_tnn(wA+NA, lambdaA/beta);
        xi_A = norm(A(:)-PNA(:)) / (1 + norm(wA(:)) + norm(NA(:)));
        % xi_H
        NH = lambdaH / beta * nablaH; PNH = prox_tnn(wH+NH, lambdaH/beta);
        xi_H = norm(H(:)-PNH(:)) / (1 + norm(wH(:)) + norm(NH(:)));
        % xi_V
        NV = lambdaV / beta * nablaV; PNV = prox_tnn(wV+NV, lambdaV/beta);
        xi_V = norm(V(:)-PNV(:)) / (1 + norm(wV(:)) + norm(NV(:)));
        % xi_D
        ND = lambdaD / beta * nabla; PND = sign(wD+ND) .* max(abs(wD+ND)-lambdaD/beta, 0);
        xi_D = norm(D(:)-PND(:)) / (1 + norm(wD(:)) + norm(ND(:)));

        xi = [xi_A, xi_H, xi_V, xi_D];
        %      cc(iter) = max(xi)
        if max(xi) < tol*10
            break
        end
    end

    beta = min(beta*rho, beta_max);
    % --- extrapolation ---
    t = (1 + sqrt(1+4*t0^2)) / 2; wt = (t0 - 1) / t;
    wA = wA + wt * (wA - wA0); wH = wH + wt * (wH - wH0);
    wV = wV + wt * (wV - wV0); wD = wD + wt * (wD - wD0);

    %--Save old variable
    X0 = X; wA0 = wA; wH0 = wH; wV0 = wV; wD0 = wD;

    if mod(iter, 5) == 0 t0 = 1; else t0 = t; end
end
end
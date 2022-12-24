currentFolder = pwd; addpath(genpath(currentFolder));
clc
clear
indimgs = [1:5];
i = 1; id = indimgs(i);
pic_name = ['ColorImage/', num2str(id), '.tiff'];
I = double(imread(pic_name));
X = I / 255;

%% Ñù±¾ÂÊ
sr = 0.3;
Omega = find(rand(numel(X), 1) < sr);
H = zeros(size(X)); H(Omega) = X(Omega); %imshow(H) imwrite(H,'11.tiff');

%% WTTC
opts = [];
opts.maxiter = 150; opts.tol = 5e-3;
opts.beta = 5e-1; opts.rho = 1.06; opts.beta_max = 10; 
opts.gamma = 1e-3; opts.theta = [20, 0.6];
opts.lambdaA = 2; opts.lambdaH = 2; opts.lambdaV = 2; opts.lambdaD = 5;
tic
[OX, wA, wH, wV, wD] = WTTC(X, Omega, opts); %imshow(OX) imshow(wA) imshow(wH+0.5) imshow(wV+0.5) imshow(wD+0.5)
time_WTTC = toc;
[psnr_WTTC, ssim_WTTC, fsim_WTTC] = quality(X, OX);

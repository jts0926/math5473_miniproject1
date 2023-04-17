%% load results data
clear all;
load("results\method1_fastrpca.mat")
X_m1 = X;
L_m1 = L;
S_m1 = S;

load("results\method2_accaltproj_rpca.mat")
X_m2 = X;
L_m2 = L;
S_m2 = S;

%% display
% method 1
mat  = @(x) reshape( x, [256, 320] );
figure(1); clf;
colormap( 'Gray' );
nFrames = 1000;

for k = 1:nFrames
    subplot(2,1,1)
    imagesc( [mat(X_m1(:,k)), mat(L_m1(:,k)),  mat(S_m1(:,k))] );
    title("Fast Robust Principal Component Analysis")
    axis off
    axis image
    drawnow;
    subplot(2,1,2)
    imagesc( [mat(X_m2(k,:)), mat(L_m2(k,:)),  mat(S_m2(k,:))] );
    title("Accelerated Alternating Projections for Robust Principal Component Analysis")
    axis off
    axis image
    drawnow;
    pause(.05);  
end
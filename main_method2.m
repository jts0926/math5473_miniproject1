%% use AccAltProj for RPCA
clear all;
load("data\shoppingmall.mat"); % contains X (data), m and n (height and width)
m=frame_m;
n=frame_n;
nFrames = size(shoppingmall,1);
X = shoppingmall;
%% parameters setting
%para.mu        = [5,10];
%para.beta_init = 0.8;
%para.beta      = 0.4;
%para.trimming  = true;
%para.tol       = 1e-5;
%para.gamma     = 0.6;
%para.max_iter  = 100;


%% Run
[L, S] = AccAltProj( X, 10, []); % use the default parameters

%% show all together in movie format
mat  = @(x) reshape( x, [frame_m, frame_n] );
figure(1); clf;
colormap( 'Gray' );
for k = 1:nFrames
    imagesc( [mat(X(k,:)), mat(L(k,:)),  mat(S(k,:))] );
    axis off
    axis image
    drawnow;
    pause(.05);  
end

%% save results
save("results/method2_accaltproj_rpca.mat", "X", "L", "S");
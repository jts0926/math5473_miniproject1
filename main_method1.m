%% use fast RPCA
load("data\shoppingmall.mat"); % contains X (data), m and n (height and width)
X = double(shoppingmall)';
nFrames = size(X,2);

%% Uncomment to Run the SPGL1 algorithm
%{
We solve
    min_{L,S}  max( ||L||_* , lambda ||S||_1 )
    subject to
            || L+S-X ||_F <= epsilon
%}


% lambda      = 2e-2;
% L0          = repmat( median(X,2), 1, nFrames );
% S0          = X - L0;
% epsilon     = 5e-3*norm(X,'fro'); % tolerance for fidelity to data
% opts        = struct('sum',false,'L0',L0,'S0',S0,'max',true,...
%     'tau0',3e5,'SPGL1_tol',1e-1,'tol',1e-3,'printEvery',1);
% 
% [L,S] = solver_RPCA_SPGL1(X,lambda,epsilon,[],opts);

%% Uncomment to Solve Lagrangian Version
% opts    = struct('sum',false,'max',false,'tol',1e-8,'printEvery',1);
% lambdaL = 115;
% lambdaS = 0.8;
% 
% [L,S] = solver_RPCA_Lagrangian(X,lambdaL,lambdaS,[],opts);


%% Run the Split-SPCP Algorithm
% Implements Split-SPCP method from ``Adapting Regularized Low-Rank Models 
% for Parallel Architectures'' (SIAM J. Sci. Comput., 2019).

% Set options for L-BFGS
params.progTol = 1e-10;
params.optTol  = 1e-10;
params.MaxIter = 500;
params.store    = 1;

% Set parameters for split_SPCP
params.k   = 10; % rank bound on L
params.gpu = 0;  % set to 1 to run on GPU
params.lambdaL = 115;
params.lambdaS = 0.8;

[L,S] = solver_split_SPCP(X,params);
%% show all together in movie format

mat  = @(x) reshape( x, m, n );
figure(1); clf;
colormap( 'Gray' );
for k = 1:nFrames
    imagesc( [mat(X(:,k)), mat(L(:,k)),  mat(S(:,k))] );
    axis off
    axis image
    drawnow;
    pause(.05);  
end

%% save results
save("results/method1_fastrpca.mat", "X", "L", "S");
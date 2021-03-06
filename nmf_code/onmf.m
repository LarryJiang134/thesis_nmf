% TODO:  check   1) implement onmf with general diverence as introduced in the paper
%                            2) vectorize equation
%
% Online Nonnegative Matrix Factorization with General Divergences
%
% <Inputs>
%        V : Input data matrix (m x n)
%        k : Target low-rank
%        W : The last W matrix
%
%        (Below are parameters pre-defined)
%        
%        MAX_ITER : Maximum number of iterations. Default is 100.
%        MIN_ITER : Minimum number of iterations. Default is 20.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x k) initial value for W.
%        H_INIT : (k x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
% <Outputs>
%        W : Obtained basis matrix (m x k)
%        H : Obtained coefficients matrix (k x n)
%
% Here we set V to be known instead of a stream of input

function [W] = onmf(V, k, W)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes her
% Default configuration
btk = 0;
acc = 0; 
% disp(size(W));
% disp(size(H));
[m, ~] = size(W);
% [~, n] = size(H);
par.m = m;
% par.n = n;
par.max_iter = 100;
par.max_time = 1e6;
par.tol = 1e-3;

ht = rand(k,1);
ht = learning_h_t(ht, W, V, btk, 100);
temp2 = (V * ht') ./ (W * ht * ht');
% disp(min(temp2));
W = W .* temp2;
% W(W<0) = 0;
% W(W>1) = 1;
min_W = min(min(W)');
max_W = max(max(W)');
% disp(min_W);
if min_W < 0
    if max_W + abs(min_W) > 1
        W = (W + abs(min_W)) / (max_W - min_W);
    else
        W = (W + abs(min_W));
    end
else
    if max_W > 1
        W = W / max_W;
    end
end
end

function ht = learning_h_t(ht, Wt1, V, btk, g)
% this is corresponding to algorithm 2 of the paper
% <Inputs>
%       ht: initial coefficient vector h_t_0
%       Wt1: basis matrix W_(t-1)
%       vt: data sample
%       btk: step size beta_t_k
%       g: maximum number of iterations gama
%<Outputs>
%       ht: final coefficient vector h_t := h_t_gama

% Try multiplicative update rule proposed in Lee & Seung's "Algorithms for Non-negative Matrix Factorization"
% as a good compromise of between speed and ease of implementation.
% Here we can ignore btk for now.

for k = 1:g
    temp1 = (Wt1' * V) ./ (Wt1' * Wt1 * ht);
    ht = ht .* temp1;
    ht(ht<0) = 0;
    ht(ht>1) = 1;
end
end

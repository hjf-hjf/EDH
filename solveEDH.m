function [B,U1,U2,R1,W,R2] = solveEDH( X1, X2, L, param)
lambda = param.lambda;
mu = param.mu;
gamma = param.gamma;
alpha = param.alpha;
belta = param.belta;
maxIter = param.maxIter;
bits = param.nbits;
%% Initialization
row = size(X1,1);
rowt = size(X2,1);
[c,n] = size(L);
U1 = rand(row, bits);
U2 = rand(rowt, bits);
V = rand(bits,n);
W = rand(c,bits);
F = rand(bits,n);
I = eye(bits);
P = I;
opts.mxitr  = 30;
opts.xtol = 1e-4;
opts.gtol = 1e-4;
opts.ftol = 1e-5;

for i = 1:param.maxIter
        fprintf('iteration %3d\n', i);
        
        % update B
        B = -1*ones(bits,n);
        X = mu * P * V + (1-mu) * P * F;
        B(X>=0) = 1;

        % update U1 and U2
        U1 = (lambda * X1  * V') / (lambda * V * V' + gamma * eye(bits));
        U2 = ((1-lambda) * X2  * V') / ((1-lambda) * V * V' + gamma * eye(bits));

        % update V
        V_1 = (lambda * U1' * U1 + (1-lambda) * U2' * U2 + mu * P'*P + gamma * eye(bits)); 
        V_2 = (lambda * U1' * X1 + (1-lambda) * U2' * X2 + mu * P' * B);
        V = (V_1 \ V_2);

        % update W
        W = (alpha * L * F') / (alpha * F * F' + gamma * eye(bits));

        % update F
        
        F = (alpha * W' * W + (1-mu) * P' * P + gamma * eye(bits)) \ (alpha * W' * L + (1-mu) * P' * B);

        % update P
        [P, ~]= OptStiefelGBB(P, @funP, opts, V, F, B, mu);
        
    end

end
function [F,G] = funP(P, V, F, B, mu)
G =  2 * mu * (P*V*V'-B*V') + 2 * (1-mu) * (P*F*F'-B*F');
F =  mu * norm(P*V -B, 'fro')^2 + (1-mu)* norm(P*F -B, 'fro')^2;
end

function [E, resp, info] = log_edge(I, params)
% Laplacian of Gaussian (Marr-Hildreth) + zero-crossing.
    if nargin < 2, params = default_edge_params(); end
    s = params.logSigma;
    n = ceil(3 * s);
    [x, y] = meshgrid(-n:n, -n:n);
    r2 = x.^2 + y.^2;
    K = ((r2 - 2 * s^2) / s^4) .* exp(-r2 / (2 * s^2));
    K = K - mean(K(:));

    resp = apply_kernel(I, K);
    thr = zero_crossing_threshold(resp, params.zcRatio, params.zcNoiseK);
    E = zero_crossing(resp, thr);
    if params.gateEdges
        E = E & (gradient_gate(I, params.logSigma, params.sensitivity));
    end
    info.threshold = thr;
end
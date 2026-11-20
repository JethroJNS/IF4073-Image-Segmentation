function [E, resp, info] = laplace_edge(I, params)
% Operator Laplace (turunan kedua) + deteksi zero-crossing
    if nargin < 2, params = default_edge_params(); end
    if params.laplaceConn == 4
        K = [0 1 0; 1 -4 1; 0 1 0];
    else
        K = [1 1 1; 1 -8 1; 1 1 1];
    end

    G = smooth_image(I, params.laplaceSigma);
    resp = apply_kernel(G, K);
    thr = zero_crossing_threshold(resp, params.laplaceRatio, params.zcNoiseK);
    E = zero_crossing(resp, thr);
    if params.gateEdges
        E = E & (gradient_gate(I, params.laplaceSigma, params.sensitivity));
    end
    info.threshold = thr;
end
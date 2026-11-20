function [E, mag, info] = roberts_edge(I, params)
% Operator Roberts cross 2x2
    if nargin < 2, params = default_edge_params(); end
    Kx = [1 0; 0 -1];
    Ky = [0 1; -1 0];
    [E, mag, info] = gradient_edge(I, Kx, Ky, params);
end
function [E, mag, info] = prewitt_edge(I, params)
% Operator Prewitt
    if nargin < 2, params = default_edge_params(); end
    Kx = [-1 0 1; -1 0 1; -1 0 1];
    Ky = Kx';
    [E, mag, info] = gradient_edge(I, Kx, Ky, params);
end
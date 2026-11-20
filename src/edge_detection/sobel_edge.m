function [E, mag, info] = sobel_edge(I, params)
% Operator Sobel
    if nargin < 2, params = default_edge_params(); end
    Kx = [-1 0 1; -2 0 2; -1 0 1];
    Ky = Kx';
    [E, mag, info] = gradient_edge(I, Kx, Ky, params);
end
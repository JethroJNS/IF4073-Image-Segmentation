function [E, resp, info] = canny_edge(I, params)
% Operator Canny
    if nargin < 2, params = default_edge_params(); end
    resp = [];
    info.backend = 'builtin';
    try
        if isempty(params.cannyThresh)
            E = edge(I, 'canny', [], params.cannySigma);
        else
            E = edge(I, 'canny', params.cannyThresh, params.cannySigma);
        end
        E = logical(E);
    catch
        info.backend = 'fallback';
        [E, resp] = canny_fallback(I, params);
    end
end

function [E, mag] = canny_fallback(I, params)
    % Penghalusan + gradien (Sobel)
    G = smooth_image(I, params.cannySigma);
    Kx = [-1 0 1; -2 0 2; -1 0 1];
    gx = apply_kernel(G, Kx);
    gy = apply_kernel(G, Kx');
    mag = sqrt(gx.^2 + gy.^2);
    [r, c] = size(mag);

    % Non-maximum suppression berdasarkan arah gradien (4 arah)
    ang = mod(atan2(gy, gx) * 180 / pi, 180);
    P = mag([1 1:r r], [1 1:c c]);
    nE = P(2:end-1, 3:end);    nW = P(2:end-1, 1:end-2);
    nN = P(1:end-2, 2:end-1);  nS = P(3:end, 2:end-1);
    nNW = P(1:end-2, 1:end-2); nSE = P(3:end, 3:end);
    nNE = P(1:end-2, 3:end);   nSW = P(3:end, 1:end-2);

    d0   = (ang < 22.5) | (ang >= 157.5);
    d45  = (ang >= 22.5) & (ang < 67.5);
    d90  = (ang >= 67.5) & (ang < 112.5);
    d135 = (ang >= 112.5) & (ang < 157.5);
    keep = (d0 & mag >= nE & mag >= nW) | (d45 & mag >= nSE & mag >= nNW) | ...
           (d90 & mag >= nN & mag >= nS) | (d135 & mag >= nNE & mag >= nSW);
    nms = mag .* keep;

    % Hysteresis threshold
    mn = normalize01(nms);
    if isempty(params.cannyThresh)
        hi = otsu_threshold(normalize01(mag));
        lo = 0.4 * hi;
    else
        lo = params.cannyThresh(1); hi = params.cannyThresh(end);
    end
    strong = mn >= hi & keep;
    weak   = mn >= lo & keep;
    [Lw, ~] = label_components(weak, 8);
    ids = unique(Lw(strong));
    ids = ids(ids > 0);
    E = ismember(Lw, ids) & (Lw > 0);
end
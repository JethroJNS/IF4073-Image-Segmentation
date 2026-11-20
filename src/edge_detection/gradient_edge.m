function [E, mag, info] = gradient_edge(I, Kx, Ky, params)
% Pipeline umum operator berbasis turunan pertama (Sobel/Prewitt/Roberts)
    G  = smooth_image(I, params.smoothSigma);
    gx = apply_kernel(G, Kx);
    gy = apply_kernel(G, Ky);
    mag = normalize01(sqrt(gx.^2 + gy.^2));

    if isempty(params.threshold)
        T = params.sensitivity * otsu_threshold(mag);
    else
        T = params.threshold;
    end
    E = mag > T;
    info.threshold = T;
end
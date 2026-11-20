function K = gaussian_kernel(sigma)
% Kernel Gaussian 2D ternormalisasi
    n = ceil(3 * sigma);
    [x, y] = meshgrid(-n:n, -n:n);
    K = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    K = K / sum(K(:));
end
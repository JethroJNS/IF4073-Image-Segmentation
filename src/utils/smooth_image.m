function G = smooth_image(I, sigma)
% Penghalusan Gaussian
    if sigma <= 0
        G = I;
    else
        G = apply_kernel(I, gaussian_kernel(sigma));
    end
end
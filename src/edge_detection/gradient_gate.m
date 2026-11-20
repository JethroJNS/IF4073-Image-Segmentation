function keep = gradient_gate(I, sigma, sensitivity)
% Mask piksel dengan gradien kuat (untuk menyaring zero-crossing lemah)
    G = smooth_image(I, sigma);
    Kx = [-1 0 1; -2 0 2; -1 0 1];
    mag = normalize01(sqrt(apply_kernel(G, Kx).^2 + apply_kernel(G, Kx').^2));
    keep = mag > sensitivity * otsu_threshold(mag);
    keep = conv2(double(keep), ones(3), 'same') > 0;
end
function R = apply_kernel(I, K)
% Korelasi 2D citra dengan kernel K
    [r, c] = size(I);
    [kr, kc] = size(K);

    % Jumlah baris/kolom padding di tiap sisi
    top = floor((kr - 1) / 2);  bottom = ceil((kr - 1) / 2);
    left = floor((kc - 1) / 2); right = ceil((kc - 1) / 2);

    % Indeks replikasi tepi
    ri = [ones(1, top), 1:r, r * ones(1, bottom)];
    ci = [ones(1, left), 1:c, c * ones(1, right)];
    P = I(ri, ci);

    % Korelasi = konvolusi dengan kernel yang diputar 180 derajat
    R = conv2(P, rot90(K, 2), 'valid');
end
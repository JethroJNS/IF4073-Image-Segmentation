function Z = zero_crossing(L, thr)
% Deteksi zero-crossing pada respons turunan kedua (Laplace/LoG)
    [r, c] = size(L);
    P = L([1 1:r r], [1 1:c c]);               % replicate padding 1 piksel
    C = P(2:end-1, 2:end-1);

    % Untuk tiap 8 tetangga N: piksel C ditandai bila C dan N berbeda tanda,
    % kemiringan |C-N| > thr, dan C lebih dekat ke nol daripada N
    % (sehingga hanya satu sisi dari tiap perpindahan tanda yang ditandai -> tepi tipis).
    Z = false(r, c);
    for dy = -1:1
        for dx = -1:1
            if dy == 0 && dx == 0, continue; end
            N = P(2+dy:end-1+dy, 2+dx:end-1+dx);
            Z = Z | ((C .* N < 0) & (abs(C - N) > thr) & (abs(C) <= abs(N)));
        end
    end
end
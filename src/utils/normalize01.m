function N = normalize01(M)
% Normalisasi matriks ke [0,1]
    mx = max(M(:));
    if mx > 0
        N = M / mx;
    else
        N = M;
    end
end
function G = to_gray(A)
% Konversi RGB ke grayscale
    if isinteger(A)
        A = double(A) / double(intmax(class(A)));
    else
        A = double(A);
        if max(A(:)) > 1
            A = A / 255;
        end
    end
    if ndims(A) == 3 && size(A, 3) >= 3
        G = 0.299 * A(:, :, 1) + 0.5870 * A(:, :, 2) + 0.114 * A(:, :, 3);
    else
        G = A(:, :, 1);
    end
end
function [rgb, gray] = load_image(img)
    map = [];
    if ischar(img) || (exist('isstring', 'builtin') && isstring(img))
        img = char(img);
        if ~exist(img, 'file')
            error('load_image:notFound', 'File citra tidak ditemukan: %s', img);
        end
        [A, map] = imread(img);
    else
        A = img;
    end

    % Citra terindeks -> RGB
    if ~isempty(map)
        A = uint8(round(255 * ind2rgb(A, map)));
    end

    % Buang kanal alpha bila ada
    if ndims(A) == 3 && size(A, 3) == 4
        A = A(:, :, 1:3);
    end

    % Konversi tipe data ke uint8
    if isa(A, 'uint16')
        A = uint8(double(A) / 257);
    elseif isa(A, 'double') || isa(A, 'single')
        A = double(A);
        if max(A(:)) <= 1
            A = A * 255;
        end
        A = uint8(round(min(max(A, 0), 255)));
    elseif islogical(A)
        A = uint8(A) * 255;
    elseif ~isa(A, 'uint8')
        A = uint8(A);
    end

    % Grayscale -> RGB (replikasi kanal)
    if ndims(A) == 2
        rgb = repmat(A, [1 1 3]);
    else
        rgb = A;
    end

    gray = to_gray(rgb);
end
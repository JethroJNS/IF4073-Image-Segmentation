function out = overlayBoundaries(img, bw, color)
% Gambar garis batas region di atas citra
    if nargin < 3
        color = [255 0 0];
    end
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    out = img;
    for c = 1:3
        ch = out(:, :, c);
        ch(bw) = color(c);
        out(:, :, c) = ch;
    end
end
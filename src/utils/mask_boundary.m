function BD = mask_boundary(mask, thickness)
% Batas dari mask biner; thickness = ketebalan (piksel)
    if nargin < 2, thickness = 1; end
    mask = logical(mask);
    BD = mask & ~binary_erode(mask, disk_element(1));
    % Pastikan sisi citra tidak dianggap batas palsu
    if thickness > 1
        BD = binary_dilate(BD, disk_element(floor(thickness / 2)));
    end
end
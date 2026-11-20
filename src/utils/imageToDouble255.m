function X = imageToDouble255(img, floatRange)
% Ubah citra apa pun menjadi double berskala [0,255]
    if nargin < 2, floatRange = 'auto'; end
    if isstring(floatRange) && isscalar(floatRange), floatRange = char(floatRange); end
    if ~(ischar(floatRange) && any(strcmpi(floatRange, {'auto', 'unit', 'byte'})))
        error('splitMerge:invalidImage', ...
              'floatRange harus ''auto'', ''unit'', atau ''byte''.');
    end
    floatRange = lower(floatRange);

    if isempty(img) || ~(isnumeric(img) || islogical(img))
        error('splitMerge:invalidImage', ...
              'Citra masukan harus berupa matriks numerik yang tidak kosong.');
    end
    if ndims(img) > 3
        error('splitMerge:invalidImage', 'Citra harus berdimensi HxW atau HxWxC.');
    end

    % Normalisasi skala berdasarkan kelas data aslinya
    if islogical(img)
        X = double(img) * 255;
    elseif isinteger(img)
        lo = double(intmin(class(img)));
        hi = double(intmax(class(img)));
        X  = (double(img) - lo) / ((hi - lo) / 255);   % [intmin,intmax] -> [0,255]
    else                                               % single / double
        X = double(img);
        switch floatRange
            case 'unit'
                X = X * 255;
            case 'auto'
                if all(isfinite(X(:))) && max(X(:)) <= 1 && min(X(:)) >= 0
                    X = X * 255;                       % citra float berskala 0..1
                end
            % 'byte': dipakai apa adanya
        end
    end

    if any(~isfinite(X(:)))
        error('splitMerge:invalidImage', 'Citra mengandung NaN/Inf.');
    end

    % Pastikan jumlah kanal 1 atau 3
    C = size(X, 3);
    if C == 2
        X = X(:, :, 1);
    elseif C >= 4
        X = X(:, :, 1:3);
    end

    X = min(max(X, 0), 255);
end
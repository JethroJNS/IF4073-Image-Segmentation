function params = splitMergeParams(varargin)
% Parameter default + validasi untuk split and merge
    params = struct( ...
        'SplitCriterion',        'std', ...
        'SplitThreshold',        12, ...
        'MinBlockSize',          4, ...
        'MergeThreshold',        20, ...
        'MergeStdMax',           Inf, ...
        'MinRegionSize',         30, ...
        'SmallRegionDistFactor', Inf, ...
        'ColorMode',             'color', ...
        'ColorSpace',            'rgb', ...
        'FloatRange',            'auto');

    if nargin == 1 && isstruct(varargin{1})
        names  = fieldnames(varargin{1});
        values = struct2cell(varargin{1});
    elseif mod(nargin, 2) == 0
        names  = varargin(1:2:end);
        values = varargin(2:2:end);
    else
        error('splitMerge:invalidParam', ...
              'Parameter harus berupa pasangan nama-nilai atau sebuah struct.');
    end

    names  = cellfun(@toChar, names(:),  'UniformOutput', false);
    values = cellfun(@toChar, values(:), 'UniformOutput', false);

    known = fieldnames(params);
    for k = 1:numel(names)
        if ~ischar(names{k})
            error('splitMerge:invalidParam', 'Nama parameter harus berupa string.');
        end
        idx = find(strcmpi(names{k}, known), 1);
        if isempty(idx)
            error('splitMerge:invalidParam', 'Parameter tidak dikenal: %s', names{k});
        end
        params.(known{idx}) = values{k};
    end

    params.SplitCriterion = requireChoice(params.SplitCriterion, 'SplitCriterion', {'std', 'range'});
    params.ColorMode      = requireChoice(params.ColorMode,      'ColorMode',      {'color', 'gray'});
    params.ColorSpace     = requireChoice(params.ColorSpace,     'ColorSpace',     {'rgb', 'lab'});
    params.FloatRange     = requireChoice(params.FloatRange,     'FloatRange',     {'auto', 'unit', 'byte'});

    requireScalar(params.SplitThreshold, 'SplitThreshold', 0, false);
    requireScalar(params.MergeThreshold, 'MergeThreshold', 0, false);
    requireScalar(params.MinBlockSize,   'MinBlockSize',   1, true);
    requireScalar(params.MinRegionSize,  'MinRegionSize',  0, true);

    requirePositiveOrInf(params.MergeStdMax,           'MergeStdMax');
    requirePositiveOrInf(params.SmallRegionDistFactor, 'SmallRegionDistFactor');
end

function x = toChar(x)
    if isstring(x) && isscalar(x)
        x = char(x);
    end
end

function out = requireChoice(v, name, options)
    if ~(ischar(v) && isrow(v) && any(strcmpi(v, options)))
        error('splitMerge:invalidParam', '%s harus salah satu dari: %s.', ...
              name, strjoin(options, ', '));
    end
    out = lower(v);
end

function requirePositiveOrInf(v, name)
    if ~(isnumeric(v) && isreal(v) && isscalar(v) && ~isnan(v) && v > 0)
        error('splitMerge:invalidParam', '%s harus skalar real > 0 (boleh Inf).', name);
    end
end

function requireScalar(v, name, minVal, mustBeInteger)
    ok = isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && v >= minVal;
    if ok && mustBeInteger
        ok = (v == round(v));
    end
    if ~ok
        if mustBeInteger
            error('splitMerge:invalidParam', ...
                  '%s harus bilangan bulat >= %d.', name, minVal);
        else
            error('splitMerge:invalidParam', ...
                  '%s harus skalar terhingga >= %d.', name, minVal);
        end
    end
end
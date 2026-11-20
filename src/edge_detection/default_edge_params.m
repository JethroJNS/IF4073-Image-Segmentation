function p = default_edge_params(user)
    p.smoothSigma = 1.5;
    p.threshold   = [];
    p.sensitivity = 1.0;
    p.laplaceConn = 8;
    p.laplaceSigma = 1.5;
    p.laplaceRatio = 0.25;
    p.logSigma    = 2.0;
    p.zcRatio     = 0.08;
    p.zcNoiseK    = 1;
    p.gateEdges   = true;
    p.cannySigma  = 1.5;
    p.cannyThresh = [];

    p.closeRadii  = 1:8;
    p.openRadius  = 2;
    p.minAreaFrac = 0.01;
    p.minFrac     = 0.03;
    p.maxFrac     = 0.80;
    p.numObjects  = 1;
    p.fillBorderObjects = true;

    p.centerWeight  = 0.25;
    p.borderPenalty = 0.8;
    p.extentWeight  = 0.5;

    if nargin >= 1 && isstruct(user)
        f = fieldnames(user);
        for i = 1:numel(f)
            if ~isfield(p, f{i})
                warning('default_edge_params:unknownField', ...
                        'Parameter "%s" tidak dikenal (kemungkinan typo).', f{i});
            end
            p.(f{i}) = user.(f{i});
        end
    end
end
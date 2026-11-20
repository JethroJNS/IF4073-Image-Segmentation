function thr = zero_crossing_threshold(resp, ratio, noiseK)
% Ambang kemiringan zero-crossing yang adaptif terhadap derau
    a = abs(resp(:));
    sigmaN = median(a) / 0.6745;
    thr = max(ratio * max(a), noiseK * sqrt(2) * sigmaN);
end
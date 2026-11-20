function [L, n] = label_components(B, conn)
% Pelabelan connected component
    if nargin < 2, conn = 8; end
    if conn == 4
        offs = [-1 0; 1 0; 0 -1; 0 1];
    else
        offs = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    end

    B = logical(B);
    [r, c] = size(B);
    L = zeros(r, c);
    n = 0;
    stack = zeros(r * c, 1);

    for p0 = 1:numel(B)
        if B(p0) && L(p0) == 0
            n = n + 1;
            sp = 1; stack(1) = p0; L(p0) = n;
            while sp > 0
                p = stack(sp); sp = sp - 1;
                y = mod(p - 1, r) + 1;
                x = floor((p - 1) / r) + 1;
                for k = 1:size(offs, 1)
                    yy = y + offs(k, 1); xx = x + offs(k, 2);
                    if yy >= 1 && yy <= r && xx >= 1 && xx <= c
                        q = yy + (xx - 1) * r;
                        if B(q) && L(q) == 0
                            L(q) = n;
                            sp = sp + 1; stack(sp) = q;
                        end
                    end
                end
            end
        end
    end
end
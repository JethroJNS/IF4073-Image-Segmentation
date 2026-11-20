function O = binary_open(B, SE)
    O = binary_dilate(binary_erode(B, SE), SE);
end
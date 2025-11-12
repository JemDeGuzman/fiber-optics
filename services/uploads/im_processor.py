import numpy as np
import cv2

# load images (grayscale or single channel)
#I0 = cv2.imread('I0.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
#I45 = cv2.imread('I45.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
#I90 = cv2.imread('I90.png', cv2.IMREAD_UNCHANGED).astype(np.float32)

test = cv2.imread('1761521896481-mitjlj.jpg')

#b, g, r = cv2.split(test)
cv2.imshow('test', test)
#cv2.imshow('b', b)
#cv2.imshow('g', g)
#cv2.imshow('r', r)
"""# optional: dark subtraction / flat-field already done

# Stokes (linear polarization)
S0 = I0 + I90
S1 = I0 - I90
S2 = 2.0 * I45 - S0

# avoid divide by zero
eps = 1e-8
DoLP = np.sqrt(S1**2 + S2**2) / (S0 + eps)
AoP = 0.5 * np.arctan2(S2, S1)  # radians; convert to degrees if needed

# simple specular mask
DoLP_thresh = 0.25
intensity_thresh = np.percentile(S0, 60)  # example
spec_mask = (DoLP > DoLP_thresh) & (S0 > intensity_thresh)

# luster proxy: local max of S0*DoLP
luster_map = S0 * DoLP
# roughness proxy (local peak DoLP inverted)
# compute local max filter
kernel = np.ones((9,9), np.uint8)
local_max = cv2.dilate(DoLP, kernel)
# normalize to 0..1
peak = (local_max - local_max.min()) / (local_max.max() - local_max.min() + eps)
roughness_proxy = 1.0 - peak  # higher -> rougher

# save results
cv2.imwrite('DoLP.png', np.uint8(np.clip(DoLP*255,0,255)))
cv2.imwrite('AoP_deg.png', np.uint8(np.clip((AoP*180/np.pi + 180)/360*255,0,255)))
cv2.imwrite('spec_mask.png', np.uint8(spec_mask*255))
cv2.imwrite('luster_map.png', np.uint8(np.clip(luster_map/luster_map.max()*255,0,255)))
cv2.imwrite('roughness_proxy.png', np.uint8(np.clip(roughness_proxy*255,0,255)))
"""
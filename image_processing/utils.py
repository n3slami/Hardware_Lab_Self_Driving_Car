import numpy as np
import matplotlib.pyplot as plt


SLOPE_FILTER_THRESHOLD = 0.25
BOTTOM_IMAGE_SLOPE_FILTER_THRESHOLD = 0.4
BOTTOM_IMAGE_RATIO = 0.9


def is_valid_line(line, H, W):
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < SLOPE_FILTER_THRESHOLD:
        return False
    if max(x1, x2) > W // 2 and slope < 0:
        return False
    if min(x1, x2) < W // 2 and slope > 0:
        return False
    if max(y1, y2) > H * BOTTOM_IMAGE_RATIO and abs(slope) < BOTTOM_IMAGE_SLOPE_FILTER_THRESHOLD:
        return False
    return True


def line_segment_dist(a, b, clamp_all=True):
    ''' 
    Given two lines defined by numpy.array pairs (a0, a1, b0, b1)
    Return the closest points on each segment and their distance
    '''
    a0, a1, b0, b1 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    a0[:2], a1[:2] = a[:2], a[2:]
    b0[:2], b1[:2] = b[:2], b[2:]

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    mag_a = np.linalg.norm(A)
    mag_b = np.linalg.norm(B)
    
    _A = A / mag_a
    _B = B / mag_b
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2
    
    # If lines are parallel (denom = 0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))
        # Overlap only possible with clamping
        d1 = np.dot(_A, (b1 - a0))
        # Is segment B before A?
        if d0 <= 0 >= d1:
            if np.absolute(d0) < np.absolute(d1):
                return a0, b0, np.linalg.norm(a0 - b0)
            return a0, b1, np.linalg.norm(a0 - b1)
            
        # Is segment B after A?
        elif d0 >= mag_a <= d1:
            if np.absolute(d0) < np.absolute(d1):
                return a1, b0, np.linalg.norm(a1 - b0)
            return a1, b1, np.linalg.norm(a1 - b1)
                
        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    det_A = np.linalg.det([t, _B, cross])
    det_B = np.linalg.det([t, _A, cross])

    t0 = det_A / denom
    t1 = det_B / denom

    p_A = a0 + (_A * t0) # Projected closest point on segment A
    p_B = b0 + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if clamp_all:
        if t0 < 0:
            p_A = a0
        elif t0 > mag_a:
            p_A = a1
        
        if t1 < 0:
            p_B = b0
        elif t1 > mag_b:
            p_B = b1
            
        # Clamp projection A
        if t0 < 0 or t0 > mag_a:
            dot = np.dot(_B, (p_A - b0))
            if dot < 0:
                dot = 0
            elif dot > mag_b:
                dot = mag_b
            p_B = b0 + (_B * dot)
    
        # Clamp projection B
        if t1 < 0 or t1 > mag_b:
            dot = np.dot(_A, (p_B - a0))
            if dot < 0:
                dot = 0
            elif dot > mag_a:
                dot = mag_a
            p_A = a0 + (_A * dot)
    
    return p_A, p_B, np.linalg.norm(p_A - p_B)

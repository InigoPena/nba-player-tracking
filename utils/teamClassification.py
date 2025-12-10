import cv2
import numpy as np


def classify_team_by_uniform(frame, bbox, view=False):

    if bbox is None:
        roi = frame
    else:
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 'blanco'
            
        roi = frame[y:y+h, x:x+w]

    if view:
        cv2.imshow("1. ROI Original", roi)
    
    h_roi, w_roi = roi.shape[:2]
    
    torso_y1 = int(h_roi * 0.20)
    torso_y2 = int(h_roi * 0.70)
    torso_x1 = int(w_roi * 0.20)
    torso_x2 = int(w_roi * 0.80)
    
    if torso_y2 - torso_y1 < 5 or torso_x2 - torso_x1 < 5:
        torso = roi
    else:
        torso = roi[torso_y1:torso_y2, torso_x1:torso_x2]
    
    if view:
        cv2.imshow("2. Torso", torso)
    
    # Conversion LAB
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    mask_not_skin = (a_channel < 135) | (b_channel < 135)
    
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask_not_court = (h < 5) | (h > 35) | (s < 40)
    mask_uniform = mask_not_skin & mask_not_court

    valid_luminosity = l_channel[mask_uniform]
    
    if len(valid_luminosity) < 30:
        valid_luminosity = l_channel.flatten()
    
    if len(valid_luminosity) < 10:
        return 'blanco'
    
    # Análisis por percentiles
    p25 = np.percentile(valid_luminosity, 25)  # Cuartil inferior
    p75 = np.percentile(valid_luminosity, 75)  # Cuartil superior
    median_l = np.median(valid_luminosity)
    
    if p75 < 80:
        team = 'negro'
    elif p25 > 140:
        team = 'blanco'
    else:
        team = 'blanco' if median_l > 110 else 'negro'
    
    return team

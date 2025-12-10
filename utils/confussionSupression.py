import cv2
import numpy as np
from mask.courtMask import CourtMaskGenerator


class CrowdSuppressor:
    
    def __init__(self):
        self.court_mask_generator = CourtMaskGenerator()
        self.court_mask = None
    
    def initialize_mask(self, frame):
        if self.court_mask is None:
            print("Generando máscara de la pista para filtrado de público...")
        self.court_mask, _ = self.court_mask_generator.court_mask(frame)
    
    def touches_court(self, bbox):

        if self.court_mask is None:
            return True  # Si no hay máscara, aceptar todas las detecciones
        
        x, y, w, h = bbox
        
        # Extraer la región de los PIES (último 5% de altura del bbox)
        feet_height = max(int(h * 0.05), 5)
        y_feet_start = y + h - feet_height
        y_feet_end = y + h
        
        H, W = self.court_mask.shape[:2]
        x = max(0, min(x, W - 1))
        y_feet_start = max(0, min(y_feet_start, H - 1))
        y_feet_end = max(0, min(y_feet_end, H - 1))
        x_end = max(0, min(x + w, W))
        
        # Extraer región de los pies en la máscara
        feet_region = self.court_mask[y_feet_start:y_feet_end, x:x_end]
        
        # Contar píxeles parquet en la región de los pies
        if feet_region.size == 0:
            return False
        
        court_pixels = np.count_nonzero(feet_region)
        
        threshold = feet_region.size * 0.3
        return court_pixels >= threshold
    
    def filter_detections(self, detections):

        if self.court_mask is None:
            return detections  # Si no hay máscara, retornar todas
        
        filtered = []
        for detection in detections:
            if isinstance(detection, tuple) and len(detection) == 2:
                bbox, confidence = detection
            else:
                bbox = detection
                confidence = None
            
            # Verificar si toca la pista
            if self.touches_court(bbox):
                filtered.append(detection)
        
        return filtered

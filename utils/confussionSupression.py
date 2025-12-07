import cv2
import numpy as np
from mask.courtMask import CourtMaskGenerator


class CrowdSuppressor:
    """
    Filtra detecciones de público utilizando la máscara de la pista.
    Concepto: "Si los pies no tocan el parquet, no es un jugador"
    """
    
    def __init__(self):
        self.court_mask_generator = CourtMaskGenerator()
        self.court_mask = None
    
    def initialize_mask(self, frame):
        """Genera/actualiza la máscara de la pista en cada frame"""
        if self.court_mask is None:
            print("Generando máscara de la pista para filtrado de público...")
        
        # Actualizar máscara en cada frame para adaptarse al movimiento de cámara
        self.court_mask, _ = self.court_mask_generator.court_mask(frame, show_steps=False)
    
    def touches_court(self, bbox):
        """
        Verifica si los pies (parte inferior del bbox) tocan el parquet.
        
        Args:
            bbox: Tupla (x, y, w, h) con las coordenadas del bounding box
        
        Returns:
            bool: True si los pies tocan el parquet, False si es público
        """
        if self.court_mask is None:
            return True  # Si no hay máscara, aceptar todas las detecciones
        
        x, y, w, h = bbox
        
        # Extraer la región de los PIES (último 5% de altura del bbox)
        feet_height = max(int(h * 0.05), 5)  # Al menos 5 píxeles
        y_feet_start = y + h - feet_height
        y_feet_end = y + h
        
        # Asegurar que está dentro de los límites de la imagen
        H, W = self.court_mask.shape[:2]
        x = max(0, min(x, W - 1))
        y_feet_start = max(0, min(y_feet_start, H - 1))
        y_feet_end = max(0, min(y_feet_end, H - 1))
        x_end = max(0, min(x + w, W))
        
        # Extraer región de los pies en la máscara
        feet_region = self.court_mask[y_feet_start:y_feet_end, x:x_end]
        
        # Contar píxeles blancos (parquet) en la región de los pies
        if feet_region.size == 0:
            return False
        
        court_pixels = np.count_nonzero(feet_region)
        
        # Si al menos 25% de los píxeles de los pies tocan el parquet, es un jugador
        threshold = feet_region.size * 0.3
        return court_pixels >= threshold
    
    def filter_detections(self, detections):
        """
        Filtra una lista de detecciones eliminando las que no tocan la pista.
        
        Args:
            detections: Lista de tuplas (bbox, confidence) o solo bboxes
        
        Returns:
            Lista filtrada de detecciones que tocan la pista
        """
        if self.court_mask is None:
            return detections  # Si no hay máscara, retornar todas
        
        filtered = []
        for detection in detections:
            # Detectar si es tupla (bbox, confidence) o solo bbox
            if isinstance(detection, tuple) and len(detection) == 2:
                bbox, confidence = detection
            else:
                bbox = detection
                confidence = None
            
            # Verificar si toca la pista
            if self.touches_court(bbox):
                filtered.append(detection)
        
        return filtered

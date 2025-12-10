import cv2
import numpy as np
from pathlib import Path


class CourtMaskGenerator:
    
    def __init__(self):

        # Rango HSV parquet
        self.lower_parquet = np.array([0, 100, 100])
        self.upper_parquet = np.array([25, 255, 255])

        # Rango YCrCb piel humana
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Rango HSV amarillo/dorado
        self.lower_yellow = np.array([22, 80, 120], dtype=np.uint8)
        self.upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    
    def court_mask(self, image_bgr):

        # Segmentar parquet
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_court = cv2.inRange(hsv, self.lower_parquet, self.upper_parquet)

        # Segmentar piel
        ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        mask_skin_ycbcr = cv2.inRange(ycbcr, self.lower_skin, self.upper_skin)
        
        # Detectar amarillos/dorados y excluirlos de la máscara de piel
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_not_yellow = cv2.bitwise_not(mask_yellow)
        
        # Combinar: piel YCrCb AND NOT amarillo
        mask_skin = cv2.bitwise_and(mask_skin_ycbcr, mask_not_yellow)

        # Restar piel del parquet
        mask_not_skin = cv2.bitwise_not(mask_skin)
        mask_court_clean = cv2.bitwise_and(mask_court, mask_not_skin)
        
        # Aplicar ROI dinámico
        camera_side = self.detect_camera_side(image_bgr)
        mask_court_clean, _ = self.apply_roi(mask_court_clean, image_bgr.shape, camera_side)
        
        return mask_court_clean, hsv
    
    def detect_camera_side(self, image_bgr):

        h, w = image_bgr.shape[:2]
        corner_h = int(h * 0.60)
        corner_w = int(w * 0.30)
        
        mask_triangle_left = np.zeros((h, w), dtype=np.uint8)
        mask_triangle_right = np.zeros((h, w), dtype=np.uint8)
        
        # Triángulo izquierda
        triangle_left = np.array([
            [0, 0],
            [0, corner_h],
            [corner_w, 0]
        ], dtype=np.int32)
        cv2.fillPoly(mask_triangle_left, [triangle_left], 255)
        
        # Triángulo derecha
        triangle_right = np.array([
            [w, 0],
            [w - corner_w, 0],
            [w, corner_h]
        ], dtype=np.int32)
        cv2.fillPoly(mask_triangle_right, [triangle_right], 255)
        
        # Detectar parquet en cada triángulo
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_court = cv2.inRange(hsv, self.lower_parquet, self.upper_parquet)
        
        court_left = cv2.bitwise_and(mask_court, mask_court, mask=mask_triangle_left)
        court_right = cv2.bitwise_and(mask_court, mask_court, mask=mask_triangle_right)
        
        pixels_left = np.count_nonzero(court_left)
        pixels_right = np.count_nonzero(court_right)
        
        # Logica para determinar lado
        if pixels_left < pixels_right * 0.5:
            return 'left'
        elif pixels_right < pixels_left * 0.5:
            return 'right'
        else:
            return 'center'
    
    def apply_roi(self, mask, image_shape, camera_side='center'):

        h, w = image_shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Definir ROI según el lado de la cámara
        if camera_side == 'left':

            margin_x = int(w * 0.25)
            margin_y_top = int(h * 0.17)
            margin_y_bottom = int(h * 0.35)

            roi_polygon = np.array([
                [margin_x, margin_y_top],
                [w, int(margin_y_top*1.4)],
                [w, h],
                [0, h],
                [0, h - margin_y_bottom]
            ], dtype=np.int32)
            
        elif camera_side == 'right':

            margin_x_right = int(w * 0.75)
            margin_y_top = int(h * 0.17)
            margin_y_bottom = int(h * 0.35)

            roi_polygon = np.array([
                [0, int(margin_y_top*1.4)],
                [margin_x_right, margin_y_top],
                [w, h - margin_y_bottom],
                [w, h],
                [0, h]
            ], dtype=np.int32)
            
        else:
            roi_polygon = np.array([
                [0, int(h * 0.15)],
                [w, int(h * 0.15)],
                [w, h],
                [0, h]
            ], dtype=np.int32)
        
        cv2.fillPoly(roi_mask, [roi_polygon], 255)
        mask_roi = cv2.bitwise_and(mask, mask, mask=roi_mask)
        
        return mask_roi, roi_mask
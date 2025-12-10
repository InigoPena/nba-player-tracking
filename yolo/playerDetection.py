from pathlib import Path
from typing import List, Tuple
from utils.confussionSupression import CrowdSuppressor

import cv2
import numpy as np

PERSON_CLASS_ID = 0

class YOLOPlayerDetector:

    def __init__(self, config_path: str, weights_path: str):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path) # Red neuronal YOLO con configuración y pesos
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.crowd_suppressor = CrowdSuppressor()
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect_players(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        # Redimensionar y normalizar la imagen de entrada para YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Ejecutar la detección
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Procesar las salidas para extraer las detecciones
        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                
                if class_id != PERSON_CLASS_ID or confidence < 0.5:
                    continue

                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                x = max(int(center_x - width / 2), 0)
                y = max(int(center_y - height / 2), 0)
                
                # Filtrar bounding boxes demasiado grandes
                box_area = width * height
                frame_area = W * H
                if box_area > 0.30 * frame_area:
                    continue
                if box_area < 0.005 * frame_area:
                    continue
                if confidence < 0.2:
                    continue
                
                # Filtrar boxes con proporciones anormales
                aspect_ratio = width / (height + 1e-6)
                if aspect_ratio > 4 or aspect_ratio < 0.2:
                    continue
                if not self.crowd_suppressor.touches_court((x, y, width, height)):
                    continue
                
                boxes.append([x, y, width, height])
                confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        for idx in (indices.flatten() if len(indices) else []):
            bbox = tuple(boxes[idx])
            detections.append((bbox, confidences[idx]))
        
        return detections


def draw_players(frame: np.ndarray, tracked_players: List[Tuple[int, Tuple[int, int, int, int], float]]) -> np.ndarray:
    for player_id, bbox, confidence in tracked_players:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"ID {player_id} ({confidence:.2f})"
        cv2.putText(frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return frame
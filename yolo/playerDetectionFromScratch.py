from pathlib import Path
from typing import List, Tuple
from utils.confussionSupression import CrowdSuppressor

import cv2
import numpy as np

# En YOLO, la clase 'person' tiene el ID 0
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

        # Procesar las salidas para extraer las detecciones de personas
        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                
                if class_id != PERSON_CLASS_ID or confidence < 0.5:
                    continue

                # De coordenadas a píxeles
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                x = max(int(center_x - width / 2), 0)
                y = max(int(center_y - height / 2), 0)
                
                # Filtrar bounding boxes demasiado grandes
                box_area = width * height
                frame_area = W * H
                if box_area > 0.30 * frame_area:  # Más del 30% del frame
                    continue
                if box_area < 0.005 * frame_area:  # Menor a 0.5% del frame
                    continue
                if confidence < 0.2:  # Confianza mínima elevada
                    continue
                
                # Filtrar boxes con proporciones anormales
                aspect_ratio = width / (height + 1e-6)
                if aspect_ratio > 4 or aspect_ratio < 0.2:  # Muy anchos o muy altos
                    continue
                if not self.crowd_suppressor.touches_court((x, y, width, height)):
                    continue  # Filtrar detecciones de público
                
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


def main():
    VIDEO_INPUT = "videos/JazzAtacaCorto.mp4"
    VIDEO_OUTPUT = "videos/JazzAtacaCorto_detected.mp4"
    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"
    
    print(f"Procesando: {VIDEO_INPUT}")
    
    detector = YOLOPlayerDetector(CONFIG, WEIGHTS)
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {VIDEO_INPUT}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    Path(VIDEO_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("Error: No se pudo crear el video de salida")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_players(frame)
        
        # Dibujar solo detecciones sin tracking
        annotated = frame.copy()
        for bbox, confidence in detections:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = f"{confidence:.2f}"
            cv2.putText(annotated, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        
        writer.write(annotated)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Procesados {frame_count}/{total_frames} frames...")
    
    cap.release()
    writer.release()
    
    print(f"\n✓ Video guardado en: {VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from yolo.playerDetectionFromScratch import YOLOPlayerDetector
from teamClassification import classify_team_by_uniform


class PlayerDetector:
    def __init__(self, detector: YOLOPlayerDetector, detection_interval: int = 5):
        self.detector = detector
        self.detection_interval = detection_interval
        self.previous_players = []  # Guardar detecciones anteriores
        self.trackers = []  # Lista de trackers para cada jugador

    def _calculate_iou(self, boxA, boxB):
        """Calcula la intersección sobre unión entre dos cajas"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def detect_and_classify(self, frame, frame_count):
        """
        Detecta jugadores en el frame y los clasifica por equipo según color.
        Usa YOLO cada 'detection_interval' frames y tracking simple en los intermedios.
        Retorna una lista de tuplas: [(bbox, equipo), ...]
        donde equipo es 'negro' o 'blanco'
        """
        # Ejecutar YOLO cada N frames
        if frame_count % self.detection_interval == 0:
            # Detectar jugadores con YOLO
            detections = self.detector.detect_players(frame)
            
            # Clasificar cada detección por color usando el método robusto
            classified_players = []
            self.trackers = []  # Reiniciar trackers
            
            for bbox, confidence in detections:
                team = classify_team_by_uniform(frame, bbox)
                classified_players.append((bbox, team))
                
                # Crear tracker para este jugador
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                self.trackers.append((tracker, team))
            
            self.previous_players = classified_players
            return classified_players
        else:
            # Usar tracking en frames intermedios
            tracked_players = []
            valid_trackers = []
            
            for tracker, team in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    bbox = tuple(map(int, bbox))
                    tracked_players.append((bbox, team))
                    valid_trackers.append((tracker, team))
            
            self.trackers = valid_trackers
            return tracked_players

def main():
    VIDEO_INPUT = "videos/JazzAtacaCorto.mp4"
    VIDEO_OUTPUT = "videos/JazzAtacaCorto_teams_try.mp4"
    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"

    print(f"Iniciando detección de jugadores por equipos...")
    
    # Inicializar detector con intervalo de detección cada 5 frames
    yolo_detector = YOLOPlayerDetector(CONFIG, WEIGHTS)
    detector = PlayerDetector(yolo_detector, detection_interval=5)

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error al abrir {VIDEO_INPUT}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar y clasificar jugadores por equipo (YOLO cada 5 frames, tracking en intermedios)
        players = detector.detect_and_classify(frame, frame_count)
        
        # Dibujar resultados
        annotated = frame.copy()
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Indicador de modo
        if frame_count % detector.detection_interval == 0:
            cv2.putText(annotated, "MODO: YOLO DETECTION", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "MODO: TRACKING", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Contadores de equipos
        equipo_negro = 0
        equipo_blanco = 0
        
        for bbox, team in players:
            x, y, w, h = bbox
            
            # Color de caja según el equipo
            if team == 'negro':
                color = (0, 0, 255)  # Rojo para equipo negro
                label = "Suns"
                equipo_negro += 1
            else:
                color = (255, 0, 0)  # Azul para equipo blanco
                label = "Jazz"
                equipo_blanco += 1
            
            # Dibujar caja y etiqueta
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, max(0, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Mostrar contadores
        cv2.putText(annotated, f"Equipo Negro: {equipo_negro}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"Equipo Blanco: {equipo_blanco}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        writer.write(annotated)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Procesado frame {frame_count}")

    cap.release()
    writer.release()
    print("Proceso finalizado.")

if __name__ == "__main__":
    main()
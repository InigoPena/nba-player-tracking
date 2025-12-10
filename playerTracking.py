import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from yolo.playerDetection import YOLOPlayerDetector
from utils.teamClassification import classify_team_by_uniform
from utils.confussionSupression import CrowdSuppressor

class PlayerTrack:
    def __init__(self, player_id, bbox, team, frame):
        self.id = player_id
        self.bbox = bbox
        self.team = team
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        
        # Contadores de vida
        self.frames_since_update = 0  # Cuántos frames lleva sin ser visto por YOLO
        self.hits = 0                 # Cuántas veces ha sido confirmado

    def predict(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = tuple(map(int, bbox))
        return success, self.bbox

    def update(self, frame, new_bbox):
        self.frames_since_update = 0
        self.hits += 1
        self.bbox = new_bbox
        
        self.team = classify_team_by_uniform(frame, new_bbox)
        
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, new_bbox)

    # Prevenir oclusión
    def mark_missed(self):
        self.frames_since_update += 1

class PlayerDetector:
    def __init__(self, detector: YOLOPlayerDetector, detection_interval: int = 5):
        self.detector = detector
        self.detection_interval = detection_interval
        
        # Diccionario de tracks
        self.tracks = {}
        self.next_id = 1
        
        self.max_disappeared = 10   # Frames que aguanta un jugador desaparecido antes de borrarlo
        self.iou_threshold = 0.3    # Umbral para saber si es el mismo jugador
        
        # Tamaño mínimo
        self.min_box_area = 0
        self.min_box_width = 30
        self.min_box_height = 50
        
        # Filtrado de público
        self.crowd_suppressor = CrowdSuppressor()

    def _calculate_iou(self, boxA, boxB):

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

        if self.min_box_area == 0:
            H, W = frame.shape[:2]
            self.min_box_area = (W * H) * 0.003  # 0.3% del área del frame
        
        # Inicializar la máscara de la pista en el primer frame
        self.crowd_suppressor.initialize_mask(frame)
        
        # 1. PREDICCIÓN: Actualizar trackers existentes (memoria visual)
        # Esto ocurre en TODOS los frames
        active_ids = list(self.tracks.keys())
        for pid in active_ids:
            success, _ = self.tracks[pid].predict(frame)
            if not success:
                # Si CSRT pierde el rastro visualmente, lo borramos
                del self.tracks[pid]

        # Detección cada n frames
        if frame_count % self.detection_interval == 0:
            # YOLO
            raw_detections = self.detector.detect_players(frame)
            detected_bboxes = [d[0] for d in raw_detections]

            # Asociar detecciones con trackers
            track_ids = list(self.tracks.keys())
            
            used_detections = set()
            used_tracks = set()

            if len(track_ids) > 0 and len(detected_bboxes) > 0:

                iou_matrix = np.zeros((len(track_ids), len(detected_bboxes)))
                for i, tid in enumerate(track_ids):
                    for j, det_box in enumerate(detected_bboxes):
                        iou_matrix[i, j] = self._calculate_iou(self.tracks[tid].bbox, det_box)

                # Algoritmo Greedy para emparejar
                while True:
                    if iou_matrix.size == 0: break

                    ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                    max_iou = iou_matrix[ind]
                    
                    if max_iou < self.iou_threshold: break
                    
                    track_idx, det_idx = ind
                    tid = track_ids[track_idx]
                    
                    # Actualizar si hay buena coincidencia
                    self.tracks[tid].update(frame, detected_bboxes[det_idx])
                    
                    used_tracks.add(tid)
                    used_detections.add(det_idx)
                    
                    iou_matrix[track_idx, :] = -1
                    iou_matrix[:, det_idx] = -1
            
            # Oclusión
            for tid in track_ids:

                if tid not in used_tracks:
                    self.tracks[tid].mark_missed()

                    if self.tracks[tid].frames_since_update > self.max_disappeared:
                        del self.tracks[tid]

            for i, bbox in enumerate(detected_bboxes):

                if i not in used_detections:

                    x, y, w, h = bbox
                    box_area = w * h

                    # Filtrar detecciones demasiado pequeñas
                    if box_area < self.min_box_area or w < self.min_box_width or h < self.min_box_height:
                        continue
                    
                    # Ignorar público
                    if not self.crowd_suppressor.touches_court(bbox):
                        continue
                    
                    team = classify_team_by_uniform(frame, bbox)
                    new_track = PlayerTrack(self.next_id, bbox, team, frame)
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1

        results = []
        
        for pid, track in self.tracks.items():

            if track.frames_since_update < 5:
                x, y, w, h = track.bbox
                box_area = w * h

                if box_area >= self.min_box_area and w >= self.min_box_width and h >= self.min_box_height:
                    results.append((track.bbox, track.team, pid))
        
        return results


def main():

    VIDEO_INPUT = "media/videos/atacaSuns.mp4"
    VIDEO_OUTPUT = "media/videos/outputs/atacaSuns_tracking.mp4"

    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"

    print(f"Iniciando detección de jugadores por equipos con OCLUSIÓN...")
    
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
        
        # Detectar y clasificar
        players = detector.detect_and_classify(frame, frame_count)
        
        annotated = frame.copy()
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        equipo_negro = 0
        equipo_blanco = 0
        
        for bbox, team, player_id in players:
            x, y, w, h = bbox
            
            if team == 'negro':
                color = (0, 0, 255) 
                label = f"Phx. Suns Player"
                equipo_negro += 1
            else:
                color = (255, 0, 0) 
                label = f"Uth. Jazz Player"
                equipo_blanco += 1
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, max(0, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Mostrar contadores
        cv2.putText(annotated, f"Suns (Negro): {equipo_negro}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"Jazz (Blanco): {equipo_blanco}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        writer.write(annotated)
        frame_count += 1
        
        if frame_count % 25 == 0:
            print(f"Procesado frame {frame_count}")

    cap.release()
    writer.release()
    
    print(f"\nProceso finalizado. Total de frames procesados: {frame_count}")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from yolo.playerDetectionFromScratch import YOLOPlayerDetector
from utils.teamClassification import classify_team_by_uniform
from utils.confussionSupression import CrowdSuppressor

# ---------------------------------------------------------
# CLASE AUXILIAR: Mantiene el estado de UN jugador individual
# ---------------------------------------------------------
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
        """Actualiza el tracker interno (CSRT) en frames intermedios"""
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = tuple(map(int, bbox))
        return success, self.bbox

    def update(self, frame, new_bbox):
        """Corrige el tracker con una nueva detección confirmada de YOLO"""
        self.frames_since_update = 0
        self.hits += 1
        self.bbox = new_bbox
        
        # RECLASIFICAR el equipo para evitar que se "pegue" la etiqueta incorrecta
        self.team = classify_team_by_uniform(frame, new_bbox)
        
        # Re-iniciamos el tracker CSRT con la caja perfecta de YOLO para corregir deriva
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, new_bbox)

    def mark_missed(self):
        """Marca que este jugador no fue visto en este frame (posible oclusión)"""
        self.frames_since_update += 1


# ---------------------------------------------------------
# DETECTOR PRINCIPAL (Lógica Integrada)
# ---------------------------------------------------------
class PlayerDetector:
    def __init__(self, detector: YOLOPlayerDetector, detection_interval: int = 5):
        self.detector = detector
        self.detection_interval = detection_interval
        
        # DICCIONARIO de tracks activos: { id: PlayerTrack }
        self.tracks = {}
        self.next_id = 1
        
        # Configuración de Oclusión
        self.max_disappeared = 10   # Frames que aguanta un jugador desaparecido antes de borrarlo
        self.iou_threshold = 0.3    # Umbral para saber si es el mismo jugador
        
        # Configuración de Tamaño Mínimo
        self.min_box_area = 0       # Se inicializa con el primer frame
        self.min_box_width = 30     # Píxeles mínimos de ancho
        self.min_box_height = 50    # Píxeles mínimos de alto
        
        # Filtrado de público
        self.crowd_suppressor = CrowdSuppressor()

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
        Gestiona el ciclo de vida de los jugadores: Detectar -> Asociar -> Actualizar.
        Retorna: [(bbox, equipo, id), ...]
        """
        # Inicializar el área mínima basado en el tamaño del frame (0.3% del frame)
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
                # Si CSRT pierde el rastro visualmente (se sale de pantalla), lo borramos
                del self.tracks[pid]

        # 2. DETECCIÓN Y CORRECCIÓN (Solo cada N frames)
        if frame_count % self.detection_interval == 0:
            # A) Ejecutar YOLO
            raw_detections = self.detector.detect_players(frame)
            # Extraemos solo las cajas de la respuesta de YOLO
            detected_bboxes = [d[0] for d in raw_detections]

            # B) Asociación de Datos (Emparejar Tracks antiguos con Detecciones nuevas)
            track_ids = list(self.tracks.keys())
            
            used_detections = set()
            used_tracks = set()

            if len(track_ids) > 0 and len(detected_bboxes) > 0:
                # Calcular matriz de IoU
                iou_matrix = np.zeros((len(track_ids), len(detected_bboxes)))
                for i, tid in enumerate(track_ids):
                    for j, det_box in enumerate(detected_bboxes):
                        iou_matrix[i, j] = self._calculate_iou(self.tracks[tid].bbox, det_box)

                # Algoritmo Greedy para emparejar
                while True:
                    if iou_matrix.size == 0: break
                    # Buscar el mayor IoU
                    ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                    max_iou = iou_matrix[ind]
                    
                    if max_iou < self.iou_threshold: break # No quedan coincidencias buenas
                    
                    track_idx, det_idx = ind
                    tid = track_ids[track_idx]
                    
                    # ¡COINCIDENCIA! Actualizamos el tracker con la caja real de YOLO
                    self.tracks[tid].update(frame, detected_bboxes[det_idx])
                    
                    used_tracks.add(tid)
                    used_detections.add(det_idx)
                    
                    # Anular fila/columna usadas
                    iou_matrix[track_idx, :] = -1
                    iou_matrix[:, det_idx] = -1

            # C) Gestión de no emparejados
            
            # 1. Tracks que YOLO no vio (OCLUSIÓN)
            for tid in track_ids:
                if tid not in used_tracks:
                    self.tracks[tid].mark_missed()
                    # Si lleva perdido demasiados frames, lo borramos
                    if self.tracks[tid].frames_since_update > self.max_disappeared:
                        del self.tracks[tid]

            # 2. Detecciones nuevas (JUGADORES NUEVOS)
            for i, bbox in enumerate(detected_bboxes):
                if i not in used_detections:
                    # Filtrar detecciones demasiado pequeñas
                    x, y, w, h = bbox
                    box_area = w * h
                    if box_area < self.min_box_area or w < self.min_box_width or h < self.min_box_height:
                        continue  # Ignorar detección muy pequeña
                    
                    # FILTRO DE PÚBLICO: Si los pies no tocan el parquet, es público
                    if not self.crowd_suppressor.touches_court(bbox):
                        continue  # Ignorar detección de público
                    
                    # Clasificar equipo solo al nacer
                    team = classify_team_by_uniform(frame, bbox)
                    new_track = PlayerTrack(self.next_id, bbox, team, frame)
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1

        # 3. PREPARAR SALIDA
        results = []
        
        for pid, track in self.tracks.items():

            # Solo devolvemos el jugador si no está "muy perdido" y cumple tamaño mínimo
            if track.frames_since_update < 5:
                x, y, w, h = track.bbox
                box_area = w * h
                # Filtrar cajas muy pequeñas en la salida final
                if box_area >= self.min_box_area and w >= self.min_box_width and h >= self.min_box_height:
                    results.append((track.bbox, track.team, pid))
        
        return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    VIDEO_INPUT = "videos/VideoConflictivo.mp4"
    VIDEO_OUTPUT = "videos/VideoConflictivo_Muestraaa2.mp4"
    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"

    print(f"Iniciando detección de jugadores por equipos con OCLUSIÓN...")
    
    # Inicializar detector
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
        
        # Detectar y clasificar (Ahora retorna también el ID)
        players = detector.detect_and_classify(frame, frame_count)
        
        annotated = frame.copy()
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Contadores
        equipo_negro = 0
        equipo_blanco = 0
        
        # Desempaquetamos 3 valores ahora: bbox, team, id
        for bbox, team, player_id in players:
            x, y, w, h = bbox
            
            if team == 'negro':
                color = (0, 0, 255) 
                label = f"#{player_id} Suns"
                equipo_negro += 1
            else:
                color = (255, 0, 0) 
                label = f"#{player_id} Jazz"
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
        
        if frame_count % 50 == 0:
            print(f"Procesado frame {frame_count}")

    cap.release()
    writer.release()
    
    print(f"\nProceso finalizado. Total de frames procesados: {frame_count}")

if __name__ == "__main__":
    main()
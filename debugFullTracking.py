import cv2
import numpy as np
from yolo.playerDetectionFromScratch import YOLOPlayerDetector
from confussionSupression import CrowdSuppressor
from playerTracking import PlayerDetector, PlayerTrack


def debug_full_tracking_system(video_path, target_second=7):
    """
    Analiza el sistema completo de tracking (YOLO + CSRT) en un segundo específico
    """
    print(f"Analizando sistema completo en segundo {target_second} de {video_path}")
    
    # Inicializar sistema completo
    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"
    yolo_detector = YOLOPlayerDetector(CONFIG, WEIGHTS)
    detector = PlayerDetector(yolo_detector, detection_interval=5)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(target_second * fps)
    
    print(f"FPS: {fps}, Frame objetivo: {target_frame}")
    print(f"Procesando desde el inicio hasta el frame {target_frame}...\n")
    
    # Procesar todos los frames hasta el objetivo
    frame_count = 0
    target_frame_data = None
    
    while frame_count <= target_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar con el sistema de tracking
        players = detector.detect_and_classify(frame, frame_count)
        
        # Si es el frame objetivo, guardamos la info
        if frame_count == target_frame:
            target_frame_data = (frame.copy(), players)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Procesado frame {frame_count}...")
    
    if target_frame_data is None:
        print("No se pudo alcanzar el frame objetivo")
        return
    
    frame, players = target_frame_data
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS COMPLETO EN FRAME {target_frame} (segundo {target_second})")
    print(f"{'='*80}")
    print(f"Total de detecciones finales: {len(players)}")
    print(f"Tracks activos: {len(detector.tracks)}")
    print(f"Es frame de detección YOLO: {target_frame % detector.detection_interval == 0}")
    print()
    
    # Analizar cada detección final
    frame_debug = frame.copy()
    H, W = frame.shape[:2]
    
    for i, (bbox, team, player_id) in enumerate(players):
        x, y, w, h = bbox
        
        # Verificar si el track existe y cuántos frames lleva sin actualizar
        track = detector.tracks.get(player_id)
        if track:
            frames_since_update = track.frames_since_update
            hits = track.hits
            source = "YOLO" if frames_since_update == 0 else f"CSRT ({frames_since_update}f)"
        else:
            frames_since_update = -1
            hits = 0
            source = "DESCONOCIDO"
        
        # Extraer región de los pies
        feet_height = max(int(h * 0.2), 5)
        y_feet_start = y + h - feet_height
        y_feet_end = y + h
        
        # Asegurar límites
        x_safe = max(0, min(x, W - 1))
        y_feet_start_safe = max(0, min(y_feet_start, H - 1))
        y_feet_end_safe = max(0, min(y_feet_end, H - 1))
        x_end_safe = max(0, min(x + w, W))
        
        # Extraer región de pies en la máscara
        feet_region = detector.crowd_suppressor.court_mask[y_feet_start_safe:y_feet_end_safe, x_safe:x_end_safe]
        
        if feet_region.size == 0:
            court_pixels = 0
            total_pixels = 0
            percentage = 0
        else:
            court_pixels = np.count_nonzero(feet_region)
            total_pixels = feet_region.size
            percentage = (court_pixels / total_pixels) * 100
        
        # Verificar si pasaría el filtro AHORA
        threshold = total_pixels * 0.05
        would_pass_filter = court_pixels >= threshold
        
        # Color según fuente: Amarillo=YOLO reciente, Cyan=CSRT, Rojo=No toca pista
        if source == "YOLO":
            color = (0, 255, 255)  # Amarillo (YOLO reciente)
        elif "CSRT" in source:
            if would_pass_filter:
                color = (255, 255, 0)  # Cyan (CSRT válido)
            else:
                color = (0, 0, 255)  # Rojo (CSRT NO toca pista)
        else:
            color = (128, 128, 128)  # Gris
        
        # Dibujar bbox
        cv2.rectangle(frame_debug, (x, y), (x + w, y + h), color, 2)
        
        # Dibujar región de pies
        cv2.rectangle(frame_debug, (x, y_feet_start), (x + w, y_feet_end), color, 1)
        
        # Etiqueta detallada
        label = f"ID{player_id} {source}"
        cv2.putText(frame_debug, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Porcentaje de pista
        pct_label = f"{percentage:.0f}%"
        cv2.putText(frame_debug, pct_label, (x, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Diagnóstico detallado
        print(f"ID {player_id} ({team}): {source}")
        print(f"  BBox: x={x}, y={y}, w={w}, h={h}")
        print(f"  Frames sin actualizar YOLO: {frames_since_update}")
        print(f"  Confirmaciones totales: {hits}")
        print(f"  Píxeles tocando parquet: {court_pixels}/{total_pixels} ({percentage:.1f}%)")
        print(f"  ¿Pasaría filtro ahora?: {'✓ SÍ' if would_pass_filter else '✗ NO'}")
        if not would_pass_filter and "CSRT" in source:
            print(f"  ⚠️ PROBLEMA: Tracker CSRT sin tocar pista (se creó en frame anterior)")
        print()
    
    # Crear visualización
    mask_colored = cv2.cvtColor(detector.crowd_suppressor.court_mask, cv2.COLOR_GRAY2BGR)
    
    h_frame, w_frame = frame.shape[:2]
    combined = np.zeros((h_frame, w_frame * 3, 3), dtype=np.uint8)
    combined[:, 0:w_frame] = frame
    combined[:, w_frame:w_frame*2] = frame_debug
    combined[:, w_frame*2:w_frame*3] = mask_colored
    
    # Títulos y leyenda
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Tracking (Amarillo=YOLO, Cyan=CSRT OK, Rojo=CSRT sin pista)", 
               (w_frame + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, "Mascara de Pista", (w_frame*2 + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Guardar
    output_path = f"debug_tracking_frame_{target_frame}_seg{target_second}.jpg"
    cv2.imwrite(output_path, combined)
    
    print(f"{'='*80}")
    print(f"RESUMEN:")
    print(f"  - Detecciones YOLO originales en este ciclo: {len(detector.tracks)} tracks totales")
    print(f"  - Detecciones finales mostradas: {len(players)}")
    print(f"  - Imagen guardada: {output_path}")
    print(f"{'='*80}")
    
    cv2.imshow("Debug Tracking Completo - Presiona cualquier tecla", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()


if __name__ == "__main__":
    VIDEO_PATH = "videos/VideoConflictivo.mp4"
    TARGET_SECOND = 1
    
    debug_full_tracking_system(VIDEO_PATH, TARGET_SECOND)

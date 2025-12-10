"""
Script simple para medir métricas de rendimiento del sistema de tracking
"""
import cv2
import time
from pathlib import Path
from playerTracking import PlayerDetector
from yolo.playerDetection import YOLOPlayerDetector


def main():
    # Configuración
    VIDEO_INPUT = "media/videos/video1.mp4"
    WEIGHTS = "yolo/yolov3.weights"
    CONFIG = "yolo/yolov3.cfg"
    DETECTION_INTERVAL = 5
    
    print("=" * 70)
    print("MÉTRICAS DE RENDIMIENTO - SISTEMA DE TRACKING DE JUGADORES NBA")
    print("=" * 70)
    print(f"\nVideo: {VIDEO_INPUT}")
    print(f"Intervalo de detección: Cada {DETECTION_INTERVAL} frames")
    
    # Inicializar detector
    yolo_detector = YOLOPlayerDetector(CONFIG, WEIGHTS)
    detector = PlayerDetector(yolo_detector, detection_interval=DETECTION_INTERVAL)
    
    # Abrir video
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir {VIDEO_INPUT}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS del video: {fps:.2f}")
    print(f"Total de frames: {total_frames}")
    print("\nProcesando...\n")
    
    # Variables para métricas
    frame_count = 0
    tiempo_inicio = time.time()
    
    # Métricas de detección
    tiempos_por_frame = []
    tiempos_con_yolo = []
    tiempos_sin_yolo = []
    
    # Métricas de tracking
    jugadores_por_frame = []
    ids_unicos = set()
    duracion_tracks = {}  # {id: número de frames que apareció}
    
    # Métricas de clasificación
    contador_equipos = {"negro": 0, "blanco": 0}
    
    # Procesar video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t_inicio_frame = time.time()
        players = detector.detect_and_classify(frame, frame_count)
        t_fin_frame = time.time()
        
        tiempo_frame = (t_fin_frame - t_inicio_frame) * 1000
        tiempos_por_frame.append(tiempo_frame)

        if frame_count % DETECTION_INTERVAL == 0:
            tiempos_con_yolo.append(tiempo_frame)
        else:
            tiempos_sin_yolo.append(tiempo_frame)
        
        jugadores_por_frame.append(len(players))
        
        for bbox, team, player_id in players:
            ids_unicos.add(player_id)
            contador_equipos[team] += 1
            
            # Contar duración de cada track
            if player_id not in duracion_tracks:
                duracion_tracks[player_id] = 0
            duracion_tracks[player_id] += 1
        
        frame_count += 1
        
        if frame_count % 25 == 0:
            print(f"Procesados {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    tiempo_total = time.time() - tiempo_inicio
    fps_procesamiento = frame_count / tiempo_total
    
    tiempo_promedio = sum(tiempos_por_frame) / len(tiempos_por_frame)
    tiempo_promedio_yolo = sum(tiempos_con_yolo) / len(tiempos_con_yolo) if tiempos_con_yolo else 0
    tiempo_promedio_tracking = sum(tiempos_sin_yolo) / len(tiempos_sin_yolo) if tiempos_sin_yolo else 0
    
    jugadores_promedio = sum(jugadores_por_frame) / len(jugadores_por_frame)
    jugadores_max = max(jugadores_por_frame)
    jugadores_min = min(jugadores_por_frame)
    
    duraciones = list(duracion_tracks.values())
    vida_promedio = sum(duraciones) / len(duraciones) if duraciones else 0
    vida_max = max(duraciones) if duraciones else 0
    
    total_detecciones = contador_equipos["negro"] + contador_equipos["blanco"]
    porcentaje_negro = (contador_equipos["negro"] / total_detecciones * 100) if total_detecciones > 0 else 0
    porcentaje_blanco = (contador_equipos["blanco"] / total_detecciones * 100) if total_detecciones > 0 else 0
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    
    print("\n📊 1. VELOCIDAD DE PROCESAMIENTO")
    print("-" * 70)
    print(f"   Tiempo total:              {tiempo_total:.2f} segundos")
    print(f"   FPS de procesamiento:      {fps_procesamiento:.2f} FPS")
    print(f"   Tiempo promedio/frame:     {tiempo_promedio:.2f} ms")
    print(f"   Tiempo con YOLO:           {tiempo_promedio_yolo:.2f} ms")
    print(f"   Tiempo solo tracking:      {tiempo_promedio_tracking:.2f} ms")
    print(f"   Speedup tracking vs YOLO:  {tiempo_promedio_yolo/tiempo_promedio_tracking:.2f}x" if tiempo_promedio_tracking > 0 else "")
    
    print("\n🎯 2. PRECISIÓN DEL TRACKING")
    print("-" * 70)
    print(f"   Jugadores únicos detectados:       {len(ids_unicos)}")
    print(f"   Jugadores promedio por frame:      {jugadores_promedio:.2f}")
    print(f"   Rango de jugadores:                {jugadores_min} - {jugadores_max}")
    print(f"   Vida promedio de un track:         {vida_promedio:.1f} frames ({vida_promedio/fps:.2f}s)")
    print(f"   Track más largo:                   {vida_max} frames ({vida_max/fps:.2f}s)")
    print(f"   Estabilidad (vida/total frames):   {(vida_promedio/frame_count)*100:.1f}%")
    
    print("\n👕 3. CLASIFICACIÓN DE EQUIPOS")
    print("-" * 70)
    print(f"   Total de detecciones:              {total_detecciones}")
    print(f"   Equipo Negro (Suns):               {contador_equipos['negro']} ({porcentaje_negro:.1f}%)")
    print(f"   Equipo Blanco (Jazz):              {contador_equipos['blanco']} ({porcentaje_blanco:.1f}%)")
    print(f"   Ratio Negro/Blanco:                {contador_equipos['negro']/contador_equipos['blanco']:.2f}" if contador_equipos['blanco'] > 0 else "")
    
    print("\n💡 4. EFICIENCIA DEL SISTEMA")
    print("-" * 70)
    print(f"   Frames con YOLO:               {len(tiempos_con_yolo)} ({len(tiempos_con_yolo)/frame_count*100:.1f}%)")
    print(f"   Frames solo tracking:          {len(tiempos_sin_yolo)} ({len(tiempos_sin_yolo)/frame_count*100:.1f}%)")
    print(f"   Ahorro computacional:          ~{100 - (len(tiempos_con_yolo)/frame_count*100):.1f}% menos YOLO")
    print(f"   Tiempo real vs procesamiento:  {fps/fps_procesamiento:.2f}x" + 
          (" (más rápido que tiempo real)" if fps_procesamiento > fps else " (más lento que tiempo real)"))
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

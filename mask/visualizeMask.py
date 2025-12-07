import cv2
import numpy as np
from courtMask import CourtMaskGenerator

# Video a procesar
VIDEO_INPUT = "videos/Suns_vs_Jazz.mp4"

# Inicializar generador de máscara
mask_gen = CourtMaskGenerator()

# Abrir video
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps:.0f} FPS")
print("Presiona 'q' para salir")

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar lado de cámara y generar máscara
    camera_side = mask_gen.detect_camera_side(frame)
    mask, _ = mask_gen.court_mask(frame, use_dynamic_roi=True)
    
    # Crear visualización
    h, w = frame.shape[:2]
    frame_roi = frame.copy()
    
    # Dibujar polígono ROI según el lado
    if camera_side == 'left':
        roi = np.array([
            [int(w*0.25), int(h*0.17)],
            [w, int(h*0.17*1.4)],
            [w, h],
            [0, h],
            [0, int(h*0.65)]
        ], dtype=np.int32)
        color = (0, 255, 255)  # Amarillo
        
    elif camera_side == 'right':
        roi = np.array([
            [0, int(h*0.17*1.4)],
            [int(w*0.75), int(h*0.17)],
            [w, int(h*0.65)],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        color = (255, 0, 255)  # Magenta
        
    else:  # center
        roi = np.array([
            [0, int(h*0.15)],
            [w, int(h*0.15)],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        color = (0, 255, 0)  # Verde
    
    # Dibujar ROI con overlay transparente
    cv2.polylines(frame_roi, [roi], True, color, 4)
    overlay = frame_roi.copy()
    cv2.fillPoly(overlay, [roi], color)
    frame_roi = cv2.addWeighted(frame_roi, 0.85, overlay, 0.15, 0)
    
    # Aplicar máscara al frame
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Agregar texto a cada ventana
    cv2.putText(frame_roi, f"Lado: {camera_side.upper()}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_roi, f"Frame: {frame_num}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(masked, f"Lado: {camera_side.upper()}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(masked, f"Frame: {frame_num}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar en dos ventanas separadas
    cv2.imshow("ROI", frame_roi)
    cv2.imshow("Mascara", masked)
    
    frame_num += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

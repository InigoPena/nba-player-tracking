import cv2
import numpy as np
from pathlib import Path
from courtMask import CourtMaskGenerator

script_dir = Path(__file__).parent
VIDEO_INPUT = str(script_dir.parent / "media" / "videos" / "Contraataque.mp4")
VIDEO_OUTPUT = str(script_dir.parent / "media" / "videos" / "outputs" / "contraataque_mask_visualization.mp4")

mask_gen = CourtMaskGenerator()

# Abrir video
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir el video {VIDEO_INPUT}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width * 2, height))

print(f"Video: {width}x{height} @ {fps:.0f} FPS")
print(f"Guardando en: {VIDEO_OUTPUT}")
print("Presiona 'q' para salir")

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    camera_side = mask_gen.detect_camera_side(frame)
    mask, _ = mask_gen.court_mask(frame)
    
    h, w = frame.shape[:2]
    frame_roi = frame.copy()
    
    if camera_side == 'left':
        roi = np.array([
            [int(w*0.25), int(h*0.17)],
            [w, int(h*0.17*1.4)],
            [w, h],
            [0, h],
            [0, int(h*0.65)]
        ], dtype=np.int32)
        color = (0, 255, 255)
        
    elif camera_side == 'right':
        roi = np.array([
            [0, int(h*0.17*1.4)],
            [int(w*0.75), int(h*0.17)],
            [w, int(h*0.65)],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        color = (255, 0, 255)
        
    else:
        roi = np.array([
            [0, int(h*0.15)],
            [w, int(h*0.15)],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        color = (0, 255, 0)
    
    cv2.polylines(frame_roi, [roi], True, color, 4)
    overlay = frame_roi.copy()
    cv2.fillPoly(overlay, [roi], color)
    frame_roi = cv2.addWeighted(frame_roi, 0.85, overlay, 0.15, 0)
    
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.putText(frame_roi, f"Lado: {camera_side.upper()}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_roi, f"Frame: {frame_num}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(masked, f"Lado: {camera_side.upper()}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(masked, f"Frame: {frame_num}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    combined = np.hstack([frame_roi, masked])
    
    out.write(combined)
    
    scale = 0.5
    frame_roi_small = cv2.resize(frame_roi, (int(w * scale), int(h * scale)))
    masked_small = cv2.resize(masked, (int(w * scale), int(h * scale)))
    
    cv2.imshow("ROI", frame_roi_small)
    cv2.imshow("Mascara", masked_small)
    
    frame_num += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Video guardado en: {VIDEO_OUTPUT}")

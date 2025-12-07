import cv2
import numpy as np
from pathlib import Path


class CourtMaskGenerator:
    
    def __init__(self):

        self.lower_parquet = np.array([0, 100, 100])
        self.upper_parquet = np.array([25, 255, 255])
    
    # Rango HSV manual
    def set_hsv_range(self, h_min, h_max, s_min, s_max, v_min, v_max):

        self.lower_parquet = np.array([h_min, s_min, v_min])
        self.upper_parquet = np.array([h_max, s_max, v_max])
    
    # Segmentar color del parquet
    def segment_court_color(self, image_bgr, show_steps=False):

        # Convertir BGR a HSV
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_court = cv2.inRange(hsv, self.lower_parquet, self.upper_parquet)
        
        if show_steps:

            cv2.imshow("4. Máscara Parquet (bruta)", mask_court)
            result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_court)
            cv2.imshow("5. Resultado (BGR con máscara)", result)
        
        return mask_court, hsv
    
    # Diferenciar piel de parquet
    def detect_skin_mask(self, image_bgr, show_steps=False):

        # Convertir BGR a YCbCr
        ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Aplicar umbralización
        mask_skin_ycbcr = cv2.inRange(ycbcr, lower_skin, upper_skin)
        
        # FILTRO ADICIONAL: Excluir amarillos/dorados usando HSV
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([18, 120, 230], dtype=np.uint8)
        upper_yellow = np.array([24, 150, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask_not_yellow = cv2.bitwise_not(mask_yellow)
        
        # Combinar: piel YCbCr AND NOT amarillo
        mask_skin = cv2.bitwise_and(mask_skin_ycbcr, mask_not_yellow)
        
        if show_steps:
            cv2.imshow("3g. Máscara Piel FINAL (sin amarillos)", mask_skin)
            skin_result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_skin)
            cv2.imshow("3h. Piel Detectada (sin amarillos)", skin_result)
        
        return mask_skin
    
    def subtract_skin_from_court(self, mask_court, mask_skin):

        mask_not_skin = cv2.bitwise_not(mask_skin)
        mask_court_clean = cv2.bitwise_and(mask_court, mask_not_skin)
        
        return mask_court_clean
    
    def court_mask(self, image_bgr, show_steps=False, use_dynamic_roi=True):
        """
        Segmenta el parquet y resta automáticamente la piel.
        Aplica ROI dinámico si use_dynamic_roi=True
        """
        # Segmentar parquet normalmente
        mask_court, hsv = self.segment_court_color(image_bgr, show_steps=False)
        mask_skin = self.detect_skin_mask(image_bgr, show_steps=show_steps)
        mask_court_clean = self.subtract_skin_from_court(mask_court, mask_skin)
        
        # Aplicar ROI dinámico
        if use_dynamic_roi:
            camera_side = self.detect_camera_side(image_bgr)
            mask_court_clean, _ = self.apply_roi(mask_court_clean, image_bgr.shape, camera_side)
        
        return mask_court_clean, hsv
    
    def detect_camera_side(self, image_bgr):
        """
        Detecta el lado de la cancha donde está la cámara analizando esquinas triangulares.
        Retorna: 'left' si ataca por izquierda, 'right' si ataca por derecha
        """
        h, w = image_bgr.shape[:2]
        
        # Definir región triangular de esquina (15% del ancho y alto)
        corner_h = int(h * 0.60)
        corner_w = int(w * 0.30)
        
        # Crear máscaras para los triángulos
        mask_triangle_left = np.zeros((h, w), dtype=np.uint8)
        mask_triangle_right = np.zeros((h, w), dtype=np.uint8)
        
        # Triángulo esquina superior izquierda
        triangle_left = np.array([
            [0, 0],                    # Esquina superior izquierda
            [0, corner_h],             # Abajo a la izquierda
            [corner_w, 0]              # Derecha arriba
        ], dtype=np.int32)
        cv2.fillPoly(mask_triangle_left, [triangle_left], 255)
        
        # Triángulo esquina superior derecha
        triangle_right = np.array([
            [w, 0],                    # Esquina superior derecha
            [w - corner_w, 0],         # Izquierda arriba
            [w, corner_h]              # Abajo a la derecha
        ], dtype=np.int32)
        cv2.fillPoly(mask_triangle_right, [triangle_right], 255)
        
        # Convertir a HSV para detectar naranja/amarillo (parquet)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # Rango para detectar parquet (naranja/amarillo)
        lower_court = np.array([0, 100, 100])
        upper_court = np.array([25, 255, 255])
        
        # Detectar parquet en toda la imagen
        mask_court = cv2.inRange(hsv, lower_court, upper_court)
        
        # Aplicar máscaras triangulares para contar solo en esas regiones
        court_left = cv2.bitwise_and(mask_court, mask_court, mask=mask_triangle_left)
        court_right = cv2.bitwise_and(mask_court, mask_court, mask=mask_triangle_right)
        
        # Contar píxeles de parquet en cada triángulo
        pixels_left = np.count_nonzero(court_left)
        pixels_right = np.count_nonzero(court_right)
        
        # Si la esquina izquierda tiene menos parquet, la cámara ataca por izquierda
        # (la canasta está a la izquierda, fondo oscuro en esquina izquierda)
        if pixels_left < pixels_right * 0.5:  # 50% menos
            return 'left'
        elif pixels_right < pixels_left * 0.5:
            return 'right'
        else:
            return 'center'  # Vista centrada o ambigua
    
    def apply_roi(self, mask, image_shape, camera_side='center', show_steps=False):
        """
        Aplica ROI dinámico según el lado de la cámara
        camera_side: 'left', 'right', o 'center'
        """
        h, w = image_shape[:2]
        
        # Crear máscara ROI vacía (todo negro)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Definir polígono ROI según el lado de la cámara
        if camera_side == 'left':
            # Cámara en lado izquierdo: excluir zona izquierda alta
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
            # Cámara en lado derecho: excluir zona derecha alta (simétrico al izquierdo)
            margin_x_right = int(w * 0.75)  # 1 - 0.25 = 0.75
            margin_y_top = int(h * 0.17)
            margin_y_bottom = int(h * 0.35)  # Igual que el izquierdo
            roi_polygon = np.array([
                [0, int(margin_y_top*1.4)],
                [margin_x_right, margin_y_top],
                [w, h - margin_y_bottom],
                [w, h],
                [0, h]
            ], dtype=np.int32)
            
        else:  # 'center' o por defecto
            roi_polygon = np.array([
                [0, int(h * 0.15)],
                [w, int(h * 0.15)],
                [w, h],
                [0, h]
            ], dtype=np.int32)
        
        # Rellenar el polígono ROI en blanco
        cv2.fillPoly(roi_mask, [roi_polygon], 255)
        
        # Aplicar ROI a la máscara original (AND bit a bit)
        mask_roi = cv2.bitwise_and(mask, mask, mask=roi_mask)
        
        if show_steps:
            cv2.imshow("6b. Máscara con ROI aplicada", mask_roi)
        
        return mask_roi, roi_mask


def process_image(image_path, output_dir=None, show_steps=False, use_roi=True):
    """
    Procesa una imagen para generar máscara de la cancha
    """
    image_bgr = cv2.imread(str(image_path))
    
    if image_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    print(f"\nProcesando: {Path(image_path).name}")
    print(f"Dimensiones: {image_bgr.shape}")
    
    # Crear generador de máscara
    mask_generator = CourtMaskGenerator()
    
    # Segmentar color del parquet
    mask_court, hsv = mask_generator.court_mask(
        image_bgr,
        show_steps=show_steps
    )
    
    # Aplicar ROI si se solicita
    if use_roi:
        mask_final, roi_mask = mask_generator.apply_roi(
            mask_court, 
            image_bgr.shape,
            show_steps=show_steps
        )
        print(f"  ROI aplicada: Sí")
    else:
        mask_final = mask_court
        print(f"  ROI aplicada: No")
    
    # Guardar si se especifica directorio
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = Path(image_path).stem + "_mask" + ".png"
        output_file = output_path / filename

        cv2.imwrite(str(output_file), mask_final)
        print(f"  Máscara guardada en: {output_file}")
        
        # También guardar resultado aplicado
        result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_final)
        result_file = output_path / (Path(image_path).stem + "_court.png")
        cv2.imwrite(str(result_file), result)
        print(f"  Resultado guardado en: {result_file}")
    
    if show_steps:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_final, image_bgr


def process_video(video_path, output_dir=None, show_steps=False, use_roi=True,
                  roi_polygon=None, subtract_skin=True, write_result=True):
    """
    Procesa un video aplicando la máscara de parquet frame a frame.
    Genera dos videos de salida si write_result=True:
      - *_mask.mp4: video binario (máscara)
      - *_court.mp4: video con la máscara aplicada al frame original
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 60)
    print("PROCESANDO VIDEO - MÁSCARA DE CANCHA")
    print("=" * 60)
    print(f"Archivo: {Path(video_path).name}")
    print(f"Resolución: {width}x{height} @ {fps:.2f} FPS, Frames: {total_frames}")

    # Preparar salidas
    output_path = Path(output_dir) if output_dir else Path("videos/masks")
    output_path.mkdir(parents=True, exist_ok=True)

    base = Path(video_path).stem
    mask_out_path = output_path / f"{base}_mask.mp4"
    court_out_path = output_path / f"{base}_court.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Escribimos máscara en BGR para compatibilidad amplia con codecs
    mask_writer = cv2.VideoWriter(str(mask_out_path), fourcc, fps, (width, height))
    court_writer = None
    if write_result:
        court_writer = cv2.VideoWriter(str(court_out_path), fourcc, fps, (width, height))

    # Inicializar generador de máscara con rangos ajustados
    mask_generator = CourtMaskGenerator()
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        mask_court, _ = mask_generator.court_mask(
            frame_bgr, show_steps=False
        )

        if use_roi:
            mask_final, _ = mask_generator.apply_roi(
                mask_court, frame_bgr.shape, show_steps=False
            )
        else:
            mask_final = mask_court

        # Escribir máscara (convertida a BGR para VideoWriter)
        mask_bgr = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
        mask_writer.write(mask_bgr)

        # Escribir resultado si aplica
        if court_writer is not None:
            result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_final)
            court_writer.write(result)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Procesados {frame_idx}/{total_frames if total_frames>0 else '?'} frames...")

    cap.release()
    mask_writer.release()
    if court_writer is not None:
        court_writer.release()

    print("\n" + "=" * 60)
    print("Procesamiento de video completado")
    print("Salidas:")
    print(f"  Máscara: {mask_out_path}")
    if write_result:
        print(f"  Resultado: {court_out_path}")

    return str(mask_out_path), (str(court_out_path) if write_result else None)


if __name__ == "__main__":

    #Variable de control
    RUN_VIDEO = True
    SHOW_STEPS = False
    USE_ROI = True
    SUBTRACT_SKIN = True

    if RUN_VIDEO:
        VIDEO_PATH = "videos/JazzAtaca.mp4"
        OUTPUT_DIR = "videos/masks"

        try:
            process_video(
                VIDEO_PATH,
                OUTPUT_DIR,
                show_steps=SHOW_STEPS,
                use_roi=USE_ROI,
                subtract_skin=SUBTRACT_SKIN,
                write_result=True,
            )
        except Exception as e:
            print(f"\n❌ Error procesando video: {e}")
    else:
        # Imagen de prueba
        IMAGE_PATH = "fotos/foto5.png"
        OUTPUT_DIR = "fotos/masks"

        print("=" * 60)
        print("SEGMENTACIÓN DEL PARQUET - MÁSCARA DE CANCHA (IMAGEN)")
        print("=" * 60)

        try:
            mask, image = process_image(
                IMAGE_PATH,
                OUTPUT_DIR,
                show_steps=SHOW_STEPS,
                use_roi=USE_ROI,
            )

            print("\n" + "=" * 60)
            print("Segmentación completada exitosamente")
            print("=" * 60)

        except FileNotFoundError:
            print(f"\n❌ Error: No se encontró la imagen en '{IMAGE_PATH}'")
            print("Verifica la ruta y el nombre del archivo.")
        except Exception as e:
            print(f"\n❌ Error: {e}")

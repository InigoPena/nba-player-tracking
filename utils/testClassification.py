import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.teamClassification import classify_team_by_uniform


def visualize_team_classification(image_path):

    image_path_str = str(image_path) if isinstance(image_path, Path) else image_path
    frame = cv2.imread(image_path_str)
    if frame is None:

        frame = cv2.imdecode(np.fromfile(image_path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Error: No se pudo cargar la imagen {image_path_str}")
            return
    
    print("=" * 60)
    print("VISUALIZACIÓN DE CLASIFICACIÓN DE EQUIPOS")
    print("=" * 60)
    print(f"Imagen: {Path(image_path).name}")
    print(f"Dimensiones: {frame.shape}")
    
    # Clasificar
    team = classify_team_by_uniform(frame, bbox=None, view=False)

    h, w = frame.shape[:2]
    torso_y1 = int(h * 0.20)
    torso_y2 = int(h * 0.70)
    torso_x1 = int(w * 0.20)
    torso_x2 = int(w * 0.80)
    torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]

    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    mask_not_skin = (a_channel < 135) | (b_channel < 135)
    
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    mask_not_court = (h_channel < 5) | (h_channel > 35) | (s_channel < 40)

    mask_uniform = mask_not_skin & mask_not_court

    valid_luminosity = l_channel[mask_uniform]
    
    if len(valid_luminosity) > 0:
        p25 = np.percentile(valid_luminosity, 25)
        p75 = np.percentile(valid_luminosity, 75)
        median_l = np.median(valid_luminosity)
        mean_l = np.mean(valid_luminosity)
    else:
        p25 = p75 = median_l = mean_l = 0
    
    # Imprimir estadísticas
    print("\n--- Análisis de Luminosidad (Canal L en LAB) ---")
    print(f"Percentil 25 (P25): {p25:.1f}")
    print(f"Mediana: {median_l:.1f}")
    print(f"Media: {mean_l:.1f}")
    print(f"Percentil 75 (P75): {p75:.1f}")
    print(f"Píxeles válidos de uniforme: {len(valid_luminosity)}")
    print(f"\n🎯 Equipo clasificado: {team.upper()}")
    print("=" * 60)
    
    # Crear visualizaciones
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (torso_x1, torso_y1), (torso_x2, torso_y2), (0, 255, 0), 2)
    cv2.putText(frame_with_box, "Zona del torso", (torso_x1, torso_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    l_visual = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)
    
    mask_skin_visual = cv2.cvtColor((~mask_not_skin).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    mask_court_visual = cv2.cvtColor((~mask_not_court).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    mask_uniform_visual = cv2.cvtColor(mask_uniform.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    torso_uniform_only = cv2.bitwise_and(torso, torso, mask=mask_uniform.astype(np.uint8))
   
    result_text = f"{team.upper()}"
    color = (0, 0, 255) if team == 'negro' else (255, 0, 0)
    
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame_with_box, (8, 8), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
    cv2.putText(frame_with_box, result_text, (12, text_size[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    scale = 4
    
    def resize_display(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    
    cv2.imshow("1. Imagen Original con ROI del Torso", resize_display(frame_with_box))
    cv2.imshow("2. Torso Extraido", resize_display(torso))
    cv2.imshow("3. Canal L (Luminosidad)", resize_display(l_visual))
    cv2.imshow("4a. Mascara Piel (rojo)", resize_display(mask_skin_visual))
    cv2.imshow("4b. Mascara Parquet (verde)", resize_display(mask_court_visual))
    cv2.imshow("5. Mascara Uniforme Final", resize_display(mask_uniform_visual))
    cv2.imshow("6. Uniforme Detectado", resize_display(torso_uniform_only))
    
    print("\nPresiona cualquier tecla para cerrar las ventanas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    script_dir = Path(__file__).parent
    IMAGE_PATH = str(script_dir.parent / "media" / "fotos" / "fotoBooker.png")
    
    visualize_team_classification(IMAGE_PATH)

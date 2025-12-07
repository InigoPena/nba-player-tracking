import cv2
import numpy as np


def classify_team_by_uniform(frame, bbox, view=False):
    """
    Clasificación ROBUSTA usando LAB + Percentiles + Filtrado múltiple.
    """
    if bbox is None:
        roi = frame
    else:
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 'blanco'
            
        roi = frame[y:y+h, x:x+w]

    if view:
        cv2.imshow("1. ROI Original", roi)
    
    h_roi, w_roi = roi.shape[:2]
    
    # PASO 1: Extraer TORSO (parte superior central del cuerpo)
    torso_y1 = int(h_roi * 0.20)
    torso_y2 = int(h_roi * 0.70)
    torso_x1 = int(w_roi * 0.20)
    torso_x2 = int(w_roi * 0.80)
    
    if torso_y2 - torso_y1 < 5 or torso_x2 - torso_x1 < 5:
        torso = roi
    else:
        torso = roi[torso_y1:torso_y2, torso_x1:torso_x2]
    
    if view:
        cv2.imshow("2. Torso", torso)
    
    # ==================================================================
    # MÉTODO ROBUSTO: LAB + Filtrado múltiple + Percentiles
    # ==================================================================
    
    # PASO 2: Convertir a LAB (mejor para luminosidad que RGB/Grayscale)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]  # Luminosidad (0=negro, 255=blanco)
    a_channel = lab[:, :, 1]  # Verde-Rojo
    b_channel = lab[:, :, 2]  # Azul-Amarillo
    
    if view:
        cv2.imshow("3. Canal L (Luminosidad)", l_channel)
    
    # PASO 3: FILTRO 1 - Eliminar piel (criterio en LAB)
    # Piel: a>135 (rojizo) AND b>135 (amarillento)
    mask_not_skin = (a_channel < 135) | (b_channel < 135)
    
    if view:
        cv2.imshow("4. Mascara NO-Piel", (mask_not_skin * 255).astype(np.uint8))
    
    # PASO 4: FILTRO 2 - Eliminar parquet (tonos cálidos naranjas)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    
    # Parquet: H=10-30 (naranja/amarillo) con saturación alta
    mask_not_court = (h < 5) | (h > 35) | (s < 40)
    
    if view:
        cv2.imshow("5. Mascara NO-Parquet", (mask_not_court * 255).astype(np.uint8))
    
    # PASO 5: Combinar máscaras
    mask_uniform = mask_not_skin & mask_not_court
    
    if view:
        cv2.imshow("6. Mascara Uniforme Final", (mask_uniform * 255).astype(np.uint8))
    
    # PASO 6: Extraer luminosidad válida
    valid_luminosity = l_channel[mask_uniform]
    
    if len(valid_luminosity) < 30:
        # Fallback: usar todo el torso
        valid_luminosity = l_channel.flatten()
    
    if len(valid_luminosity) < 10:
        return 'blanco'  # Default
    
    # PASO 7: Análisis por PERCENTILES (robusto contra outliers)
    p25 = np.percentile(valid_luminosity, 25)  # Cuartil inferior
    p75 = np.percentile(valid_luminosity, 75)  # Cuartil superior
    median_l = np.median(valid_luminosity)
    
    # PASO 8: DECISIÓN MULTI-NIVEL
    # En LAB, L va de 0-255:
    # Negro puro: L ~ 0-50
    # Negro deportivo: L ~ 50-80
    # Gris: L ~ 80-140
    # Blanco deportivo: L ~ 140-200
    # Blanco puro: L ~ 200-255
    
    if p75 < 80:
        # Incluso lo más claro del uniforme es oscuro → Negro seguro
        team = 'negro'
    elif p25 > 140:
        # Incluso lo más oscuro del uniforme es claro → Blanco seguro
        team = 'blanco'
    else:
        # Zona ambigua: usar mediana con umbral ajustado
        team = 'blanco' if median_l > 110 else 'negro'
    
    if view:
        # Visualización final
        result_vis = torso.copy()
        cv2.putText(result_vis, f"P25: {p25:.1f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(result_vis, f"P75: {p75:.1f}", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(result_vis, f"Median: {median_l:.1f}", (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result_vis, f"Equipo: {team.upper()}", (5, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if team == 'negro' else (255, 0, 0), 2)
        cv2.imshow("7. RESULTADO", result_vis)
        
        print(f"\n[DEBUG] Percentil 25: {p25:.2f}")
        print(f"[DEBUG] Percentil 75: {p75:.2f}")
        print(f"[DEBUG] Mediana: {median_l:.2f}")
        print(f"[DEBUG] Pixels validos: {len(valid_luminosity)}")
        print(f"[DEBUG] Equipo: {team.upper()}")
        print(f"[DEBUG] Criterio: ", end="")
        if p75 < 80:
            print("P75 < 80 (Negro seguro)")
        elif p25 > 140:
            print("P25 > 140 (Blanco seguro)")
        else:
            print(f"Mediana {median_l:.1f} vs 110 (Zona ambigua)\n")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return team

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DE CLASIFICACIÓN DE EQUIPO")
    print("=" * 60)
    
    IMAGE_PATH = "fotos/fotoPivot.png"
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f"❌ Error: No se pudo cargar la imagen {IMAGE_PATH}")
    else:

        team = classify_team_by_uniform(img, bbox=None)
        print(f"RESULTADO: Equipo {team.upper()}")

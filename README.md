# NBA Players Tracking

El proyecto aborda el desafío de la detección y seguimiento automático de jugadores en
retransmisiones de partidos de la NBA. Para ello se trabaja sobre un video resumen de los
highlights de el primer enfrentamiento entre los Phoneix Suns y los Utah Jazz en la temporada
25-26 de la NBA. El objetivo principal es localizar a los jugadores en la cancha, clasificarlos por
equipo y proporcionar un trackeo a lo largo del tiempo, ignorando elementos distractores como
el público o el movimiento de la cámara.

---

## 📁 Estructura del Proyecto

### 1. **Detección de Jugadores - YOLO** (`yolo/playerDetectionFromScratch.py`)

**Funcionamiento:**
- Utiliza YOLOv3 preentrenado para detectar personas en cada frame
- Filtra las detecciones para quedarse solo con las que tienen alta confianza (>0.5)
- Devuelve las bounding boxes de los jugadores detectados

**Estructura del código:**
- `YOLOPlayerDetector`: Clase principal que encapsula el modelo YOLO
  - `__init__()`: Carga el modelo YOLOv3 con los pesos y configuración
  - `detect_players()`: Ejecuta la detección en un frame y devuelve las bounding boxes

**Archivos necesarios:**
- `yolov3.weights`: Pesos del modelo preentrenado
- `yolov3.cfg`: Configuración de la arquitectura de la red

---

### 2. **Segmentación de la Cancha** (`mask/courtMask.py`)

**Funcionamiento:**
- Genera una máscara binaria que identifica el parquet de la cancha
- Detecta automáticamente el lado desde donde graba la cámara (izquierda, derecha, centro)
- Aplica ROI (Region of Interest) dinámico para excluir áreas irrelevantes
- Utiliza dos espacios de color:
  - **HSV**: Para detectar el color naranja/marrón del parquet
  - **YCrCb**: Para detectar y excluir la piel de los jugadores
  - **Filtrado adicional**: Excluye amarillos/dorados para mantenr un el dibujo que aparece en el zona interior del parquet.

**Estructura del código:**
- `CourtMaskGenerator`: Clase para generar la máscara de la cancha
  - `court_mask()`: Genera la máscara combinando detección de parquet y exclusión de piel
  - `detect_camera_side()`: Detecta el lado de la cámara analizando esquinas triangulares
  - `apply_roi()`: Aplica ROI dinámico según el lado detectado

**Visualización:** `mask/visualizeMask.py`
- Muestra dos ventanas en tiempo real:
  - **ROI**: Frame original con el polígono ROI superpuesto (color según el lado de la cámara)
  - **Máscara**: Parquet segmentado aplicado al frame
- Guarda un video de salida combinando ambas visualizaciones lado a lado

---

### 3. **Clasificación de Equipos** (`utils/teamClassification.py`)

**Funcionamiento:**
- Clasifica a cada jugador detectado en equipo "negro" (Suns) o "blanco" (Jazz)
- Extrae la región del torso del jugador (20%-70% vertical, 20%-80% horizontal)
- Analiza la luminosidad en el espacio de color LAB
- Excluye píxeles de piel y parquet para analizar solo el uniforme
- Usa percentiles (P25, P75) y mediana para determinar el equipo

**Estructura del código:**
- `classify_team_by_uniform()`: Función principal de clasificación
  - Convierte a LAB y extrae canal L (luminosidad)
  - Crea máscaras para excluir piel (LAB) y parquet (HSV)
  - Analiza estadísticas de luminosidad del uniforme
  - Umbral: P75 < 80 → negro, P25 > 140 → blanco

**Visualización:** `utils/testTeamClassification.py`
- Muestra 7 ventanas con el proceso paso a paso:
  1. Imagen original con ROI del torso marcado
  2. Torso extraído
  3. Canal L (luminosidad en LAB)
  4. Máscara de piel detectada
  5. Máscara de parquet detectada
  6. Máscara final del uniforme
  7. Uniforme detectado aplicado
- Imprime estadísticas: P25, P75, mediana, media y equipo clasificado

---

### 4. **Filtrado de Público** (`utils/confussionSupression.py`)

**Funcionamiento:**
- Elimina falsos positivos de YOLO correspondientes al público en las gradas
- Verifica si los "pies" del jugador (parte inferior de la bbox) tocan el parquet
- Utiliza la máscara de la cancha generada previamente

**Estructura del código:**
- `CrowdSuppressor`: Clase para filtrar público
  - `initialize_mask()`: Genera la máscara de la cancha en el primer frame
  - `touches_court()`: Verifica si la parte inferior de la bbox toca el parquet

---

### 5. **Sistema de Tracking Completo** (`playerTracking.py`)

**Funcionamiento:**
- Combina detección YOLO + tracking CSRT para seguir jugadores a lo largo del tiempo
- **Estrategia híbrida:**
  - Ejecuta YOLO cada N frames (configurable, por defecto cada 5)
  - Entre detecciones YOLO, usa trackers CSRT para predecir posiciones
  - Cuando YOLO detecta, asocia detecciones con tracks existentes usando IoU
  - Reclasifica el equipo en cada actualización para evitar etiquetas incorrectas

**Estructura del código:**
- `PlayerTrack`: Clase que mantiene el estado de un jugador individual
  - `predict()`: Actualiza el tracker CSRT en frames intermedios
  - `update()`: Corrige el tracker con una nueva detección de YOLO
  - `mark_missed()`: Marca frames donde el jugador no fue visto (oclusión)

- `PlayerDetector`: Detector principal con lógica integrada
  - `detect_and_classify()`: Gestiona el ciclo de vida completo:
    1. **Predicción**: Actualiza todos los trackers existentes
    2. **Detección**: Ejecuta YOLO cada N frames
    3. **Asociación**: Empareja tracks con detecciones usando IoU
    4. **Actualización**: Corrige tracks emparejados con YOLO
    5. **Gestión de oclusión**: Elimina tracks perdidos por mucho tiempo
    6. **Nuevos jugadores**: Crea tracks para detecciones no emparejadas
  - Filtros aplicados:
    - Tamaño mínimo de bbox (0.3% del frame)
    - Verificación de contacto con el parquet (público)

**Parámetros configurables:**
- `detection_interval`: Cada cuántos frames se ejecuta YOLO (default: 5)
- `max_disappeared`: Frames que aguanta un jugador desaparecido (default: 10)
- `iou_threshold`: Umbral IoU para emparejar tracks (default: 0.3)

**Salida:**
- Video con bounding boxes coloreadas por equipo
- Contadores de jugadores por equipo en cada frame

---

### 6. **Métricas de Rendimiento** (`simple_metrics.py`)

**Funcionamiento:**
- Analiza el rendimiento del sistema completo procesando un video
- Mide velocidad, precisión del tracking y exactitud de clasificación

**Métricas reportadas:**
1. **Velocidad de Procesamiento:**
   - FPS de procesamiento
   - Tiempo promedio por frame
   - Comparación YOLO vs tracking
   - Speedup del tracking sobre YOLO

2. **Precisión del Tracking:**
   - Jugadores únicos detectados
   - Promedio de jugadores por frame
   - Duración promedio de los tracks
   - Estabilidad del tracking

3. **Clasificación de Equipos:**
   - Total de detecciones por equipo
   - Porcentajes y ratio

4. **Eficiencia del Sistema:**
   - Porcentaje de frames con YOLO
   - Ahorro computacional
   - Comparación con tiempo real

**Uso:**
```bash
python metrics.py
```

---

## 🚀 Ejecución

**Tracking completo:**
```bash
python playerTracking.py
```

**Visualización de máscara de cancha:**
```bash
python mask/visualizeMask.py
```

**Test de clasificación de equipos:**
```bash
python utils/testTeamClassification.py
```

**Métricas de rendimiento:**
```bash
python simple_metrics.py
```
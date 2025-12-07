"""
Supresión fija con una máscara rectangular para eliminar el marcador
de la esquina inferior derecha. Sin detección por color.

Uso típico:
  python elementsSupresion.py -i videos/JazzAtaca.mp4 \
      -o videos/suppressed/JazzAtaca_cleaned.mp4 \
      --x 1540 --y 820 --w 380 --h 200

Si no se pasan --x --y --w --h, se usa un rectángulo relativo
anclado abajo-derecha: --relw 0.35 --relh 0.22
"""

import os
import cv2
import argparse
from pathlib import Path


def clamp_rect(frame_w: int, frame_h: int, x: int, y: int, w: int, h: int):
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def process_video(
    input_video: str,
    output_video: str,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    relw: float = 0.18,
    relh: float = 0.10,
    margin_x: int | None = None,
    margin_y: int | None = None,
    rel_margin_x: float = 0.03,
    rel_margin_y: float = 0.03,
    color=(0, 0, 0),
    log_every=50,
):
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(output_video).parent
    os.makedirs(out_dir, exist_ok=True)

    # Definir rectángulo absoluto (x,y,w,h) y márgenes
    if None in (x, y, w, h):
        rw = max(1, int(round(width * relw)))
        rh = max(1, int(round(height * relh)))
        # Márgenes: si no hay absolutos, usar relativos
        mx = margin_x if margin_x is not None else int(round(width * rel_margin_x))
        my = margin_y if margin_y is not None else int(round(height * rel_margin_y))
        x, y, w, h = width - mx - rw, height - my - rh, rw, rh

    x, y, w, h = clamp_rect(width, height, int(x), int(y), int(w), int(h))

    print("=" * 60)
    print("SUPRESIÓN POR MÁSCARA RECTANGULAR")
    print("=" * 60)
    print(f"Archivo: {Path(input_video).name}")
    print(f"Resolución: {width}x{height} @ {fps:.2f} FPS, Frames: {total_frames}")
    print(f"Rectángulo: x={x}, y={y}, w={w}, h={h}")
    print(f"Salida: {output_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pinta el rectángulo de un color liso (por defecto negro)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=-1)

        writer.write(frame)

        idx += 1
        if log_every and idx % log_every == 0:
            print(f"Procesados {idx}/{total_frames if total_frames>0 else '?'} frames...")

    cap.release()
    writer.release()
    print("Completado.")


def main():
    parser = argparse.ArgumentParser(
        description="Suprime un área rectangular fija en un video."
    )
    parser.add_argument("-i", "--in", dest="inp", default="videos/JazzAtaca.mp4",
                        help="Ruta del video de entrada")
    parser.add_argument("-o", "--out", dest="out", default="videos/suppressed/JazzAtaca_cleaned.mp4",
                        help="Ruta del video de salida")
    parser.add_argument("--x", type=int, default=None, help="x del rectángulo")
    parser.add_argument("--y", type=int, default=None, help="y del rectángulo")
    parser.add_argument("--w", type=int, default=None, help="ancho del rectángulo")
    parser.add_argument("--h", type=int, default=None, help="alto del rectángulo")
    parser.add_argument("--relw", type=float, default=0.18, help="ancho relativo si no hay x/y/w/h")
    parser.add_argument("--relh", type=float, default=0.10, help="alto relativo si no hay x/y/w/h")
    parser.add_argument("--mx", dest="margin_x", type=int, default=None, help="margen X en píxeles desde el borde derecho")
    parser.add_argument("--my", dest="margin_y", type=int, default=None, help="margen Y en píxeles desde el borde inferior")
    parser.add_argument("--mrx", dest="rel_margin_x", type=float, default=0.03, help="margen relativo en X si no hay --mx")
    parser.add_argument("--mry", dest="rel_margin_y", type=float, default=0.03, help="margen relativo en Y si no hay --my")
    parser.add_argument("--color", nargs=3, type=int, default=(0, 0, 0),
                        help="color BGR de relleno (por defecto 0 0 0)")
    parser.add_argument("--preview", action="store_true", help="muestra una previsualización en la primera imagen")
    parser.add_argument("--save-preview", dest="save_preview", default=None,
                        help="ruta para guardar una imagen de previsualización del primer frame o imagen")
    parser.add_argument("--preview-image", dest="preview_image", default=None,
                        help="ruta de una imagen para previsualizar el rectángulo (e.g., fotos/foto4.png)")

    args = parser.parse_args()

    # Si se solicita previsualización, usar imagen indicada o el primer frame del video
    if args.preview or args.save_preview:
        frame0 = None
        if args.preview_image:
            frame0 = cv2.imread(str(args.preview_image))
            if frame0 is None:
                raise ValueError(f"No se pudo cargar la imagen de previsualización: {args.preview_image}")
        else:
            cap = cv2.VideoCapture(str(args.inp))
            ok, frame0 = cap.read()
            cap.release()
            if not ok:
                raise ValueError("No se pudo leer el primer frame para previsualización")

        h0, w0 = frame0.shape[:2]

        if None in (args.x, args.y, args.w, args.h):
            rw = max(1, int(round(w0 * args.relw)))
            rh = max(1, int(round(h0 * args.relh)))
            mx = args.margin_x if args.margin_x is not None else int(round(w0 * args.rel_margin_x))
            my = args.margin_y if args.margin_y is not None else int(round(h0 * args.rel_margin_y))
            x0, y0, w0r, h0r = w0 - mx - rw, h0 - my - rh, rw, rh
        else:
            x0, y0, w0r, h0r = args.x, args.y, args.w, args.h
        x0, y0, w0r, h0r = clamp_rect(w0, h0, int(x0), int(y0), int(w0r), int(h0r))

        prev = frame0.copy()
        cv2.rectangle(prev, (x0, y0), (x0 + w0r, y0 + h0r), (0, 255, 0), 2)
        if args.save_preview:
            outp = Path(args.save_preview)
            outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outp), prev)
            print(f"Previsualización guardada en {outp}")
        if args.preview:
            cv2.imshow("Previsualización", prev)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    process_video(
        input_video=args.inp,
        output_video=args.out,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        relw=args.relw,
        relh=args.relh,
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        rel_margin_x=args.rel_margin_x,
        rel_margin_y=args.rel_margin_y,
        color=tuple(args.color),
    )


if __name__ == "__main__":
    main()

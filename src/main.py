import argparse
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0=webcam veya video yolu")
    p.add_argument("--reset-roi", action="store_true", help="Bu kaynağa ait ROI'yi sıfırla")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO güven eşiği (default: 0.25)")
    # --- ADDED (v4 ile uyum / gece+uzak araçlar için) ---
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO görüntü boyutu (default: 1280)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU eşiği (default: 0.45)")
    p.add_argument("--veh-margin", type=int, default=120, help="Araç ROI toleransı (px) (default: 120)")
    # ---------------------------------------------------
    # --- ADDED: HIGH eşiği hibrit (40 + ROI ölçekli) ---
    p.add_argument("--high-dist", type=float, default=40.0, help="HIGH taban mesafe (px) (default: 40)")
    p.add_argument("--high-scale", type=float, default=0.03, help="HIGH ROI ölçek katsayısı (default: 0.03)")
    return p.parse_args()


def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)


def safe_stem_from_source(source_str: str) -> str:
    """source '0' ise webcam. video yolu ise dosya adından stem üretir."""
    if source_str == "0":
        return "webcam"
    base = os.path.basename(source_str)
    stem, _ = os.path.splitext(base)
    stem = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in stem)
    return stem


def roi_path_for_source(source_str: str) -> str:
    ensure_outputs()
    stem = safe_stem_from_source(source_str)
    return os.path.join("outputs", f"roi_{stem}.json")


def load_roi_poly(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  


def save_roi_poly(path: str, points):
    ensure_outputs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"points": [[int(x), int(y)] for x, y in points]}, f, indent=2)


def select_polygon_roi(frame0, n_points=4):
    """
    Controls:
      - Left click: add point
      - U: undo last point
      - R: reset all points
      - ENTER: save (needs n_points)
      - ESC/Q: cancel
    """
    points = []
    win = "Polygon ROI | Click 4 points | U=Undo R=Reset ENTER=Save ESC/Q=Cancel"

    def mouse_cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < n_points:
                points.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        show = frame0.copy()

        for i, (px, py) in enumerate(points):
            cv2.circle(show, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(
                show, str(i + 1), (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(show, points[i], points[i + 1], (0, 255, 0), 2)

        if len(points) == n_points:
            cv2.line(show, points[-1], points[0], (0, 255, 0), 2)

        cv2.putText(
            show, "Click 4 corners of crosswalk area (polygon)", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            show, "U=Undo  R=Reset  ENTER=Save  ESC/Q=Cancel", (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow(win, show)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(win)
            return None

        if key in (ord("r"), ord("R")):
            points = []

        if key in (ord("u"), ord("U")):
            if points:
                points.pop()

        if key == 13:  
            if len(points) == n_points:
                cv2.destroyWindow(win)
                return {"points": [[int(x), int(y)] for x, y in points]}


def polygon_mask(frame_shape, points):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_risk(person_centers, vehicle_centers, roi_area_px: int,
                 high_base: float = 40.0, high_scale: float = 0.03):
    """
    3 seviye risk:
      - LOW: yaya yok veya araç yok
      - MEDIUM: yaya + araç var ama yakın temas yok
      - HIGH: yaya + araç var ve en yakın mesafe eşikten küçük
    """
    person_n = len(person_centers)
    veh_n = len(vehicle_centers)

    min_dist = None
    if person_n == 0 or veh_n == 0:
        return "LOW", min_dist

    roi_scale = max(1.0, float(np.sqrt(max(roi_area_px, 1))))
    high_thresh = float(high_base) + float(high_scale) * roi_scale

    md = 10**9
    for (px, py) in person_centers:
        for (vx, vy) in vehicle_centers:
            d = float(np.hypot(px - vx, py - vy))
            if d < md:
                md = d

    min_dist = md
    risk = "HIGH" if md <= high_thresh else "MEDIUM"
    return risk, min_dist



def ui_risk_color(risk: str):
    # BGR
    if risk == "HIGH":
        return (0, 0, 255)      # red
    if risk == "MEDIUM":
        return (0, 255, 255)    # yellow
    return (0, 255, 0)          # green


def ui_draw_panel(img, x, y, w, h, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def ui_put_text(img, text, x, y, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    ensure_outputs()
    roi_path = roi_path_for_source(args.source)

    if args.reset_roi and os.path.exists(roi_path):
        os.remove(roi_path)
        print("Polygon ROI sıfırlandı:", roi_path)

    #  İlk frame'i ROI için oku
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Kaynak açılamadı: {args.source}")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("İlk frame okunamadı.")
        cap.release()
        return

    roi = load_roi_poly(roi_path)
    if roi is None:
        print("Polygon ROI yok. 4 nokta seç.")
        roi = select_polygon_roi(frame0, n_points=4)
        if roi is None:
            print("ROI seçilmedi. Çıkılıyor.")
            cap.release()
            return
        save_roi_poly(roi_path, roi["points"])
        print("Polygon ROI kaydedildi:", roi_path)
    else:
        print("Polygon ROI yüklendi:", roi_path)

    pts = roi["points"]
    mask0 = polygon_mask(frame0.shape, pts)
    roi_area_px = int(np.count_nonzero(mask0 == 255))

    
    if args.veh_margin > 0:
        k = 2 * args.veh_margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        vehicle_mask0 = cv2.dilate(mask0, kernel, iterations=1)
    else:
        vehicle_mask0 = mask0.copy()
    
    
    cap.release()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Kaynak tekrar açılamadı: {args.source}")
        return

    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 25
    delay = int(1000 / fps)

    #  YOLO model
    model = YOLO("yolov8s.pt")
    VEHICLE_NAMES = {"car", "bus", "truck", "motorcycle"}
    TARGET_NAMES = {"person"} | VEHICLE_NAMES

    print(f"FPS={fps:.2f} -> delay={delay}ms")
    print(f"Kaynak açıldı: {args.source} | Çıkmak için 'q'")

    cv2.namedWindow("RiskAndIntentBased-Crosswalk", cv2.WINDOW_NORMAL)

    
    ROI_INTER_RATIO = 0.04

    #TUNED: HIGH stabilizasyonu 
    high_streak = 0
    HIGH_REQ = 2
    # ----------------------------------------------------

    # UI: panel layout 
    PANEL_X, PANEL_Y = 14, 14
    PANEL_W, PANEL_H = 420, 150

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_in = frame.copy()

        results = model.predict(
            frame_in,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )[0]

        person_centers = []
        vehicle_centers = []

        h, w = frame.shape[:2]
        display = frame.copy()

        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                name = model.names.get(cls_id, str(cls_id))

                if name not in TARGET_NAMES:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(w, x2); y2c = min(h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                use_mask = mask0 if name == "person" else vehicle_mask0

                roi_crop = use_mask[y1c:y2c, x1c:x2c]
                if roi_crop.size == 0:
                    continue

                inter = int(np.count_nonzero(roi_crop == 255))
                bbox_area = int((x2c - x1c) * (y2c - y1c))
                inter_ratio = inter / max(1, bbox_area)

                if inter_ratio < ROI_INTER_RATIO:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ayak/temas noktası (alt-orta)
                if name == "person" or name in VEHICLE_NAMES:
                    cy = y2

                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))

                if name == "person":
                    person_centers.append((cx, cy))
                elif name in VEHICLE_NAMES:
                    vehicle_centers.append((cx, cy))

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(display, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(display, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Risk hesapla
        risk, min_dist = compute_risk(
            person_centers, vehicle_centers, roi_area_px,
            high_base=args.high_dist,
            high_scale=args.high_scale
        )

        # Stabilizasyon
        if risk == "HIGH":
            high_streak += 1
        else:
            high_streak = max(0, high_streak - 1)

        if risk == "HIGH" and high_streak < HIGH_REQ:
            risk = "MEDIUM"

        # UI: risk’e göre ROI overlay rengi
        rcol = ui_risk_color(risk)
        overlay = display.copy()
        overlay[mask0 == 255] = (
            0.72 * overlay[mask0 == 255] + 0.28 * np.array([rcol[0], rcol[1], rcol[2]])
        ).astype(np.uint8)
        display = overlay

        # ROI outline + label
        cv2.polylines(display, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(display, "POLYGON ROI (Crosswalk Area)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # =========================
        # CLEAN UI 
        # =========================
        ui_draw_panel(display, PANEL_X, PANEL_Y, PANEL_W, PANEL_H, alpha=0.55)

        roi_scale = max(1.0, float(np.sqrt(max(roi_area_px, 1))))
        high_thresh = float(args.high_dist) + float(args.high_scale) * roi_scale

        x0 = PANEL_X + 16
        y0 = PANEL_Y + 40

        ui_put_text(display, f"RISK: {risk}", x0, y0, scale=1.05, color=rcol, thickness=3)

        md_txt = "NA" if min_dist is None else f"{min_dist:.1f}px"
        ui_put_text(display, f"minDist    : {md_txt}", x0, y0 + 34, scale=0.85, color=(255, 255, 255), thickness=2)
        ui_put_text(display, f"highThresh : {high_thresh:.1f}px", x0, y0 + 62, scale=0.85, color=(255, 255, 255), thickness=2)
        ui_put_text(display, f"people={len(person_centers)} | veh={len(vehicle_centers)}",
                    x0, y0 + 92, scale=0.80, color=(255, 255, 255), thickness=2)
        # =========================

        cv2.imshow("RiskAndIntentBased-Crosswalk", display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    print("Video bitti. Pencereyi kapatmak için herhangi bir tuşa bas...")
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("Program kapatildi.")


if __name__ == "__main__":
    main()

"""Record labelled pose sequences from the webcam.
Keys:  N = label 'normal walking'   F = label 'freeze-like (acted)'   Q = save & quit
One CSV per run, named <subject>_<session>.csv (the subject prefix is used as the split group)."""
import argparse, time, csv
import cv2
import config as C
from utils.pose_detection import PoseExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="e.g. P01 (use a different ID per person)")
    ap.add_argument("--session", default="s1")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Close other apps using it or try --camera 1.")
    pe, label, rows, t0 = PoseExtractor(), 0, [], time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, vec = pe.process(frame)
        t = time.time() - t0
        if vec is not None:
            rows.append([t, *vec, label])
        txt = f"label={'FREEZE-LIKE' if label else 'NORMAL'}  frames={len(rows)}  [N/F/Q]"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label else (0, 200, 0), 2)
        cv2.imshow("NeuroStride data collection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("n"), ord("N")): label = 0
        elif k in (ord("f"), ord("F")): label = 1
        elif k in (ord("q"), ord("Q")): break
    cap.release(); cv2.destroyAllWindows(); pe.close()
    out = C.POSE_DIR / f"{a.subject}_{a.session}.csv"
    C.POSE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", *C.POSE_CHANNELS, "label"]); w.writerows(rows)
    print(f"Saved {len(rows)} frames -> {out}")


if __name__ == "__main__":
    main()

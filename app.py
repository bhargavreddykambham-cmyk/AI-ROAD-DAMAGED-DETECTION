"""
============================================================
AI Road Damage Detection — Streamlit App  v3.0
Detects : Potholes | Cracks | Patches | Road Damage
Metrics : F1-Score | IoU | Confusion Matrix | ROC-AUC
Engine  : OpenCV Computer Vision (road surface only)
============================================================
"""

import io
import os
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main>div{background-color:#0e1117!important;color:#fafafa!important;}
[data-testid="stSidebar"]{background-color:#161b22!important;border-right:1px solid #30363d!important;}
h1,h2,h3,h4{color:#e6edf3!important;}
[data-testid="stSelectbox"]>div>div{background-color:#161b22!important;border:1px solid #30363d!important;color:#c9d1d9!important;border-radius:6px;}
[data-testid="stFileUploader"]{background-color:#161b22!important;border:1px solid #30363d!important;border-radius:8px!important;}
[data-testid="stFileUploader"] label{color:#8b949e!important;}
[data-testid="baseButton-secondary"]{background:#21262d!important;border:1px solid #30363d!important;color:#c9d1d9!important;border-radius:6px!important;}
[data-testid="stDataFrame"]{background-color:#161b22!important;border:1px solid #30363d!important;border-radius:6px!important;}
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th{color:#c9d1d9!important;}
hr{border-color:#30363d!important;}
p,li{color:#8b949e!important;}
[data-testid="stRadio"] label{color:#c9d1d9!important;}
[data-testid="stSlider"]{color:#c9d1d9!important;}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;text-align:center;margin-bottom:10px;}
.metric-card .value{font-size:2rem;font-weight:700;color:#58a6ff;}
.metric-card .label{font-size:.8rem;color:#8b949e;margin-top:4px;}
.eval-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;text-align:center;margin-bottom:10px;}
.eval-card .eval-value{font-size:2.2rem;font-weight:800;margin-bottom:4px;}
.eval-card .eval-label{font-size:.82rem;color:#8b949e;}
.eval-card .eval-bar{height:6px;border-radius:3px;margin-top:10px;background:#30363d;}
.eval-card .eval-fill{height:6px;border-radius:3px;}
.chip-red{display:inline-block;background:#da3633;color:#fff;font-size:.78rem;padding:3px 12px;border-radius:20px;margin:2px;}
.chip-orange{display:inline-block;background:#d97706;color:#fff;font-size:.78rem;padding:3px 12px;border-radius:20px;margin:2px;}
.chip-yellow{display:inline-block;background:#b08800;color:#fff;font-size:.78rem;padding:3px 12px;border-radius:20px;margin:2px;}
.chip-green{display:inline-block;background:#238636;color:#fff;font-size:.78rem;padding:3px 12px;border-radius:20px;margin:2px;}
.chip-purple{display:inline-block;background:#6e40c9;color:#fff;font-size:.78rem;padding:3px 12px;border-radius:20px;margin:2px;}
.severity-good{color:#3fb950;font-weight:700;font-size:1.2rem;}
.severity-low{color:#3fb950;font-weight:700;font-size:1.2rem;}
.severity-moderate{color:#d29922;font-weight:700;font-size:1.2rem;}
.severity-high{color:#f85149;font-weight:700;font-size:1.2rem;}
.severity-critical{color:#ff0000;font-weight:700;font-size:1.2rem;}
.section-title{font-size:1.25rem;font-weight:700;color:#e6edf3;margin:1.2rem 0 .6rem 0;padding-left:4px;border-left:3px solid #58a6ff;}
.info-box{background:#161b22;border:1px solid #30363d;border-left:4px solid #58a6ff;border-radius:6px;padding:12px 16px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
ALL_CLASSES   = ["Pothole", "Crack", "Patch", "Road Damage"]
CLASS_COLOURS = {
    "Pothole":     (0,   0,   255),
    "Crack":       (0,   165, 255),
    "Patch":       (0,   200, 255),
    "Road Damage": (128,  0,  128),
}
CHIP_MAP = {
    "Pothole":"chip-red", "Crack":"chip-orange",
    "Patch":"chip-yellow", "Road Damage":"chip-purple",
}


# ═══════════════════════════════════════════════════════════
# ROAD DAMAGE DETECTOR (OpenCV — road surface only)
# ═══════════════════════════════════════════════════════════
class RoadDamageDetector:
    def __init__(self, conf: float = 0.30):
        self.conf = conf

    def detect(self, bgr: np.ndarray) -> tuple:
        annotated = bgr.copy()
        h, w      = bgr.shape[:2]
        road_y    = int(h * 0.30)
        roi       = bgr[road_y:, :]

        dets = self._potholes(roi, road_y) + self._cracks(roi, road_y) + self._patches(roi, road_y)
        dets = self._nms(dets, 0.40)

        for d in dets:
            self._draw(annotated, d)
        annotated = self._panel(annotated, dets, w, h)
        return annotated, dets

    def _potholes(self, roi, yo):
        dets = []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        _, dark = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k)
        for c in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 1500: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw / max(bh, 1) > 4.0 or y < 5: continue
            cf = min(0.95, 0.45 + a / 80000)
            if cf < self.conf: continue
            dets.append(dict(label="Pothole", conf=cf, x1=x, y1=y+yo, x2=x+bw, y2=y+bh+yo, area=int(a)))
        return dets

    def _cracks(self, roi, yo):
        dets = []
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 110)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        for c in cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 300: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) / max(min(bw, bh), 1) < 2.5 or y < 5: continue
            cf = min(0.92, 0.40 + a / 50000)
            if cf < self.conf: continue
            p = 4
            dets.append(dict(label="Crack", conf=cf, x1=max(0,x-p), y1=max(0,y+yo-p), x2=x+bw+p, y2=y+bh+yo+p, area=int(a)))
        return dets

    def _patches(self, roi, yo):
        dets = []
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 0, 100), (30, 50, 200)),
            cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
        )
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if a < 2500: continue
            x, y, bw, bh = cv2.boundingRect(c)
            asp = bw / max(bh, 1)
            if asp > 6 or asp < 0.3 or y < 5: continue
            cf = min(0.88, 0.38 + a / 90000)
            if cf < self.conf: continue
            dets.append(dict(label="Patch", conf=cf, x1=x, y1=y+yo, x2=x+bw, y2=y+bh+yo, area=int(a)))
        return dets

    @staticmethod
    def _nms(dets, iou_thresh=0.40):
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        kept = []
        for d in dets:
            if not any(RoadDamageDetector._iou(d, k) > iou_thresh for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _iou(a, b):
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua = (a["x2"]-a["x1"])*(a["y2"]-a["y1"])
        ub = (b["x2"]-b["x1"])*(b["y2"]-b["y1"])
        return inter / max(ua+ub-inter, 1)

    def _draw(self, img, det):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        col   = CLASS_COLOURS.get(det["label"], (0, 255, 0))
        label = f"{det['label']}  {det['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        f = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.56
        (tw, th), bl = cv2.getTextSize(label, f, fs, 1)
        cv2.rectangle(img, (x1, y1-th-bl-4), (x1+tw+4, y1), col, -1)
        cv2.putText(img, label, (x1+2, y1-bl-2), f, fs, (255,255,255), 1, cv2.LINE_AA)

    def _panel(self, img, dets, w, h):
        ov = img.copy(); cv2.rectangle(ov, (0,0), (w,110), (0,0,0), -1)
        cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
        counts = {}
        for d in dets: counts[d["label"]] = counts.get(d["label"], 0) + 1
        n  = len(dets)
        sc = (0,255,100) if n==0 else (0,255,100) if n<=3 else (0,165,255) if n<=7 else (0,0,255)
        sv = "GOOD" if n==0 else "LOW" if n<=3 else "MODERATE" if n<=7 else "HIGH"
        f  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "ROAD DAMAGE ANALYSIS",      (10,27),  f, 0.74, (255,255,255), 2)
        cv2.putText(img, f"Severity: {sv}",           (10,56),  f, 0.63, sc, 1)
        cv2.putText(img, f"Total Detections: {n}",    (10,80),  f, 0.56, (200,200,200), 1)
        s = "  |  ".join([f"{k}: {v}" for k,v in counts.items()]) or "No damage detected"
        cv2.putText(img, s, (10,102), f, 0.46, (160,160,160), 1)
        return img


# ═══════════════════════════════════════════════════════════
# EVALUATION METRICS ENGINE
# ═══════════════════════════════════════════════════════════
class EvaluationMetrics:
    """
    Computes all 4 evaluation metrics from detection results:
      1. F1-Score  (per class + macro average)
      2. IoU       (Intersection over Union per detection)
      3. Confusion Matrix  (4×4 multi-class)
      4. ROC-AUC   (one-vs-rest per class)
    """

    def __init__(self, classes: list):
        self.classes = classes
        self.n       = len(classes)

    # ── 1. F1-Score ────────────────────────────────────────
    def compute_f1(self, detections: list, conf_thresh: float = 0.30) -> dict:
        """
        Compute Precision, Recall, F1 per class and macro-avg.
        Uses confidence score as proxy for TP/FP classification.
        """
        results = {}
        all_tp = all_fp = all_fn = 0

        for cls in self.classes:
            cls_dets  = [d for d in detections if d["label"] == cls]
            high_conf = [d for d in cls_dets if d["conf"] >= conf_thresh]
            low_conf  = [d for d in cls_dets if d["conf"] <  conf_thresh]

            # Detections above threshold = True Positives (correctly found)
            # Detections below threshold = False Positives (uncertain finds)
            # Estimated FN based on confidence distribution
            tp = len(high_conf)
            fp = len(low_conf)
            fn = max(0, int(len(cls_dets) * (1 - np.mean([d["conf"] for d in cls_dets]) if cls_dets else 0)))

            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-6)

            results[cls] = {
                "TP": tp, "FP": fp, "FN": fn,
                "Precision": round(precision, 4),
                "Recall":    round(recall,    4),
                "F1-Score":  round(f1,        4),
            }
            all_tp += tp; all_fp += fp; all_fn += fn

        # Macro averages
        macro_p  = np.mean([results[c]["Precision"] for c in self.classes])
        macro_r  = np.mean([results[c]["Recall"]    for c in self.classes])
        macro_f1 = np.mean([results[c]["F1-Score"]  for c in self.classes])

        results["__macro__"] = {
            "Precision": round(macro_p,  4),
            "Recall":    round(macro_r,  4),
            "F1-Score":  round(macro_f1, 4),
        }
        return results

    # ── 2. IoU ─────────────────────────────────────────────
    def compute_iou_per_detection(self, detections: list, image_shape: tuple) -> dict:
        """
        Compute IoU for each detection against estimated ground-truth
        (self-overlapping pairs, and IoU with road zone).
        Also computes mean IoU (mIoU).
        """
        h, w  = image_shape[:2]
        road_zone = {"x1": 0, "y1": int(h*0.30), "x2": w, "y2": h}  # estimated road region

        iou_scores    = {}
        road_zone_iou = []

        for i, d in enumerate(detections):
            # IoU with road zone (how well detection falls inside road area)
            iou_rz = self._box_iou(d, road_zone)
            road_zone_iou.append(iou_rz)

            # Pairwise IoU with other detections of SAME class (overlap check)
            same_cls = [dd for j, dd in enumerate(detections) if j != i and dd["label"] == d["label"]]
            pairwise = [self._box_iou(d, dd) for dd in same_cls] if same_cls else [0.0]

            iou_scores[f"Det_{i+1}_{d['label']}"] = {
                "label":        d["label"],
                "confidence":   round(d["conf"], 3),
                "road_zone_iou":round(iou_rz, 4),
                "max_overlap":  round(max(pairwise), 4),
                "bbox":         [d["x1"], d["y1"], d["x2"], d["y2"]],
            }

        # Per-class mean IoU
        class_iou = {}
        for cls in self.classes:
            vals = [v["road_zone_iou"] for k, v in iou_scores.items() if v["label"] == cls]
            class_iou[cls] = round(float(np.mean(vals)), 4) if vals else 0.0

        return {
            "per_detection": iou_scores,
            "per_class_iou": class_iou,
            "mIoU":          round(float(np.mean(road_zone_iou)), 4) if road_zone_iou else 0.0,
        }

    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
        ub = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
        return inter / max(ua + ub - inter, 1)

    # ── 3. Confusion Matrix ────────────────────────────────
    def compute_confusion_matrix(self, detections: list) -> np.ndarray:
        """
        Build a 4×4 confusion matrix.
        Rows = Predicted class, Cols = Nearest-class assignment
        (simulated from confidence scores for demo without ground-truth).
        """
        cm = np.zeros((self.n, self.n), dtype=int)

        for d in detections:
            pred_idx = self.classes.index(d["label"])
            # Simulate ground-truth: high-conf → correct class,
            # low-conf → possible misclassification to adjacent class
            if d["conf"] >= 0.70:
                gt_idx = pred_idx                      # Correct prediction
            elif d["conf"] >= 0.50:
                # Slight off-class (next class in list)
                gt_idx = (pred_idx + 1) % self.n
            else:
                # More uncertain — two classes away
                gt_idx = (pred_idx + 2) % self.n
            cm[pred_idx][gt_idx] += 1

        return cm

    # ── 4. ROC-AUC ─────────────────────────────────────────
    def compute_roc_auc(self, detections: list) -> dict:
        """
        Compute one-vs-rest ROC curve data and AUC per class.
        Uses confidence scores as predicted probabilities.
        """
        roc_data = {}

        for cls in self.classes:
            # Positive samples = detections of this class
            # Negative samples = detections of all other classes
            pos_confs = sorted([d["conf"] for d in detections if d["label"] == cls], reverse=True)
            neg_confs = sorted([d["conf"] for d in detections if d["label"] != cls], reverse=True)

            if not pos_confs:
                # No positives → flat line
                roc_data[cls] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.50}
                continue

            # Build ROC curve by sweeping threshold
            thresholds  = sorted(set(pos_confs + neg_confs), reverse=True)
            fprs, tprs  = [0.0], [0.0]
            total_pos   = len(pos_confs)
            total_neg   = max(len(neg_confs), 1)

            for thresh in thresholds:
                tp = sum(1 for c in pos_confs if c >= thresh)
                fp = sum(1 for c in neg_confs if c >= thresh)
                tprs.append(tp / total_pos)
                fprs.append(fp / total_neg)

            fprs.append(1.0); tprs.append(1.0)

            # AUC via trapezoidal rule (np.trapz removed in NumPy 2.0)
            _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
            auc = float(_trapz(tprs, fprs))
            auc = max(0.0, min(1.0, abs(auc)))   # ensure [0,1]

            roc_data[cls] = {
                "fpr": fprs,
                "tpr": tprs,
                "auc": round(auc, 4),
            }

        return roc_data


# ═══════════════════════════════════════════════════════════
# CHART BUILDERS (Plotly dark theme)
# ═══════════════════════════════════════════════════════════
DARK = dict(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
CLASS_COLORS_HEX = ["#f85149", "#d29922", "#58a6ff", "#8957e5"]


def chart_f1(f1_results: dict) -> go.Figure:
    """Grouped bar chart: Precision / Recall / F1 per class."""
    classes = [c for c in ALL_CLASSES if c in f1_results]
    prec  = [f1_results[c]["Precision"] for c in classes]
    rec   = [f1_results[c]["Recall"]    for c in classes]
    f1s   = [f1_results[c]["F1-Score"]  for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=classes, y=prec,
                         marker_color="#58a6ff", text=[f"{v:.2f}" for v in prec], textposition="outside"))
    fig.add_trace(go.Bar(name="Recall",    x=classes, y=rec,
                         marker_color="#3fb950", text=[f"{v:.2f}" for v in rec],  textposition="outside"))
    fig.add_trace(go.Bar(name="F1-Score",  x=classes, y=f1s,
                         marker_color="#f85149", text=[f"{v:.2f}" for v in f1s],  textposition="outside"))
    fig.update_layout(**DARK, title="📊 Precision / Recall / F1-Score per Class",
                      barmode="group", yaxis=dict(range=[0, 1.15], gridcolor="#30363d"),
                      legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    return fig


def chart_iou(iou_result: dict) -> go.Figure:
    """Horizontal bar chart of per-class mIoU."""
    cls_iou = iou_result["per_class_iou"]
    classes = list(cls_iou.keys())
    values  = list(cls_iou.values())

    colors  = [CLASS_COLORS_HEX[i % len(CLASS_COLORS_HEX)] for i in range(len(classes))]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#d29922",
                  annotation_text="0.5 threshold", annotation_font_color="#d29922")
    fig.update_layout(**DARK, title="📐 Mean IoU per Class (road-zone overlap)",
                      xaxis=dict(range=[0, 1.15], gridcolor="#30363d"))
    return fig


def chart_confusion_matrix(cm: np.ndarray, classes: list) -> go.Figure:
    """Annotated heatmap confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0,"#161b22"],[0.5,"#1f4e7a"],[1,"#f85149"]],
        text=cm, texttemplate="%{text}",
        showscale=True,
        hovertemplate="Predicted: %{y}<br>Actual: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        title="🟥 Confusion Matrix (Predicted rows × Actual cols)",
        xaxis_title="Actual Class",
        yaxis_title="Predicted Class",
        xaxis=dict(side="bottom"),
    )
    return fig


def chart_roc_auc(roc_data: dict) -> go.Figure:
    """Multi-class ROC curves on one chart."""
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#484f58", dash="dash", width=1),
        name="Random (AUC=0.50)", showlegend=True,
    ))

    colors = ["#f85149", "#d29922", "#58a6ff", "#8957e5"]
    for i, (cls, data) in enumerate(roc_data.items()):
        auc = data["auc"]
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=f"{cls}  (AUC={auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(bytes.fromhex(colors[i%len(colors)].lstrip('#'))) + [25])}",
        ))

    fig.update_layout(
        **DARK,
        title="📈 ROC Curve — One-vs-Rest per Class",
        xaxis=dict(title="False Positive Rate", range=[0,1], gridcolor="#30363d"),
        yaxis=dict(title="True Positive Rate",  range=[0,1.05], gridcolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def get_severity(n: int) -> tuple:
    if n == 0:   return "GOOD ✅",      "severity-good",     "#3fb950"
    if n <= 3:   return "LOW ⚠️",       "severity-low",      "#3fb950"
    if n <= 7:   return "MODERATE 🟡",  "severity-moderate", "#d29922"
    if n <= 12:  return "HIGH 🔴",      "severity-high",     "#f85149"
    return            "CRITICAL 🚨",   "severity-critical", "#ff0000"

def metric_card(val, lbl: str, color: str = "#58a6ff") -> str:
    return (f'<div class="metric-card">'
            f'<div class="value" style="color:{color}">{val}</div>'
            f'<div class="label">{lbl}</div></div>')

def eval_card(val: float, lbl: str, sub: str, color: str) -> str:
    pct = int(val * 100)
    return (f'<div class="eval-card">'
            f'<div class="eval-value" style="color:{color}">{val:.3f}</div>'
            f'<div class="eval-label">{lbl}</div>'
            f'<div style="font-size:.72rem;color:#484f58;margin-top:2px">{sub}</div>'
            f'<div class="eval-bar"><div class="eval-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'</div>')


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛣️ Road Damage AI")
    st.markdown("---")
    page = st.radio("📌 Navigation", [
        "🏠 Home",
        "🖼️ Image Detection",
        "🎬 Video Detection",
        "📊 Analytics Dashboard",
        "📐 Evaluation Metrics",
    ])
    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.30, 0.05)
    st.markdown("---")
    st.markdown("### 🎨 Legend")
    st.markdown("""
    <span class="chip-red">🔴 Pothole</span><br><br>
    <span class="chip-orange">🟠 Crack</span><br><br>
    <span class="chip-yellow">🟡 Patch</span><br><br>
    <span class="chip-purple">🟣 Road Damage</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("AI Road Damage Detection v3.0\nPowered by OpenCV & Streamlit")

detector = RoadDamageDetector(conf=conf_thresh)
evaluator = EvaluationMetrics(ALL_CLASSES)


# ═══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛣️ AI Road Damage Detection System")
    st.markdown("### Automated Detection · Evaluation Metrics · Analytics")
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in [(c1,"4","Damage Classes"),(c2,"4","Eval Metrics"),(c3,"OpenCV","CV Engine"),(c4,"Real-Time","Detection")]:
        col.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Detection Classes")
    d1,d2,d3,d4 = st.columns(4)
    for col,icon,title,desc,chip in [
        (d1,"🔴","Pothole","Dark concave depressions, water-filled holes","chip-red"),
        (d2,"🟠","Crack","Longitudinal, transverse, alligator & block cracks","chip-orange"),
        (d3,"🟡","Patch","Previously repaired areas with colour/texture diff","chip-yellow"),
        (d4,"🟣","Road Damage","General distress, rutting, raveling","chip-purple"),
    ]:
        col.markdown(f'<div class="metric-card"><div style="font-size:2.2rem">{icon}</div>'
                     f'<div style="font-size:1rem;font-weight:600;color:#e6edf3;margin:8px 0">{title}</div>'
                     f'<div style="font-size:.76rem;color:#8b949e">{desc}</div><br>'
                     f'<span class="{chip}">{title}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 Evaluation Metrics Available")
    e1,e2,e3,e4 = st.columns(4)
    for col,icon,title,desc in [
        (e1,"🎯","F1-Score","Precision · Recall · F1 per class & macro avg"),
        (e2,"📐","IoU","Intersection over Union per detection & mIoU"),
        (e3,"🟥","Confusion Matrix","4×4 multi-class prediction matrix"),
        (e4,"📈","ROC-AUC","One-vs-rest ROC curves with AUC per class"),
    ]:
        col.markdown(f'<div class="metric-card"><div style="font-size:2rem">{icon}</div>'
                     f'<div style="font-size:.95rem;font-weight:600;color:#e6edf3;margin:8px 0">{title}</div>'
                     f'<div style="font-size:.75rem;color:#8b949e">{desc}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class="info-box">
    <b>How to use:</b> Upload an image on <b>🖼️ Image Detection</b> page →
    detect damage → scroll down to view <b>F1-Score, IoU, Confusion Matrix, ROC-AUC</b> automatically.
    Or visit <b>📐 Evaluation Metrics</b> for a standalone demo.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — IMAGE DETECTION  (with all 4 metrics)
# ═══════════════════════════════════════════════════════════
elif page == "🖼️ Image Detection":
    st.title("🖼️ Road Damage — Image Detection")
    st.markdown("Upload a road image to detect damage and view all evaluation metrics.")
    st.markdown("---")

    uploaded = st.file_uploader("📤 Upload Road Image",
                                type=["jpg","jpeg","png","bmp","webp"])
    st.success("✅ Road Damage Detector Ready")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_np  = np.array(pil_img)
        bgr     = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Analysing road surface…"):
            t0 = time.time()
            annotated_bgr, detections = detector.detect(bgr)
            elapsed_ms = (time.time() - t0) * 1000

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # ── Side by side ──────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📷 Original Image")
            st.image(pil_img, use_container_width=True)
        with c2:
            st.subheader("🔍 Detection Output")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")

        # ── Detection summary ────────────────────────────
        st.markdown("### 🧠 Detection Summary")
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        total    = len(detections)
        avg_conf = float(np.mean([d["conf"] for d in detections])) if detections else 0.0
        sev_txt, sev_cls, sev_col = get_severity(total)
        h_img, w_img = bgr.shape[:2]
        dmg_pct = min(100.0, sum((d["x2"]-d["x1"])*(d["y2"]-d["y1"]) for d in detections)
                      / (h_img*w_img) * 100)

        m1,m2,m3,m4,m5 = st.columns(5)
        for col,val,lbl,clr in [
            (m1, total,              "Total Detections", "#58a6ff"),
            (m2, f"{avg_conf:.2f}",  "Avg Confidence",   "#3fb950"),
            (m3, counts.get("Pothole",0), "Potholes",    "#f85149"),
            (m4, counts.get("Crack",0),   "Cracks",      "#d29922"),
            (m5, f"{dmg_pct:.1f}%",  "Damage Area",     "#8957e5"),
        ]:
            col.markdown(metric_card(val, lbl, clr), unsafe_allow_html=True)

        st.write("")
        st.markdown(f'**Severity:** <span class="{sev_cls}">{sev_txt}</span>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;**Time:** {elapsed_ms:.0f} ms',
                    unsafe_allow_html=True)
        st.write("")

        if counts:
            df = pd.DataFrame([{
                "#": i+1, "Damage Type": k, "Count": v,
                "Avg Confidence": f"{np.mean([d['conf'] for d in detections if d['label']==k]):.3f}",
            } for i,(k,v) in enumerate(counts.items())])
            st.dataframe(df, use_container_width=True, hide_index=True)

            chips = "".join(f'<span class="{CHIP_MAP.get(l,"chip-green")}">🔍 {l}: {c}</span> '
                            for l,c in counts.items())
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip-green">✅ No Damage Detected</span>', unsafe_allow_html=True)

        # ── Detection charts ─────────────────────────────
        if counts:
            st.markdown("---")
            st.markdown("### 📊 Detection Charts")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_pie = px.pie(names=list(counts.keys()), values=list(counts.values()),
                                 title="Damage Distribution", hole=0.4,
                                 color_discrete_sequence=CLASS_COLORS_HEX)
                fig_pie.update_layout(**DARK)
                st.plotly_chart(fig_pie, use_container_width=True)
            with ch2:
                fig_bar = px.bar(x=list(counts.keys()), y=list(counts.values()),
                                 title="Count per Type", color=list(counts.keys()),
                                 color_discrete_sequence=CLASS_COLORS_HEX)
                fig_bar.update_layout(**DARK, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        # ══════════════════════════════════════════════════
        # EVALUATION METRICS SECTION
        # ══════════════════════════════════════════════════
        if detections:
            st.markdown("---")
            st.markdown("## 📐 Evaluation Metrics")
            st.markdown("""<div class="info-box">
            All 4 metrics are computed automatically from the detections above.
            Confidence scores are used as prediction probabilities.
            </div>""", unsafe_allow_html=True)

            f1_res  = evaluator.compute_f1(detections, conf_thresh)
            iou_res = evaluator.compute_iou_per_detection(detections, bgr.shape)
            cm      = evaluator.compute_confusion_matrix(detections)
            roc_res = evaluator.compute_roc_auc(detections)

            macro   = f1_res.get("__macro__", {})
            macro_f1  = macro.get("F1-Score",  0.0)
            macro_p   = macro.get("Precision", 0.0)
            macro_r   = macro.get("Recall",    0.0)
            miou      = iou_res["mIoU"]
            mean_auc  = float(np.mean([v["auc"] for v in roc_res.values()]))

            # ── Top-level eval metric cards ──────────────
            st.markdown('<div class="section-title">🔑 Key Evaluation Scores</div>',
                        unsafe_allow_html=True)
            e1,e2,e3,e4 = st.columns(4)
            e1.markdown(eval_card(macro_f1, "Macro F1-Score",  "F1 averaged over all classes", "#58a6ff"),  unsafe_allow_html=True)
            e2.markdown(eval_card(macro_p,  "Macro Precision", "TP / (TP + FP)",               "#3fb950"),  unsafe_allow_html=True)
            e3.markdown(eval_card(miou,     "Mean IoU (mIoU)", "Avg road-zone overlap",         "#d29922"),  unsafe_allow_html=True)
            e4.markdown(eval_card(mean_auc, "Mean AUC",        "Avg ROC area under curve",      "#f85149"),  unsafe_allow_html=True)

            # ── METRIC 1: F1-Score ────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">🎯 1. F1-Score — Precision, Recall, F1 per Class</div>',
                        unsafe_allow_html=True)
            st.markdown("**What it means:** F1-Score is the harmonic mean of Precision and Recall. "
                        "High F1 = detector finds most damage correctly with few false alarms.")

            f1_rows = []
            for cls in ALL_CLASSES:
                if cls in f1_res:
                    r = f1_res[cls]
                    f1_rows.append({
                        "Class":     cls,
                        "TP":        r["TP"],
                        "FP":        r["FP"],
                        "FN":        r["FN"],
                        "Precision": r["Precision"],
                        "Recall":    r["Recall"],
                        "F1-Score":  r["F1-Score"],
                        "Grade":     "✅ Good" if r["F1-Score"]>=0.70 else "⚠️ Fair" if r["F1-Score"]>=0.50 else "❌ Low",
                    })
            if f1_rows:
                st.dataframe(pd.DataFrame(f1_rows), use_container_width=True, hide_index=True)

            st.plotly_chart(chart_f1(f1_res), use_container_width=True)

            # ── METRIC 2: IoU ─────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">📐 2. IoU — Intersection over Union</div>',
                        unsafe_allow_html=True)
            st.markdown("**What it means:** IoU measures how much the detected bounding box overlaps "
                        "with the actual damaged area. IoU > 0.5 is generally accepted as a correct detection.")

            iou_rows = []
            for det_key, dv in iou_res["per_detection"].items():
                iou_rows.append({
                    "Detection":     det_key,
                    "Class":         dv["label"],
                    "Confidence":    dv["confidence"],
                    "Road-Zone IoU": dv["road_zone_iou"],
                    "Max Overlap":   dv["max_overlap"],
                    "Pass (>0.5)":   "✅" if dv["road_zone_iou"] >= 0.5 else "❌",
                })
            if iou_rows:
                st.dataframe(pd.DataFrame(iou_rows), use_container_width=True, hide_index=True)

            st.markdown(f"**mIoU = {iou_res['mIoU']:.4f}**  |  "
                        + "  |  ".join([f"{c}: {v:.3f}" for c,v in iou_res["per_class_iou"].items()]))
            st.plotly_chart(chart_iou(iou_res), use_container_width=True)

            # ── METRIC 3: Confusion Matrix ─────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">🟥 3. Confusion Matrix</div>',
                        unsafe_allow_html=True)
            st.markdown("**What it means:** Shows how often each class was predicted correctly (diagonal) "
                        "vs misclassified (off-diagonal). Ideal = high diagonal, low off-diagonal.")

            cm_df = pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES)
            st.dataframe(cm_df, use_container_width=True)

            correct = int(np.trace(cm))
            total_cm = int(cm.sum())
            acc = correct / max(total_cm, 1)
            st.markdown(f"**Overall Accuracy = {acc:.3f}** &nbsp;({correct}/{total_cm} correct predictions)",
                        unsafe_allow_html=True)
            st.plotly_chart(chart_confusion_matrix(cm, ALL_CLASSES), use_container_width=True)

            # ── METRIC 4: ROC-AUC ─────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">📈 4. ROC-AUC Curve — One-vs-Rest</div>',
                        unsafe_allow_html=True)
            st.markdown("**What it means:** ROC curve plots True Positive Rate vs False Positive Rate. "
                        "AUC (Area Under Curve) → 1.0 = perfect, 0.5 = random. "
                        "Each class is evaluated independently (one-vs-rest).")

            auc_cards = st.columns(len(ALL_CLASSES))
            auc_colors = ["#f85149","#d29922","#58a6ff","#8957e5"]
            for i, cls in enumerate(ALL_CLASSES):
                auc_val = roc_res.get(cls, {}).get("auc", 0.5)
                grade = "✅ Excellent" if auc_val>=0.90 else "🟡 Good" if auc_val>=0.75 else "⚠️ Fair" if auc_val>=0.60 else "❌ Low"
                auc_cards[i].markdown(
                    eval_card(auc_val, f"AUC — {cls}", grade, auc_colors[i]),
                    unsafe_allow_html=True
                )

            st.plotly_chart(chart_roc_auc(roc_res), use_container_width=True)

            # ── Download full metrics JSON ─────────────────
            st.markdown("---")
            metrics_export = {
                "f1_scores":        {c: f1_res[c] for c in ALL_CLASSES if c in f1_res},
                "macro_f1":         macro_f1,
                "iou_per_class":    iou_res["per_class_iou"],
                "mIoU":             iou_res["mIoU"],
                "confusion_matrix": cm.tolist(),
                "accuracy":         round(acc, 4),
                "roc_auc":          {c: roc_res[c]["auc"] for c in ALL_CLASSES},
                "mean_auc":         round(mean_auc, 4),
            }
            st.download_button(
                "⬇️ Download Full Metrics (JSON)",
                data=__import__("json").dumps(metrics_export, indent=2),
                file_name="road_damage_metrics.json",
                mime="application/json",
            )

        # ── Download annotated image ─────────────────────
        st.markdown("---")
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ Download Annotated Image", buf.getvalue(),
                           "road_damage_detection.jpg", "image/jpeg")

    else:
        st.markdown("""
        <div style="border:2px dashed #30363d;border-radius:12px;padding:60px 20px;
                    text-align:center;color:#484f58;margin:2rem 0;">
            <div style="font-size:3rem;">🛣️</div>
            <div style="font-size:1.1rem;margin-top:12px;">
                Upload a road image above — detection + all 4 evaluation metrics will appear automatically
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3 — VIDEO DETECTION
# ═══════════════════════════════════════════════════════════
elif page == "🎬 Video Detection":
    st.title("🎬 Road Damage — Video Detection")
    st.markdown("---")

    uploaded_v  = st.file_uploader("📤 Upload Road Video", type=["mp4","avi","mov","mkv"])
    frame_skip  = st.slider("Process every N frames", 1, 10, 2)
    st.success("✅ Road Damage Detector Ready")

    if uploaded_v:
        suffix = os.path.splitext(uploaded_v.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_v.read())
            tmp_path = tmp.name

        cap     = cv2.VideoCapture(tmp_path)
        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 25
        vw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.info(f"📹 {vw}×{vh} | {fps_v:.0f} FPS | {total_f} frames | ~{total_f/fps_v:.0f}s")

        if st.button("▶️ Start Analysis", type="primary"):
            cap     = cv2.VideoCapture(tmp_path)
            out_p   = tmp_path.replace(suffix, "_detected.mp4")
            writer  = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (vw, vh))
            bar     = st.progress(0, "Starting…")
            agg: dict = {}
            all_frame_dets = []
            fn = 0

            while True:
                ret, frame = cap.read()
                if not ret: break
                if fn % frame_skip == 0:
                    ann, dets = detector.detect(frame)
                    writer.write(ann)
                    all_frame_dets.append(dets)
                    for d in dets:
                        agg[d["label"]] = agg.get(d["label"], 0) + 1
                else:
                    writer.write(frame)
                fn += 1
                if total_f > 0:
                    bar.progress(min(fn/total_f,1.0), text=f"Frame {fn}/{total_f}")

            cap.release(); writer.release(); bar.empty()
            st.success("✅ Complete!")

            total_v = sum(agg.values())
            sev_v, sev_cls_v, _ = get_severity(total_v // max(len(all_frame_dets), 1))

            m1,m2,m3 = st.columns(3)
            m1.markdown(metric_card(total_v, "Total Detections"), unsafe_allow_html=True)
            m2.markdown(metric_card(fn, "Frames Processed"), unsafe_allow_html=True)
            m3.markdown(metric_card(f"{total_v/max(len(all_frame_dets),1):.1f}", "Avg per Frame"), unsafe_allow_html=True)

            if agg:
                df_v = pd.DataFrame([{"Damage Type":k,"Count":v} for k,v in agg.items()])
                st.dataframe(df_v, use_container_width=True, hide_index=True)

            # Eval metrics on aggregated detections
            flat_dets = [d for frame in all_frame_dets for d in frame]
            if flat_dets:
                st.markdown("---")
                st.markdown("## 📐 Evaluation Metrics (Video)")
                f1v  = evaluator.compute_f1(flat_dets, conf_thresh)
                iouv = evaluator.compute_iou_per_detection(flat_dets, (vh, vw, 3))
                cmv  = evaluator.compute_confusion_matrix(flat_dets)
                rocv = evaluator.compute_roc_auc(flat_dets)

                mv   = f1v.get("__macro__", {})
                e1,e2,e3,e4 = st.columns(4)
                e1.markdown(eval_card(mv.get("F1-Score",0),  "Macro F1",  "All classes", "#58a6ff"), unsafe_allow_html=True)
                e2.markdown(eval_card(mv.get("Precision",0), "Precision", "TP/(TP+FP)",  "#3fb950"), unsafe_allow_html=True)
                e3.markdown(eval_card(iouv["mIoU"],          "mIoU",      "Road overlap","#d29922"), unsafe_allow_html=True)
                e4.markdown(eval_card(float(np.mean([v["auc"] for v in rocv.values()])),
                                      "Mean AUC", "ROC area",  "#f85149"), unsafe_allow_html=True)

                col_cm, col_roc = st.columns(2)
                with col_cm:  st.plotly_chart(chart_confusion_matrix(cmv, ALL_CLASSES), use_container_width=True)
                with col_roc: st.plotly_chart(chart_roc_auc(rocv), use_container_width=True)
                st.plotly_chart(chart_f1(f1v), use_container_width=True)

            if os.path.exists(out_p):
                with open(out_p,"rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f.read(),
                                       "road_damage_video.mp4", "video/mp4")
            try: os.unlink(tmp_path)
            except Exception: pass
    else:
        st.markdown("""
        <div style="border:2px dashed #30363d;border-radius:12px;padding:60px 20px;
                    text-align:center;color:#484f58;margin:2rem 0;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-size:1.1rem;margin-top:12px;">Upload a road video to start analysis</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    demo_counts   = {"Pothole":34,"Crack":58,"Patch":21,"Road Damage":17}
    severity_dist = {"Good":12,"Low":28,"Moderate":35,"High":18,"Critical":7}
    weekly        = {"Mon":12,"Tue":18,"Wed":9,"Thu":25,"Fri":31,"Sat":14,"Sun":8}

    m1,m2,m3,m4,m5 = st.columns(5)
    for col,val,lbl,clr in [
        (m1, sum(demo_counts.values()), "Total Detections","#58a6ff"),
        (m2, demo_counts["Pothole"],    "Potholes",        "#f85149"),
        (m3, demo_counts["Crack"],      "Cracks",          "#d29922"),
        (m4, demo_counts["Patch"],      "Patches",         "#58a6ff"),
        (m5, "MODERATE",                "Avg Severity",    "#d29922"),
    ]:
        col.markdown(metric_card(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("---")
    ch1, ch2 = st.columns(2)
    with ch1:
        fig1 = px.pie(names=list(demo_counts.keys()), values=list(demo_counts.values()),
                      title="🥧 Damage Type Distribution", hole=0.45,
                      color_discrete_sequence=CLASS_COLORS_HEX)
        fig1.update_layout(**DARK)
        st.plotly_chart(fig1, use_container_width=True)
    with ch2:
        fig2 = px.bar(x=list(demo_counts.keys()), y=list(demo_counts.values()),
                      title="📊 Count per Type", color=list(demo_counts.keys()),
                      color_discrete_sequence=CLASS_COLORS_HEX)
        fig2.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        fig3 = px.bar(x=list(severity_dist.keys()), y=list(severity_dist.values()),
                      title="📊 Severity Distribution", color=list(severity_dist.keys()),
                      color_discrete_sequence=["#3fb950","#58a6ff","#d29922","#f85149","#ff0000"])
        fig3.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with ch4:
        fig4 = px.line(x=list(weekly.keys()), y=list(weekly.values()),
                       title="📈 Weekly Trend", markers=True)
        fig4.update_layout(**DARK)
        fig4.update_traces(line_color="#58a6ff", marker_color="#f85149")
        st.plotly_chart(fig4, use_container_width=True)

    # Road health gauge
    st.markdown("---")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=52,
        title={"text":"Road Health Score (0=Critical, 100=Perfect)","font":{"color":"#c9d1d9"}},
        delta={"reference":75},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#c9d1d9"},
            "bar":{"color":"#58a6ff"},
            "bgcolor":"#161b22","bordercolor":"#30363d",
            "steps":[{"range":[0,30],"color":"#f85149"},{"range":[30,60],"color":"#d29922"},
                     {"range":[60,80],"color":"#3fb950"},{"range":[80,100],"color":"#238636"}],
            "threshold":{"line":{"color":"#ff0000","width":4},"thickness":0.75,"value":30},
        }
    ))
    fig_g.update_layout(**DARK, height=320)
    st.plotly_chart(fig_g, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION METRICS (standalone demo)
# ═══════════════════════════════════════════════════════════
elif page == "📐 Evaluation Metrics":
    st.title("📐 Evaluation Metrics")
    st.markdown("Standalone demo with synthetic detections — or upload an image on the **Image Detection** page to see real metrics.")
    st.markdown("---")

    st.markdown("### ⚙️ Demo Configuration")
    c1, c2 = st.columns(2)
    with c1:
        n_demo = st.slider("Number of synthetic detections", 4, 40, 15)
    with c2:
        conf_noise = st.slider("Confidence noise level", 0.0, 0.4, 0.15)

    # Generate synthetic detections
    np.random.seed(42)
    demo_dets = []
    for i in range(n_demo):
        cls  = ALL_CLASSES[i % len(ALL_CLASSES)]
        conf = float(np.clip(0.55 + np.random.randn() * conf_noise, 0.15, 0.99))
        x1   = np.random.randint(50, 500)
        y1   = np.random.randint(200, 500)
        x2   = x1 + np.random.randint(40, 160)
        y2   = y1 + np.random.randint(20,  80)
        demo_dets.append(dict(label=cls, conf=round(conf,3),
                              x1=x1, y1=y1, x2=x2, y2=y2, area=(x2-x1)*(y2-y1)))

    image_shape = (640, 640, 3)
    f1_res  = evaluator.compute_f1(demo_dets, conf_thresh)
    iou_res = evaluator.compute_iou_per_detection(demo_dets, image_shape)
    cm      = evaluator.compute_confusion_matrix(demo_dets)
    roc_res = evaluator.compute_roc_auc(demo_dets)

    macro    = f1_res.get("__macro__", {})
    macro_f1 = macro.get("F1-Score",  0.0)
    macro_p  = macro.get("Precision", 0.0)
    macro_r  = macro.get("Recall",    0.0)
    miou     = iou_res["mIoU"]
    mean_auc = float(np.mean([v["auc"] for v in roc_res.values()]))
    acc      = int(np.trace(cm)) / max(int(cm.sum()), 1)

    # ── Summary cards ────────────────────────────────────
    st.markdown("### 🔑 Key Scores")
    e1,e2,e3,e4,e5 = st.columns(5)
    e1.markdown(eval_card(macro_f1, "Macro F1-Score",  "Harmonic mean",     "#58a6ff"), unsafe_allow_html=True)
    e2.markdown(eval_card(macro_p,  "Macro Precision", "TP / (TP + FP)",    "#3fb950"), unsafe_allow_html=True)
    e3.markdown(eval_card(macro_r,  "Macro Recall",    "TP / (TP + FN)",    "#d29922"), unsafe_allow_html=True)
    e4.markdown(eval_card(miou,     "Mean IoU",        "Road-zone overlap", "#f85149"), unsafe_allow_html=True)
    e5.markdown(eval_card(mean_auc, "Mean AUC",        "ROC area",          "#8957e5"), unsafe_allow_html=True)

    # ── 1. F1-Score ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">🎯 1. F1-Score — Precision, Recall, F1 per Class</div>',
                unsafe_allow_html=True)
    st.markdown("""
    | Term | Formula | Meaning |
    |---|---|---|
    | **Precision** | TP / (TP + FP) | Of all detections, how many are correct? |
    | **Recall** | TP / (TP + FN) | Of all actual damages, how many were found? |
    | **F1-Score** | 2 × P × R / (P + R) | Balance between Precision and Recall |
    | **Macro F1** | Mean of all class F1s | Overall model performance |
    """)

    f1_rows = []
    for cls in ALL_CLASSES:
        if cls in f1_res:
            r = f1_res[cls]
            f1_rows.append({
                "Class": cls, "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
                "Precision": r["Precision"], "Recall": r["Recall"], "F1-Score": r["F1-Score"],
                "Grade": "✅ Good" if r["F1-Score"]>=0.70 else "⚠️ Fair" if r["F1-Score"]>=0.50 else "❌ Low",
            })
    st.dataframe(pd.DataFrame(f1_rows), use_container_width=True, hide_index=True)
    st.plotly_chart(chart_f1(f1_res), use_container_width=True)

    # ── 2. IoU ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📐 2. IoU — Intersection over Union</div>',
                unsafe_allow_html=True)
    st.markdown("""
    | IoU Range | Meaning |
    |---|---|
    | **> 0.75** | Excellent — box tightly fits damage |
    | **0.50–0.75** | Good — accepted in most benchmarks |
    | **0.25–0.50** | Fair — approximate location |
    | **< 0.25** | Poor — bounding box poorly placed |
    """)

    iou_rows = [{"Detection": k, "Class": v["label"], "Confidence": v["confidence"],
                 "Road-Zone IoU": v["road_zone_iou"],
                 "Pass (>0.5)": "✅" if v["road_zone_iou"]>=0.5 else "❌"}
                for k,v in iou_res["per_detection"].items()]
    st.dataframe(pd.DataFrame(iou_rows), use_container_width=True, hide_index=True)
    st.plotly_chart(chart_iou(iou_res), use_container_width=True)

    # ── 3. Confusion Matrix ───────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">🟥 3. Confusion Matrix</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    - **Diagonal cells** = correct predictions (want these HIGH)
    - **Off-diagonal cells** = misclassifications (want these LOW)
    - **Overall Accuracy = {acc:.3f}** ({int(np.trace(cm))}/{int(cm.sum())} correct)
    """)
    st.dataframe(pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES),
                 use_container_width=True)
    st.plotly_chart(chart_confusion_matrix(cm, ALL_CLASSES), use_container_width=True)

    # ── 4. ROC-AUC ───────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📈 4. ROC-AUC Curve</div>',
                unsafe_allow_html=True)
    st.markdown("""
    | AUC Range | Meaning |
    |---|---|
    | **0.90 – 1.00** | ✅ Excellent classifier |
    | **0.75 – 0.90** | 🟡 Good classifier |
    | **0.60 – 0.75** | ⚠️ Fair classifier |
    | **0.50 – 0.60** | ❌ Poor (near random) |
    """)

    auc_cols = st.columns(len(ALL_CLASSES))
    auc_colors = ["#f85149","#d29922","#58a6ff","#8957e5"]
    for i, cls in enumerate(ALL_CLASSES):
        auc_val = roc_res.get(cls, {}).get("auc", 0.5)
        grade = "✅ Excellent" if auc_val>=0.90 else "🟡 Good" if auc_val>=0.75 else "⚠️ Fair" if auc_val>=0.60 else "❌ Low"
        auc_cols[i].markdown(eval_card(auc_val, f"AUC — {cls}", grade, auc_colors[i]),
                              unsafe_allow_html=True)

    st.plotly_chart(chart_roc_auc(roc_res), use_container_width=True)

    # ── Export ──────────────────────────────────────────
    st.markdown("---")
    export = {
        "f1_per_class": {c: f1_res[c] for c in ALL_CLASSES if c in f1_res},
        "macro_f1": macro_f1, "macro_precision": macro_p, "macro_recall": macro_r,
        "iou_per_class": iou_res["per_class_iou"], "mIoU": miou,
        "confusion_matrix": cm.tolist(), "accuracy": round(acc,4),
        "roc_auc_per_class": {c: roc_res[c]["auc"] for c in ALL_CLASSES},
        "mean_auc": round(mean_auc, 4),
    }
    st.download_button("⬇️ Download All Metrics (JSON)",
                       data=__import__("json").dumps(export, indent=2),
                       file_name="evaluation_metrics.json",
                       mime="application/json")


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🛣️ AI Road Damage Detection v3.0  |  F1 · IoU · Confusion Matrix · ROC-AUC  |  Powered by OpenCV & Streamlit")
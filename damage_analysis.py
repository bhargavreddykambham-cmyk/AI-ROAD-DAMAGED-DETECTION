"""
============================================================
AI Road Damage Detection — damage_analysis.py
Damage scoring, severity classification, and reporting
============================================================
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


# Severity thresholds & weights
SEVERITY_WEIGHTS = {"Pothole": 3.0, "Crack": 2.0, "Patch": 1.0, "Road Damage": 2.0}
SEVERITY_LEVELS  = [
    (0,    0.10, "GOOD",     "#3fb950"),
    (0.10, 0.25, "LOW",      "#58a6ff"),
    (0.25, 0.50, "MODERATE", "#d29922"),
    (0.50, 0.75, "HIGH",     "#f85149"),
    (0.75, 1.00, "CRITICAL", "#ff0000"),
]


@dataclass
class DamageReport:
    """Complete damage analysis report for one image/frame."""
    timestamp:         str   = ""
    source:            str   = ""
    total_detections:  int   = 0
    class_counts:      dict  = field(default_factory=dict)
    damage_score:      float = 0.0
    severity_level:    str   = "GOOD"
    severity_color:    str   = "#3fb950"
    damage_percentage: float = 0.0
    avg_confidence:    float = 0.0
    processing_ms:     float = 0.0
    recommendations:   List[str] = field(default_factory=list)
    image_shape:       tuple = (0,0,3)

    def to_dict(self) -> dict:
        return {k: v for k,v in self.__dict__.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class DamageAnalyzer:
    """
    Analyses detection results and computes:
    - Damage score (0-1)
    - Severity level (GOOD / LOW / MODERATE / HIGH / CRITICAL)
    - Damage percentage (area covered)
    - Maintenance recommendations
    """

    def __init__(self):
        self.history: List[DamageReport] = []

    def analyse(
        self,
        detections: List[Dict],
        image_shape: Tuple,
        source: str = "",
        processing_ms: float = 0.0
    ) -> DamageReport:
        """
        Compute full damage analysis from detection list.

        Args:
            detections : list of detection dicts from detector
            image_shape: (H, W, C) of original image
            source     : filename/label for report
            processing_ms: inference time in ms

        Returns:
            DamageReport
        """
        report = DamageReport(
            timestamp      = time.strftime("%Y-%m-%d %H:%M:%S"),
            source         = source,
            total_detections = len(detections),
            image_shape    = image_shape,
            processing_ms  = processing_ms,
        )

        if not detections:
            report.severity_level = "GOOD"
            report.severity_color = "#3fb950"
            report.recommendations = ["✅ Road surface is in good condition."]
            self.history.append(report)
            return report

        # Class counts
        for d in detections:
            lbl = d["label"]
            report.class_counts[lbl] = report.class_counts.get(lbl, 0) + 1

        # Damage score (weighted)
        total_weight = sum(
            SEVERITY_WEIGHTS.get(d["label"], 1.0) * d["conf"]
            for d in detections
        )
        max_weight = len(detections) * max(SEVERITY_WEIGHTS.values())
        report.damage_score = round(min(1.0, total_weight / max(max_weight, 1)), 4)

        # Damage area percentage
        h, w = image_shape[:2]
        img_area = h * w
        damaged_px = sum(
            (d["x2"]-d["x1"]) * (d["y2"]-d["y1"])
            for d in detections
        )
        report.damage_percentage = round(min(100.0, damaged_px/img_area*100), 2)

        # Average confidence
        report.avg_confidence = round(float(np.mean([d["conf"] for d in detections])), 4)

        # Severity level
        for lo, hi, sev, color in SEVERITY_LEVELS:
            if lo <= report.damage_score < hi or (sev == "CRITICAL" and report.damage_score >= 0.75):
                report.severity_level = sev
                report.severity_color = color
                break

        # Recommendations
        report.recommendations = self._recommendations(report)

        self.history.append(report)
        logger.info(f"Damage analysis: score={report.damage_score:.3f} "
                    f"severity={report.severity_level} pct={report.damage_percentage:.1f}%")
        return report

    def _recommendations(self, r: DamageReport) -> List[str]:
        recs = []
        counts = r.class_counts

        if counts.get("Pothole", 0) > 0:
            n = counts["Pothole"]
            if n >= 3:
                recs.append(f"🔴 URGENT: {n} potholes detected — immediate patching required")
            else:
                recs.append(f"🟠 {n} pothole(s) detected — schedule repair within 7 days")

        if counts.get("Crack", 0) > 0:
            n = counts["Crack"]
            recs.append(f"🟠 {n} crack(s) detected — apply crack sealant to prevent water ingress")

        if counts.get("Patch", 0) > 0:
            n = counts["Patch"]
            recs.append(f"🟡 {n} patched area(s) — inspect patch integrity")

        if counts.get("Road Damage", 0) > 0:
            recs.append("🟣 General road damage — schedule full surface assessment")

        if r.severity_level in ("HIGH", "CRITICAL"):
            recs.append("🚨 Road safety hazard — consider temporary speed restriction")
            recs.append("📋 Schedule full road structural assessment")

        if not recs:
            recs.append("✅ Minor damage — monitor during routine inspection")

        return recs

    def analyse_video_frames(
        self,
        frame_detections: List[List[Dict]],
        image_shape: Tuple,
        video_path: str = ""
    ) -> Dict:
        """
        Aggregate analysis across all video frames.

        Args:
            frame_detections: list of detection lists (one per frame)
            image_shape      : shape of video frames
            video_path       : for reporting

        Returns:
            Summary dictionary
        """
        all_dets_flat = [d for frame in frame_detections for d in frame]
        frame_scores  = []
        agg_counts: Dict[str,int] = {}

        for frame_dets in frame_detections:
            if frame_dets:
                tw = sum(SEVERITY_WEIGHTS.get(d["label"],1)*d["conf"] for d in frame_dets)
                mw = len(frame_dets)*max(SEVERITY_WEIGHTS.values())
                frame_scores.append(min(1.0, tw/max(mw,1)))
                for d in frame_dets:
                    agg_counts[d["label"]] = agg_counts.get(d["label"],0)+1
            else:
                frame_scores.append(0.0)

        avg_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        max_score = float(np.max(frame_scores))  if frame_scores else 0.0

        sev = "GOOD"
        for lo,hi,s,_ in SEVERITY_LEVELS:
            if lo <= avg_score < hi or (s=="CRITICAL" and avg_score>=0.75):
                sev=s; break

        summary = {
            "video_path":     video_path,
            "total_frames":   len(frame_detections),
            "total_detections": len(all_dets_flat),
            "avg_per_frame":  round(len(all_dets_flat)/max(len(frame_detections),1),2),
            "avg_damage_score": round(avg_score,4),
            "max_damage_score": round(max_score,4),
            "overall_severity": sev,
            "class_counts":   agg_counts,
            "damage_score_timeline": [round(s,3) for s in frame_scores],
        }

        # Save report
        out = Path("outputs/reports")
        out.mkdir(parents=True, exist_ok=True)
        report_file = out / f"video_report_{int(time.time())}.json"
        with open(report_file,"w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Video report saved: {report_file}")

        return summary

    def save_report(self, report: DamageReport, output_dir: str = "outputs/reports") -> str:
        """Save a DamageReport to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"report_{int(time.time())}.json")
        with open(path,"w") as f:
            f.write(report.to_json())
        return path

    def session_summary(self) -> pd.DataFrame:
        """Return DataFrame of all analyses in this session."""
        if not self.history:
            return pd.DataFrame()
        rows = [{
            "Timestamp":      r.timestamp,
            "Source":         r.source,
            "Detections":     r.total_detections,
            "Damage Score":   r.damage_score,
            "Severity":       r.severity_level,
            "Damage %":       r.damage_percentage,
            "Avg Confidence": r.avg_confidence,
        } for r in self.history]
        return pd.DataFrame(rows)


import os

if __name__ == "__main__":
    # Demo
    analyzer = DamageAnalyzer()
    fake_dets = [
        {"label":"Pothole","conf":0.87,"x1":100,"y1":300,"x2":200,"y2":380},
        {"label":"Crack",  "conf":0.72,"x1":250,"y1":400,"x2":420,"y2":420},
        {"label":"Patch",  "conf":0.65,"x1":50, "y1":500,"x2":200,"y2":560},
    ]
    r = analyzer.analyse(fake_dets, (640,640,3), source="test_road.jpg", processing_ms=45.2)
    print(r.to_json())
    print("\nRecommendations:")
    for rec in r.recommendations:
        print(" ", rec)
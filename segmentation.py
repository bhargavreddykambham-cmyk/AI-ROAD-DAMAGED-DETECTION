"""
============================================================
AI Road Damage Detection — segmentation.py
Road surface segmentation and damage zone masking
============================================================
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from loguru import logger


class RoadSegmenter:
    """
    Segments road surface from background using colour and
    geometric heuristics. Used to:
    - Focus detection on road pixels only
    - Compute damaged area as % of total road area
    - Generate damage heatmaps
    """

    def __init__(self):
        self.road_mask: Optional[np.ndarray] = None

    def segment_road(self, bgr: np.ndarray) -> np.ndarray:
        """
        Estimate road surface mask using colour + edge cues.

        Returns:
            Binary mask (255 = road, 0 = non-road)
        """
        h, w = bgr.shape[:2]

        # Strategy 1: lower 2/3 of frame is likely road
        region_mask = np.zeros((h,w), dtype=np.uint8)
        region_mask[int(h*0.30):, :] = 255

        # Strategy 2: colour-based (road is dark grey/asphalt)
        hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray_road = cv2.inRange(hsv, (0, 0, 20), (180, 50, 140))  # dark, low saturation

        # Strategy 3: remove sky (very bright or very blue regions)
        sky   = cv2.inRange(hsv, (90, 30, 180), (140, 255, 255))
        not_sky = cv2.bitwise_not(sky)

        # Combine
        road  = cv2.bitwise_and(gray_road, region_mask)
        road  = cv2.bitwise_and(road, not_sky)

        # Clean up with morphology
        ker   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        road  = cv2.morphologyEx(road, cv2.MORPH_CLOSE, ker)
        road  = cv2.morphologyEx(road, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))

        self.road_mask = road
        return road

    def overlay_mask(self, bgr: np.ndarray,
                     mask: np.ndarray,
                     color: Tuple[int,int,int] = (0,255,0),
                     alpha: float = 0.25) -> np.ndarray:
        """Overlay semi-transparent coloured mask on image."""
        overlay = bgr.copy()
        overlay[mask>0] = color
        return cv2.addWeighted(overlay, alpha, bgr, 1-alpha, 0)

    def generate_damage_heatmap(
        self,
        bgr: np.ndarray,
        detections: List[Dict],
        sigma: int = 40
    ) -> np.ndarray:
        """
        Generate a Gaussian heatmap showing damage intensity.

        Args:
            bgr       : original BGR image
            detections: list of detection dicts
            sigma     : Gaussian blur radius

        Returns:
            Heatmap overlay image (BGR)
        """
        h, w = bgr.shape[:2]
        heatmap = np.zeros((h,w), dtype=np.float32)

        weights = {"Pothole":1.0,"Crack":0.7,"Patch":0.4,"Road Damage":0.8}

        for det in detections:
            x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
            cx = (x1+x2)//2; cy = (y1+y2)//2
            w_box = x2-x1;   h_box = y2-y1

            intensity = weights.get(det["label"],0.5) * det["conf"]

            # Draw Gaussian blob
            for dy in range(-h_box//2-sigma, h_box//2+sigma):
                for dx in range(-w_box//2-sigma, w_box//2+sigma):
                    px, py = cx+dx, cy+dy
                    if 0 <= px < w and 0 <= py < h:
                        dist = (dx**2+dy**2)**0.5
                        val  = intensity * np.exp(-dist**2/(2*sigma**2))
                        heatmap[py,px] += val

        # Vectorized Gaussian alternative (faster)
        heatmap2 = np.zeros((h,w), dtype=np.float32)
        for det in detections:
            x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
            intensity = weights.get(det["label"],0.5) * det["conf"]
            heatmap2[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] += intensity

        heatmap2 = cv2.GaussianBlur(heatmap2, (0,0), sigmaX=sigma)

        # Normalise and colorize
        if heatmap2.max() > 0:
            heatmap2 = heatmap2 / heatmap2.max()

        heatmap_u8  = (heatmap2 * 255).astype(np.uint8)
        heatmap_col = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        # Blend with original
        blended = cv2.addWeighted(bgr, 0.55, heatmap_col, 0.45, 0)
        return blended

    def compute_road_damage_ratio(
        self,
        bgr: np.ndarray,
        detections: List[Dict]
    ) -> Dict[str,float]:
        """
        Compute how much of the road area is damaged.

        Returns:
            dict with road_area_px, damaged_area_px, damage_ratio
        """
        mask = self.segment_road(bgr)
        road_area  = int(np.sum(mask > 0))

        damaged = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for det in detections:
            x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
            damaged[max(0,y1):min(damaged.shape[0],y2),
                    max(0,x1):min(damaged.shape[1],x2)] = 255

        damaged_on_road = cv2.bitwise_and(damaged, mask)
        damaged_area    = int(np.sum(damaged_on_road > 0))
        ratio = damaged_area / max(road_area,1)

        return {
            "road_area_px":    road_area,
            "damaged_area_px": damaged_area,
            "damage_ratio":    round(ratio, 4),
            "damage_pct":      round(ratio*100, 2),
        }

    def lane_line_detection(self, bgr: np.ndarray) -> np.ndarray:
        """
        Detect lane markings using Hough Line Transform.

        Returns:
            Image with lane lines drawn
        """
        annotated = bgr.copy()
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5,5), 0)
        edges  = cv2.Canny(blur, 50, 150)

        # Focus on bottom half (road area)
        h,w = edges.shape
        mask = np.zeros_like(edges)
        roi  = np.array([[0,h],[0,int(h*0.55)],[w,int(h*0.55)],[w,h]])
        cv2.fillPoly(mask, [roi], 255)
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50, minLineLength=80, maxLineGap=40)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(annotated,(x1,y1),(x2,y2),(0,255,255),2)

        return annotated


if __name__ == "__main__":
    import sys
    seg = RoadSegmenter()
    img = np.random.randint(40,90,(640,640,3),dtype=np.uint8)

    mask    = seg.segment_road(img)
    overlay = seg.overlay_mask(img, mask)
    hm      = seg.generate_damage_heatmap(img,[
        {"label":"Pothole","conf":0.85,"x1":100,"y1":300,"x2":220,"y2":390},
        {"label":"Crack",  "conf":0.72,"x1":300,"y1":420,"x2":500,"y2":440},
    ])
    ratio   = seg.compute_road_damage_ratio(img,[
        {"label":"Pothole","conf":0.85,"x1":100,"y1":300,"x2":220,"y2":390},
    ])
    print("Damage ratio:", ratio)
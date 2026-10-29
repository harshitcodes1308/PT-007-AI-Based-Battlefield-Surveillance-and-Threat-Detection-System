"""Region-of-interest activity helpers for uploaded civilian video."""
import cv2


def zone_bounds(width, height, normalized=(0.30, 0.25, 0.70, 0.75)):
    a, b, c, d = normalized
    return (int(width*a), int(height*b), int(width*c), int(height*d))


def inspect_zone(detections, bounds):
    x1, y1, x2, y2 = bounds
    return [d for d in detections if x1 <= (d.x1+d.x2)/2 <= x2 and y1 <= (d.y1+d.y2)/2 <= y2]


def draw_zone(frame, bounds, active=False):
    x1, y1, x2, y2 = bounds
    color = (0, 80, 255) if active else (0, 215, 255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, 'MONITORED ZONE' + (' | ACTIVITY' if active else ''),
                (x1, max(22, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

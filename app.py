"""PT-007 Streamlit monitoring dashboard."""

from __future__ import annotations

import uuid
import hashlib
import pandas as pd
from pathlib import Path

import cv2
import streamlit as st

from detection import DetectionEngine
from event_store import EventStore
from video_processor import VideoProcessor

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
DATA_DIR = ROOT / "data"
EVENT_DATABASE = DATA_DIR / "events.sqlite"


def render_review_queue(store: EventStore) -> None:
    """Render durable history on every Streamlit run, including application startup."""
    st.subheader("Operator review queue")
    job_options = ["All video jobs", *store.job_ids()]
    selected_job = st.selectbox("Filter event history", job_options, key="event_history_job")
    job_id = None if selected_job == "All video jobs" else selected_job
    event_data = pd.DataFrame(store.list(job_id))
    if event_data.empty:
        st.info("No persisted reviewable activity events yet.")
        return
    pending_ids = event_data.loc[event_data["review_status"] == "pending", "id"].tolist()
    if pending_ids and st.button("Acknowledge all pending events", key="acknowledge_pending_events"):
        store.acknowledge([int(item) for item in pending_ids])
        # Query the durable database again now, so a rerun always reflects saved state.
        event_data = pd.DataFrame(store.list(job_id))
        st.success("Pending events acknowledged.")
    st.dataframe(event_data, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed event history",
        event_data.to_csv(index=False).encode("utf-8"),
        "pt007_event_history.csv",
        "text/csv",
        key="download_event_history",
    )
    st.caption("Events are simulated zone observations attached to temporary tracking IDs. They are not confirmed threats or identity claims.")

st.set_page_config(page_title="PT-007 | Surveillance", page_icon="◉", layout="wide")
st.markdown("""<style>
  .stApp { background: #0b1117; color: #d9e4ec; }
  [data-testid='stMetric'] { background: #121e27; border: 1px solid #243845; border-radius: 8px; padding: 12px; }
  h1 { color: #5eead4; letter-spacing: .06em; }
</style>""", unsafe_allow_html=True)

st.title("PT-007 · AI Battlefield Surveillance")
st.caption("Academic MVP · uploaded civilian aerial-video analysis. Detection confidence is not a verified threat probability.")

# Initialized once per Streamlit execution so prior history is available before upload.
store = EventStore(EVENT_DATABASE)
store.migrate_from(RUNTIME / "events.sqlite")

with st.sidebar:
    st.header("Mission Controls")
    threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    monitor_zone = st.checkbox("Enable monitored-zone activity alerts", value=True)
    zone_x = st.slider("Zone horizontal range (%)", 0, 100, (30, 70))
    zone_y = st.slider("Zone vertical range (%)", 0, 100, (25, 75))
    presence_seconds = st.slider("Prolonged presence after (seconds)", 1.0, 30.0, 5.0, 0.5)
    st.caption("Orange rectangle marks a simulated review zone. Events require operator review.")
    upload = st.file_uploader("Upload surveillance footage", type=["mp4", "mov", "avi"])
    st.info("YOLO11n uses Apple MPS when available and falls back to CPU if necessary.")

left, right = st.columns([2, 1])
with left:
    frame_slot = st.empty()
with right:
    status_slot = st.empty()
    counts_slot = st.empty()
    total_slot = st.empty()
progress = st.empty()

if upload:
    suffix = Path(upload.name).suffix.lower()
    upload_bytes = upload.getvalue()
    upload_digest = hashlib.sha256(upload_bytes).hexdigest()
    if st.session_state.get("upload_digest") != upload_digest:
        job_dir = RUNTIME / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)
        upload_path = job_dir / f"upload{suffix}"
        upload_path.write_bytes(upload_bytes)
        st.session_state.update({"upload_digest": upload_digest, "job_dir": str(job_dir), "upload_path": str(upload_path)})
    job_dir = Path(st.session_state["job_dir"])
    upload_path = Path(st.session_state["upload_path"])
    st.caption(f"Secure temporary job: `{job_dir.name}` · source name retained only in your browser")

    if st.button("Start analysis", type="primary"):
        try:
            with st.spinner("Loading YOLO11n model…"):
                engine = DetectionEngine()
                device = engine.load()
            status_slot.success(f"Model ready on {device.upper()}")
            bar = progress.progress(0, text="Preparing video…")

            def update(current, total, annotated, counts):
                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Annotated frame {current}", use_container_width=True)
                denominator = max(total, current, 1)
                bar.progress(min(current / denominator, 1.0), text=f"Processing frame {current} of {total or '?'}")
                counts_slot.markdown("**Current frame object counts**  \n" + (" · ".join(f"{name}: {count}" for name, count in sorted(counts.items())) if counts else "No supported detections"))

            zone = (zone_x[0] / 100, zone_y[0] / 100, zone_x[1] / 100, zone_y[1] / 100)
            result = VideoProcessor(engine).process(upload_path, job_dir, threshold, update, monitor_zone=monitor_zone, zone=zone, prolonged_presence_seconds=presence_seconds, event_store=store, job_id=job_dir.name)
            bar.progress(1.0, text="Analysis complete")
            total_slot.metric("Total detections", result.frame_level_detections, help="Frame-level detections; these are not unique object counts.")
            st.success(f"Processed {result.processed_frames} frames. Source FPS: {result.fps:.2f}. Processing FPS: {result.processing_fps:.2f}. Encoder: {result.codec}.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frame-level detections", result.frame_level_detections)
            m2.metric("Unique tracked objects", result.unique_tracked_objects)
            m3.metric("Reviewable activity events", result.event_count)
            m4.metric("Processing FPS", f"{result.processing_fps:.1f}")
            st.caption("Event counts by class: " + (" · ".join(f"{key}: {value}" for key, value in sorted((result.events_by_class or {}).items())) or "none"))
            st.download_button("Download event log", result.events_csv.read_bytes(), "pt007_events.csv", "text/csv")
            if not result.browser_compatible:
                st.warning("H.264 encoding was unavailable; the MP4 uses a fallback codec. Download it if your browser cannot preview it.")
            st.video(result.annotated_video.read_bytes(), format="video/mp4")
            download_a, download_b = st.columns(2)
            download_a.download_button("Download annotated video", result.annotated_video.read_bytes(), "pt007_annotated.mp4", "video/mp4")
            download_b.download_button("Download detection CSV", result.csv_file.read_bytes(), "pt007_detections.csv", "text/csv")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)
else:
    frame_slot.info("Upload MP4, MOV, or AVI footage to begin live frame annotation.")

render_review_queue(store)

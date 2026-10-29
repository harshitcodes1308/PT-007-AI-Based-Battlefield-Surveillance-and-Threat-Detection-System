# PT-007: Academic demonstration guide

PT-007 processes uploaded civilian aerial video with YOLO11n, temporary object tracking, a configurable simulated review zone, persistent SQLite event history, and exportable evidence. It does not determine hostility or make autonomous decisions.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python streamlit pandas
streamlit run app.py
```

Upload a short video and press **Start analysis**. Enable zone monitoring in the sidebar. The orange rectangle represents a simulated test zone. A deduplicated `zone_entry` event is recorded when a tracked object enters the zone. A `prolonged_presence` event is recorded when it stays longer than the selected threshold. Events remain available after Streamlit restarts and can be acknowledged by the operator.

## Limitations

* Tracking IDs are temporary associations and may change after occlusion.
* Events can be noisy when detections flicker. They are prompts for human review only.
* Video processing FPS is measured wall-clock throughput and differs from source FPS.
* No ground-truth accuracy claim is made by the demonstration.

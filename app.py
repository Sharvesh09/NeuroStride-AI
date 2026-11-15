"""NeuroStride AI dashboard. Research prototype - NOT a medical device."""
import time
import cv2
import numpy as np
import streamlit as st
import config as C
import feature_extraction as F
from inference import load_model, predict
from utils.gait_analysis import SignalBuffer, AlertManager
from utils.visualization import gait_plot

st.set_page_config(page_title="NeuroStride AI", page_icon="🧠", layout="wide")
st.markdown("""<style>
.stApp{background:#0B1F3A;color:#fff} h1,h2,h3{color:#00C8A6}
[data-testid=stMetricValue]{color:#00C8A6} section[data-testid=stSidebar]{background:#081529}
.ok{background:#0f3d3a;border-left:6px solid #00C8A6;padding:12px;border-radius:6px}
.warn{background:#4a1822;border-left:6px solid #E5484D;padding:12px;border-radius:6px}
.na{background:#16294a;border-left:6px solid #7f8fa9;padding:12px;border-radius:6px}</style>""", unsafe_allow_html=True)
st.title("🧠 NeuroStride AI")
st.caption("Research prototype for freeze-like gait pattern screening. Not a clinically validated or diagnostic device.")

S = st.session_state
S.setdefault("alerts", AlertManager()); S.setdefault("log", []); S.setdefault("total_s", 0.0)
bundle = load_model()
with st.sidebar:
    st.header("Controls")
    cam = st.number_input("Camera index", 0, 5, 0)
    run = st.toggle("▶ Start camera", value=False)
    if st.button("Reset session"):
        S.alerts, S.log, S.total_s = AlertManager(), [], 0.0
    if bundle is None:
        st.warning("No trained model found (models/pose_model.pkl). Pose tracking and gait metrics run, but "
                   "FoG classification is DISABLED. Train first (see README).")
    else:
        m = bundle["metrics"]
        st.success("Model loaded")
        st.caption(f"Validation: {m['validation']}; F1 (FoG-like) = {m['f1_fog']:.2f} on {m['n_groups']} groups")
    st.caption("Confidence shown only if the model is calibrated.")

c1, c2 = st.columns([3, 2])
vid, status = c1.empty(), c2.empty()
mcols = c2.columns(3); m_step, m_var, m_ev = [c.empty() for c in mcols]
m_dur = c2.empty(); plot_ph = c2.empty()
hist = st.container()


def render_history():
    with hist:
        st.subheader("Session summary")
        st.write(f"Duration this session: **{S.total_s:.0f} s** | Suspected freeze-like events: **{len(S.alerts.events)}**")
        if S.alerts.events:
            st.dataframe(S.alerts.events, use_container_width=True)
        else:
            st.caption("No suspected events recorded.")
        st.caption("Events are model-flagged movement patterns, not confirmed clinical FoG.")


if run:
    try:
        from utils.pose_detection import PoseExtractor
        cap = cv2.VideoCapture(int(cam), cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(int(cam))
        if not cap.isOpened():
            st.error("Cannot open the webcam. Close other apps using it, or change the camera index.")
            st.stop()
        pe, buf, t0, last_eval, last_plot = PoseExtractor(), SignalBuffer(), time.time(), 0.0, 0.0
        base = S.total_s; step_f = var = 0.0; label, prob = None, None
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Camera stopped returning frames."); break
            frame, vec = pe.process(frame)
            now = time.time() - t0
            S.total_s = base + now
            if vec is not None:
                buf.add(now, vec)
            vid.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            if now - last_eval >= C.POSE_STEP_S:
                last_eval = now
                win = buf.window()
                if win is None:
                    status.markdown('<div class="na">Collecting data… keep your whole body (hips to feet) in view.</div>', unsafe_allow_html=True)
                else:
                    feats = F.pose_window_features(win); names = F.pose_feature_names()
                    step_f = feats[names.index("step_freq_hz")]; var = feats[names.index("knee_ang_std_mean")]
                    if bundle is None:
                        status.markdown('<div class="na">Tracking only - no model loaded, classification disabled.</div>', unsafe_allow_html=True)
                    else:
                        label, prob = predict(bundle, feats)
                        S.alerts.update(now + base, label == 1)
                        if S.alerts.active:
                            status.markdown('<div class="warn">⚠ <b>Possible freezing-like movement</b> (sustained over several windows). Not a confirmed clinical event.</div>', unsafe_allow_html=True)
                        else:
                            status.markdown('<div class="ok">✔ <b>Normal walking pattern</b></div>', unsafe_allow_html=True)
                        if prob is not None:
                            status.caption(f"Calibrated probability: {prob:.0%}")
                    if now - last_plot > 2:
                        last_plot = now
                        fig = gait_plot(win, C.POSE_FS, S.alerts.active); plot_ph.pyplot(fig); 
                        import matplotlib.pyplot as plt; plt.close(fig)
                m_step.metric("Step freq (Hz)", f"{step_f:.2f}"); m_var.metric("Movement var.", f"{var:.1f}")
                m_ev.metric("Suspected events", len(S.alerts.events)); m_dur.metric("Session (s)", f"{S.total_s:.0f}")
    except Exception as e:
        st.error(f"Runtime error: {e}")
    finally:
        try:
            cap.release(); pe.close()
        except Exception:
            pass
else:
    status.markdown('<div class="na">Camera off. Toggle “Start camera” in the sidebar.</div>', unsafe_allow_html=True)
render_history()

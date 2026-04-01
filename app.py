from __future__ import annotations

import streamlit as st

from ftd.detector import ObjectDetector
from ftd.difficulty import DifficultyLevel, resolve_config, sample_params
from ftd.modifier import ChangeType
from ftd.pipeline import PuzzleGenerator
from ftd.utils import decode_uploaded_image, encode_image_to_png, resize_to_max
from ftd.video import export_puzzle_video_bytes


st.set_page_config(page_title="Find the Difference - Puzzle Generator", page_icon="🧩", layout="wide")


@st.cache_resource(show_spinner="Loading SAM model (first time takes ~30s)...")
def load_detector() -> ObjectDetector:
    return ObjectDetector()


@st.cache_resource
def get_generator(_detector: ObjectDetector) -> PuzzleGenerator:
    return PuzzleGenerator(_detector)


for key, value in {
    "uploaded_image": None,
    "detected_objects": None,
    "puzzle_result": None,
    "prev_file_id": None,
    "video_bytes": None,
    "video_signature": None,
}.items():
    st.session_state.setdefault(key, value)


st.sidebar.title("🧩 Puzzle Settings")
mode = st.sidebar.radio("Mode", ["Auto", "Manual"], horizontal=True)
difficulty_label = st.sidebar.select_slider(
    "Difficulty",
    options=["Easy", "Medium", "Hard"],
    value="Medium",
)

difficulty_map = {
    "Easy": DifficultyLevel.EASY,
    "Medium": DifficultyLevel.MEDIUM,
    "Hard": DifficultyLevel.HARD,
}
difficulty = difficulty_map[difficulty_label]
video_duration = st.sidebar.number_input(
    "Video duration (seconds)",
    min_value=5,
    max_value=600,
    value=90,
    step=5,
)

uploaded_file = st.file_uploader("Upload a cartoon image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if file_id != st.session_state.prev_file_id:
    st.session_state.detected_objects = None
    st.session_state.puzzle_result = None
    st.session_state.video_bytes = None
    st.session_state.video_signature = None
    st.session_state.prev_file_id = file_id
    st.session_state.uploaded_image = resize_to_max(decode_uploaded_image(uploaded_file.getvalue()), 1024)
elif st.session_state.uploaded_image is None:
    st.session_state.uploaded_image = resize_to_max(decode_uploaded_image(uploaded_file.getvalue()), 1024)

image = st.session_state.uploaded_image
st.image(image, width=600)
if max(image.shape[:2]) < 256:
    st.warning("Image is very small. Results may be less reliable.")

generate_btn = st.sidebar.button("🎲 Generate Puzzle", type="primary", disabled=image is None)

detector = None
generator = None
selections: list[tuple] = []

if mode == "Manual":
    try:
        detector = load_detector()
        generator = get_generator(detector)
        if st.session_state.detected_objects is None:
            with st.spinner("Analyzing image - detecting objects with SAM..."):
                st.session_state.detected_objects = generator.detect_objects(image)
        objects = st.session_state.detected_objects
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load SAM detector: {exc}")
        st.stop()

    if not objects:
        st.error("No objects were detected in this image. Try a clearer cartoon scene.")
        st.stop()

    st.subheader("Detected Objects")
    columns = st.columns(4)
    for index, obj in enumerate(objects):
        column = columns[index % 4]
        with column:
            thumb = detector.get_thumbnail(image, obj)
            st.image(thumb, use_container_width=True)
            selected = st.checkbox(f"Object {obj.id}", key=f"sel_{obj.id}")
            change_value = st.selectbox(
                "Change",
                [ct.value for ct in ChangeType],
                key=f"change_{obj.id}",
                label_visibility="collapsed",
            )
            if selected:
                selections.append((obj, ChangeType(change_value)))

    st.info(f"**{len(selections)} change(s) selected**")

if generate_btn:
    try:
        detector = detector or load_detector()
        generator = generator or get_generator(detector)
        with st.spinner("Generating puzzle..."):
            if mode == "Auto":
                result = generator.auto_generate(image, difficulty)
            else:
                config = resolve_config(difficulty)
                manual_selections = [
                    (obj, change_type, sample_params(config, change_type))
                    for obj, change_type in selections
                ]
                result = generator.manual_generate(image, manual_selections)
        st.session_state.puzzle_result = result
        st.session_state.video_bytes = None
        st.session_state.video_signature = None
    except ValueError as exc:
        st.error(str(exc))
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")

result = st.session_state.puzzle_result

if result is not None:
    video_signature = (id(result), int(video_duration))
    if st.session_state.video_signature != video_signature or st.session_state.video_bytes is None:
        with st.spinner("Preparing video..."):
            st.session_state.video_bytes = export_puzzle_video_bytes(
                original=result.original,
                modified=result.modified,
                puzzle_duration=float(video_duration),
            )
            st.session_state.video_signature = video_signature

    st.sidebar.download_button(
        "📥 Download Puzzle",
        data=encode_image_to_png(result.puzzle_image),
        file_name="puzzle.png",
        mime="image/png",
    )
    st.sidebar.download_button(
        "📥 Download Solution",
        data=encode_image_to_png(result.solution_image),
        file_name="solution.png",
        mime="image/png",
    )
    st.sidebar.download_button(
        "🎬 Download Video",
        data=st.session_state.video_bytes,
        file_name="puzzle_video.mp4",
        mime="video/mp4",
    )

    st.divider()
    st.subheader("Puzzle")
    st.caption(f"{len(result.changes)} difference(s)")
    st.image(result.puzzle_image, use_container_width=True)

    with st.expander("🔍 Show Solution"):
        st.image(result.solution_image, use_container_width=True)

    with st.expander("📋 Change Details"):
        for idx, change in enumerate(result.changes, start=1):
            st.write(
                f"{idx}. Object {change.object_id} - {change.change_type.value} - params: {change.params}"
            )

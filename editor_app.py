from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ftd.detector import DetectedObject, ObjectDetector
from ftd.editor_session import EditorSession
from ftd.modifier import ChangeType
from ftd.utils import load_image, resize_to_max, save_image
from ftd.video import save_puzzle_video


def rgb_to_qimage(image: np.ndarray) -> QImage:
    height, width, channels = image.shape
    bytes_per_line = channels * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


def image_to_pixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(rgb_to_qimage(image))


class ObjectOverlayItem(QGraphicsPathItem):
    def __init__(self, obj: DetectedObject, path: QPainterPath) -> None:
        super().__init__(path)
        self.object_id = obj.id
        self.default_pen = QPen(QColor(0, 180, 255, 220), 2)
        self.selected_pen = QPen(QColor(255, 120, 0, 255), 3)
        self.setPen(self.default_pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(2)

    def set_selected_style(self, selected: bool) -> None:
        self.setPen(self.selected_pen if selected else self.default_pen)


class CanvasView(QGraphicsView):
    object_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: dict[int, ObjectOverlayItem] = {}
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor(28, 28, 30))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, image: np.ndarray) -> None:
        scene = self.scene()
        scene.clear()
        self._overlay_items.clear()
        self._pixmap_item = scene.addPixmap(image_to_pixmap(image))
        scene.setSceneRect(self._pixmap_item.boundingRect())

    def set_overlays(self, objects: list[DetectedObject]) -> None:
        self._overlay_items.clear()
        if self._pixmap_item is None:
            return

        for obj in objects:
            path = self._mask_to_path(obj.mask)
            item = ObjectOverlayItem(obj, path)
            self.scene().addItem(item)
            self._overlay_items[obj.id] = item

    def select_object(self, object_id: int | None) -> None:
        for obj_id, item in self._overlay_items.items():
            item.set_selected_style(obj_id == object_id)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self.scene().itemAt(scene_pos, self.transform())
            if isinstance(item, ObjectOverlayItem):
                self.object_clicked.emit(item.object_id)
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        scale_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(scale_factor, scale_factor)

    @staticmethod
    def _mask_to_path(mask: np.ndarray) -> QPainterPath:
        mask_uint8 = (mask.astype(np.uint8) * 255).copy()
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for contour in contours:
            if len(contour) == 0:
                continue
            first = contour[0][0]
            path.moveTo(QPointF(float(first[0]), float(first[1])))
            for point in contour[1:]:
                x, y = point[0]
                path.lineTo(QPointF(float(x), float(y)))
            path.closeSubpath()
        return path


class EditorMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Find the Difference - Manual Editor")
        self.resize(1400, 900)

        self.detector: ObjectDetector | None = None
        self.current_image: np.ndarray | None = None
        self.detected_objects: list[DetectedObject] = []
        self.session: EditorSession | None = None
        self.selected_object_id: int | None = None

        self.canvas = CanvasView()
        self.canvas.object_clicked.connect(self.on_object_selected)

        self.status_label = QLabel("Load a cartoon image to begin.")
        self.selected_label = QLabel("Selected object: none")
        self.change_combo = QComboBox()
        self.change_combo.addItems([ct.value for ct in ChangeType])

        self.hue_slider = self._build_slider(10, 90, 30)
        self.saturation_slider = self._build_slider(80, 120, 100)
        self.shift_x_slider = self._build_slider(-30, 30, 8)
        self.shift_y_slider = self._build_slider(-30, 30, 8)
        self.resize_slider = self._build_slider(60, 120, 85)
        self.flip_axis_combo = QComboBox()
        self.flip_axis_combo.addItems(["horizontal", "vertical"])
        self.video_duration_spin = QSpinBox()
        self.video_duration_spin.setRange(5, 600)
        self.video_duration_spin.setValue(90)
        self.video_duration_spin.setSuffix(" sec")

        self.detect_button = QPushButton("Detect Objects")
        self.detect_button.clicked.connect(self.detect_objects)
        self.apply_button = QPushButton("Apply Change")
        self.apply_button.clicked.connect(self.apply_selected_change)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_last_change)
        self.clear_button = QPushButton("Clear Edits")
        self.clear_button.clicked.connect(self.clear_edits)

        self._build_ui()
        self._build_actions()
        self._update_controls()

    def _build_ui(self) -> None:
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        form = QFormLayout()
        form.addRow("Change type", self.change_combo)
        form.addRow("Hue shift", self.hue_slider)
        form.addRow("Saturation %", self.saturation_slider)
        form.addRow("Shift X", self.shift_x_slider)
        form.addRow("Shift Y", self.shift_y_slider)
        form.addRow("Resize %", self.resize_slider)
        form.addRow("Flip axis", self.flip_axis_combo)
        form.addRow("Video duration", self.video_duration_spin)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.selected_label)
        right_layout.addWidget(self.detect_button)
        right_layout.addLayout(form)
        right_layout.addWidget(self.apply_button)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.undo_button)
        buttons_row.addWidget(self.clear_button)
        right_layout.addLayout(buttons_row)
        right_layout.addStretch()

        container = QWidget()
        layout = QHBoxLayout(container)
        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1100, 300])
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        edit_menu = self.menuBar().addMenu("Edit")

        open_action = QAction("Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_modified_action = QAction("Save Edited Image...", self)
        save_modified_action.setShortcut(QKeySequence.StandardKey.Save)
        save_modified_action.triggered.connect(self.save_modified_image)
        file_menu.addAction(save_modified_action)

        save_puzzle_action = QAction("Save Puzzle Image...", self)
        save_puzzle_action.triggered.connect(self.save_puzzle_image)
        file_menu.addAction(save_puzzle_action)

        save_solution_action = QAction("Save Solution Image...", self)
        save_solution_action.triggered.connect(self.save_solution_image)
        file_menu.addAction(save_solution_action)

        save_video_action = QAction("Save Puzzle Video...", self)
        save_video_action.triggered.connect(self.save_video)
        file_menu.addAction(save_video_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_last_change)
        edit_menu.addAction(undo_action)

        detect_action = QAction("Detect Objects", self)
        detect_action.triggered.connect(self.detect_objects)
        edit_menu.addAction(detect_action)

    def _build_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        return slider

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return

        image = resize_to_max(load_image(path), 1024)
        self.current_image = image
        self.detected_objects = []
        self.session = None
        self.selected_object_id = None
        self.canvas.resetTransform()
        self.canvas.set_image(image)
        self.status_label.setText(f"Loaded image: {Path(path).name}")
        self.selected_label.setText("Selected object: none")
        self._update_controls()

    def detect_objects(self) -> None:
        if self.current_image is None:
            self._show_info("Load an image first.")
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            if self.detector is None:
                self.detector = ObjectDetector()
            self.detected_objects = self.detector.detect(self.current_image)
        except Exception as exc:
            self._show_error(f"Object detection failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not self.detected_objects:
            self._show_info("No objects detected in this image.")
            return

        self.session = EditorSession(self.current_image, self.detected_objects)
        self.canvas.set_image(self.current_image)
        self.canvas.set_overlays(self.detected_objects)
        self.status_label.setText(f"Detected {len(self.detected_objects)} objects. Click one to edit it.")
        self.selected_object_id = None
        self.selected_label.setText("Selected object: none")
        self._update_controls()

    def on_object_selected(self, object_id: int) -> None:
        self.selected_object_id = object_id
        self.canvas.select_object(object_id)
        self.selected_label.setText(f"Selected object: {object_id}")
        self._update_controls()

    def apply_selected_change(self) -> None:
        if self.session is None or self.selected_object_id is None:
            self._show_info("Detect objects and select one before applying a change.")
            return

        change_type = ChangeType(self.change_combo.currentText())
        params = self._current_params(change_type)

        try:
            snapshot = self.session.apply_edit(self.selected_object_id, change_type, params)
        except Exception as exc:
            self._show_error(f"Could not apply change: {exc}")
            return

        self.canvas.set_image(snapshot.modified)
        self.canvas.set_overlays(self.detected_objects)
        self.canvas.select_object(self.selected_object_id)
        self.status_label.setText(
            f"Applied {change_type.value} to object {self.selected_object_id}. "
            f"Total edits: {len(snapshot.changes)}"
        )
        self._update_controls()

    def undo_last_change(self) -> None:
        if self.session is None or not self.session.can_undo():
            return

        snapshot = self.session.undo()
        self.canvas.set_image(snapshot.modified)
        self.canvas.set_overlays(self.detected_objects)
        self.canvas.select_object(self.selected_object_id)
        self.status_label.setText(f"Undo complete. Total edits: {len(snapshot.changes)}")
        self._update_controls()

    def clear_edits(self) -> None:
        if self.session is None:
            return

        snapshot = self.session.clear()
        self.canvas.set_image(snapshot.modified)
        self.canvas.set_overlays(self.detected_objects)
        self.canvas.select_object(self.selected_object_id)
        self.status_label.setText("All edits cleared.")
        self._update_controls()

    def save_modified_image(self) -> None:
        if self.session is None:
            self._show_info("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Edited Image", "edited.png", "PNG Image (*.png)")
        if path:
            save_image(self.session.modified_image, path)
            self.status_label.setText(f"Saved edited image to {Path(path).name}")

    def save_puzzle_image(self) -> None:
        if self.session is None:
            self._show_info("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Puzzle Image", "puzzle.png", "PNG Image (*.png)")
        if path:
            save_image(self.session.build_puzzle_image(), path)
            self.status_label.setText(f"Saved puzzle image to {Path(path).name}")

    def save_solution_image(self) -> None:
        if self.session is None:
            self._show_info("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Solution Image", "solution.png", "PNG Image (*.png)")
        if path:
            save_image(self.session.build_solution_image(), path)
            self.status_label.setText(f"Saved solution image to {Path(path).name}")

    def save_video(self) -> None:
        if self.session is None:
            self._show_info("Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Puzzle Video", "puzzle_video.mp4", "MP4 Video (*.mp4)")
        if path:
            save_puzzle_video(
                original=self.session.original,
                modified=self.session.modified_image,
                output_path=path,
                puzzle_duration=float(self.video_duration_spin.value()),
            )
            self.status_label.setText(f"Saved puzzle video to {Path(path).name}")

    def _current_params(self, change_type: ChangeType) -> dict:
        if change_type == ChangeType.RECOLOR:
            return {
                "hue_shift": int(self.hue_slider.value()),
                "saturation_scale": float(self.saturation_slider.value()) / 100.0,
            }
        if change_type == ChangeType.REMOVE:
            return {}
        if change_type == ChangeType.SHIFT:
            return {
                "dx": int(self.shift_x_slider.value()),
                "dy": int(self.shift_y_slider.value()),
            }
        if change_type == ChangeType.RESIZE:
            return {"scale": float(self.resize_slider.value()) / 100.0}
        if change_type == ChangeType.FLIP:
            return {"axis": 1 if self.flip_axis_combo.currentText() == "horizontal" else 0}
        raise ValueError(f"Unsupported change type: {change_type}")

    def _update_controls(self) -> None:
        has_image = self.current_image is not None
        has_session = self.session is not None
        has_selection = self.selected_object_id is not None

        self.detect_button.setEnabled(has_image)
        self.apply_button.setEnabled(has_session and has_selection)
        self.undo_button.setEnabled(has_session and self.session.can_undo())
        self.clear_button.setEnabled(has_session and self.session.can_undo())

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "Editor", message)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Editor", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = EditorMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

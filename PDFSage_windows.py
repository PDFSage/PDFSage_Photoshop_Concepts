#!/usr/bin/env python3
"""
PDFSage – a minimal yet extensible Python GUI image-editor that demonstrates
advanced photo-editing features with a PDF export option.

New in this version
-------------------
* **Windows OCR** – uses the built-in Windows.Media.Ocr engine  
  (via the *winsdk* package) instead of Tesseract; recognised text can be
  edited and optionally burned back onto the picture.
* **True-size display & zoom** – images load at 100 % (1-to-1 pixels) and you
  can zoom in/out with the toolbar, shortcuts or Ctrl+mouse-wheel.
* Miscellaneous UI polish.

Extra runtime dependencies
--------------------------
winsdk  – Python bindings for Windows SDK  (pip install winsdk)
Pillow  – image handling
opencv-python – Gaussian blur
reportlab – PDF export
PyQt5  – GUI
"""

import sys
import os
import base64
import asyncio
from pathlib import Path

from PyQt5 import QtWidgets, QtGui, QtCore           # GUI widgets & events
from PIL import Image, ImageQt, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import cv2                                           # extra filters
from reportlab.pdfgen import canvas                  # PDF export

# --- Windows OCR ----------------------------------------------------------- #

try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import (
        SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
    )
    from winsdk.windows.security.cryptography import CryptographicBuffer
except ImportError:                                 # pragma: no cover
    OcrEngine = None                                # handled at runtime


def _ibuffer(data: bytes):
    """Create WinRT IBuffer from *data*."""
    return CryptographicBuffer.decode_from_base64_string(
        base64.b64encode(data).decode("ascii")
    )


def _swbmp_from_pil(img: Image.Image) -> SoftwareBitmap:
    """Convert a PIL image to WinRT SoftwareBitmap (RGBA8, straight alpha)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = _ibuffer(img.tobytes())
    return SoftwareBitmap.create_copy_from_buffer(
        buf, BitmapPixelFormat.RGBA8, img.width, img.height,
        BitmapAlphaMode.STRAIGHT
    )


def run_windows_ocr(img: Image.Image, lang_tag: str = "en") -> str:
    """Return plain text recognised by Windows.Media.Ocr."""
    if OcrEngine is None:
        raise RuntimeError("Windows OCR requires ‘winsdk’ (pip install winsdk).")

    lang = Language(lang_tag)
    if not OcrEngine.is_language_supported(lang):
        raise RuntimeError(f"Language “{lang_tag}” is not installed.")

    engine = OcrEngine.try_create_from_language(lang)
    sbmp = _swbmp_from_pil(img)

    async def _do_ocr():
        res = await engine.recognize_async(sbmp)
        return res.text

    return asyncio.run(_do_ocr())


# ---------------------------- Utility helpers ----------------------------- #

def pil_to_qimage(pil_img: Image.Image) -> QtGui.QImage:
    """Convert :pyclass:`PIL.Image` to :pyclass:`QtGui.QImage`."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    return QtGui.QImage(data, pil_img.width, pil_img.height,
                        QtGui.QImage.Format_RGBA8888)


def qimage_to_pil(qimg: QtGui.QImage) -> Image.Image:
    """Convert :pyclass:`QtGui.QImage` back to :pyclass:`PIL.Image`."""
    qimg = qimg.convertToFormat(QtGui.QImage.Format_RGBA8888)
    ptr = qimg.bits()
    ptr.setsize(qimg.width() * qimg.height() * 4)
    return Image.frombuffer("RGBA", (qimg.width(), qimg.height()),
                            bytes(ptr), "raw", "RGBA", 0, 1)


# ------------------------ Custom label with selection ---------------------- #

class ImageLabel(QtWidgets.QLabel):
    """
    QLabel that exposes a rubber-band selection rectangle.  The selection
    rectangle is expressed in *label* coordinates – the caller must convert to
    image coordinates if the pixmap is scaled.
    """

    selection_changed = QtCore.pyqtSignal()

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._band: QtWidgets.QRubberBand | None = None
        self._origin: QtCore.QPoint | None = None
        self._rect: QtCore.QRect | None = None
        self.setMouseTracking(True)

    # --- API ----------------------------------------------------------------

    def selection_rect(self) -> QtCore.QRect | None:
        """Return the current selection rect (label coords) or *None*."""
        return QtCore.QRect(self._rect) if self._rect else None

    def clear_selection(self) -> None:
        """Hide rubber-band & forget rectangle."""
        if self._band and self._band.isVisible():
            self._band.hide()
        self._rect = None
        self.selection_changed.emit()

    # --- mouse events -------------------------------------------------------

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos()
            if not self._band:
                self._band = QtWidgets.QRubberBand(
                    QtWidgets.QRubberBand.Rectangle, self
                )
            self._band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._band.show()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._band and self._origin:
            self._band.setGeometry(
                QtCore.QRect(self._origin, ev.pos()).normalized()
            )
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton and self._band:
            self._rect = self._band.geometry()
            self.selection_changed.emit()
        super().mouseReleaseEvent(ev)


# ------------------------------- Main Window ------------------------------ #

class ImageEditor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDFSage Photo Editor")
        self.resize(1100, 800)

        # Central area: a scrollable, selectable QLabel.
        self.label = ImageLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setBackgroundRole(QtGui.QPalette.Base)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                 QtWidgets.QSizePolicy.Ignored)
        self.label.setScaledContents(False)
        self.label.selection_changed.connect(self._on_selection_change)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.label)
        self.setCentralWidget(self.scroll)

        # state
        self.image: Image.Image | None = None
        self.undo_stack: list[Image.Image] = []      # undo history
        self.redo_stack: list[Image.Image] = []
        self._selection_rect: QtCore.QRect | None = None
        self.scale_factor: float = 1.0               # current zoom

        # build GUI chrome
        self._create_actions()
        self._create_menus()
        self._create_toolbar()

        # wheel-zoom
        self.installEventFilter(self)

    # -------------------- GUI construction helpers -------------------- #

    def _create_actions(self):
        # File
        self.open_act = QtWidgets.QAction("Open…", self, shortcut="Ctrl+O",
                                          triggered=self.open_image)
        self.save_act = QtWidgets.QAction("Save As…", self, shortcut="Ctrl+S",
                                          triggered=self.save_image)
        self.export_pdf_act = QtWidgets.QAction("Export PDF…", self,
                                               triggered=self.export_pdf)
        self.exit_act = QtWidgets.QAction("Quit", self, shortcut="Ctrl+Q",
                                          triggered=self.close)

        # Edit
        self.undo_act = QtWidgets.QAction("Undo", self, shortcut="Ctrl+Z",
                                          triggered=self.undo)
        self.redo_act = QtWidgets.QAction("Redo", self, shortcut="Ctrl+Y",
                                          triggered=self.redo)

        # View – zoom
        self.zoom_in_act = QtWidgets.QAction("Zoom In", self,
                                             shortcut="Ctrl++",
                                             triggered=lambda: self.zoom(1.25))
        self.zoom_out_act = QtWidgets.QAction("Zoom Out", self,
                                              shortcut="Ctrl+-",
                                              triggered=lambda: self.zoom(0.8))
        self.zoom_reset_act = QtWidgets.QAction("Actual Size", self,
                                                shortcut="Ctrl+0",
                                                triggered=lambda: self.set_zoom(1))

        # Image operations
        self.rotate_left_act  = QtWidgets.QAction("Rotate ⟲", self,
                                   triggered=lambda: self.rotate(-90))
        self.rotate_right_act = QtWidgets.QAction("Rotate ⟳", self,
                                   triggered=lambda: self.rotate(90))
        self.flip_h_act = QtWidgets.QAction("Flip Horiz", self,
                                            triggered=lambda: self.flip(True))
        self.flip_v_act = QtWidgets.QAction("Flip Vert", self,
                                            triggered=lambda: self.flip(False))
        self.resize_act = QtWidgets.QAction("Resize…", self,
                                            triggered=self.resize_image)
        self.crop_act   = QtWidgets.QAction("Crop Selection", self,
                                            triggered=self.crop)

        # Adjustments
        self.brightness_act = QtWidgets.QAction("Brightness…", self,
                               triggered=lambda: self.adjust("brightness"))
        self.contrast_act   = QtWidgets.QAction("Contrast…", self,
                               triggered=lambda: self.adjust("contrast"))

        # Filters
        self.blur_act      = QtWidgets.QAction("Blur", self, triggered=self.blur)
        self.sharpen_act   = QtWidgets.QAction("Sharpen", self,
                                               triggered=self.sharpen)
        self.grayscale_act = QtWidgets.QAction("Grayscale", self,
                                               triggered=self.grayscale)
        self.edge_act      = QtWidgets.QAction("Edge Detect", self,
                                               triggered=self.edge_detect)

        # NEW – OCR & insertion
        self.ocr_act = QtWidgets.QAction("OCR (Edit)…", self, shortcut="Ctrl+T",
                                         triggered=self.ocr_edit)
        self.insert_act = QtWidgets.QAction("Insert Image…", self,
                                            triggered=self.insert_image)

    def _create_menus(self):
        mb = self.menuBar()

        file_m = mb.addMenu("&File")
        file_m.addActions((self.open_act, self.save_act, self.export_pdf_act))
        file_m.addSeparator()
        file_m.addAction(self.exit_act)

        edit_m = mb.addMenu("&Edit")
        edit_m.addActions((self.undo_act, self.redo_act))

        view_m = mb.addMenu("&View")
        view_m.addActions((self.zoom_in_act, self.zoom_out_act,
                           self.zoom_reset_act))

        img_m = mb.addMenu("&Image")
        img_m.addActions((self.rotate_left_act, self.rotate_right_act,
                          self.flip_h_act, self.flip_v_act,
                          self.resize_act, self.crop_act, self.insert_act))

        adj_m = mb.addMenu("&Adjust")
        adj_m.addActions((self.brightness_act, self.contrast_act))

        filt_m = mb.addMenu("&Filters")
        filt_m.addActions((self.blur_act, self.sharpen_act,
                           self.grayscale_act, self.edge_act))

        tools_m = mb.addMenu("&Tools")
        tools_m.addAction(self.ocr_act)

    def _create_toolbar(self):
        tb = QtWidgets.QToolBar("Main", self)
        for act in (self.open_act, self.save_act,
                    self.undo_act, self.redo_act,
                    self.rotate_left_act, self.rotate_right_act,
                    self.crop_act, self.insert_act,
                    self.brightness_act, self.contrast_act,
                    self.zoom_in_act, self.zoom_out_act, self.zoom_reset_act,
                    self.ocr_act):
            tb.addAction(act)
        self.addToolBar(tb)

    # ------------------------- File operations ------------------------- #

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif);;All files (*)")
        if path:
            self.image = Image.open(path)
            self.scale_factor = 1.0
            self._push_undo()
            self.label.clear_selection()
            self._update_label()

    def save_image(self):
        if not self.image:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", str(Path.home() / "untitled.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All files (*)")
        if path:
            self.image.save(path)
            self.statusBar().showMessage(f"Saved to {path}", 2000)

    def export_pdf(self):
        if not self.image:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", str(Path.home() / "output.pdf"),
            "PDF Files (*.pdf);;All files (*)")
        if not path:
            return
        c = canvas.Canvas(path)
        tmp = Path(path).with_suffix(".png")
        self.image.save(tmp)
        c.drawImage(str(tmp), 0, 0, width=self.image.width,
                    height=self.image.height)
        c.showPage()
        c.save()
        tmp.unlink(missing_ok=True)
        self.statusBar().showMessage(f"Exported PDF {path}", 2000)

    # ------------------------ Undo / Redo logic ------------------------ #

    def _push_undo(self):
        if self.image:
            self.undo_stack.append(self.image.copy())
            self.redo_stack.clear()
            self.undo_act.setEnabled(True)

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.image = self.undo_stack[-1].copy()
            self._update_label()
            self.redo_act.setEnabled(True)
        if len(self.undo_stack) <= 1:
            self.undo_act.setEnabled(False)

    def redo(self):
        if self.redo_stack:
            self.image = self.redo_stack.pop()
            self.undo_stack.append(self.image.copy())
            self._update_label()
            self.undo_act.setEnabled(True)
        if not self.redo_stack:
            self.redo_act.setEnabled(False)

    # ---------------------------- Zooming ----------------------------- #

    def zoom(self, factor: float):
        if self.image:
            self.scale_factor = max(0.1, min(8.0, self.scale_factor * factor))
            self._update_label()

    def set_zoom(self, factor: float):
        if self.image:
            self.scale_factor = factor
            self._update_label()

    def eventFilter(self, obj, ev):
        if (ev.type() == QtCore.QEvent.Wheel and
                ev.modifiers() & QtCore.Qt.ControlModifier):
            self.zoom(1.1 if ev.angleDelta().y() > 0 else 0.9)
            return True
        return super().eventFilter(obj, ev)   # <-- fixed: pass obj as well

    # -------------------- Basic image transforms -------------------- #

    def rotate(self, angle: int):
        if self.image:
            self._push_undo()
            self.image = self.image.rotate(angle, expand=True)
            self._update_label()

    def flip(self, horizontal: bool):
        if self.image:
            self._push_undo()
            self.image = self.image.transpose(
                Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM)
            self._update_label()

    def resize_image(self):
        if not self.image:
            return
        w, ok = QtWidgets.QInputDialog.getInt(self, "Resize", "Width:",
                                              value=self.image.width)
        if not ok:
            return
        h, ok = QtWidgets.QInputDialog.getInt(self, "Resize", "Height:",
                                              value=self.image.height)
        if ok:
            self._push_undo()
            self.image = self.image.resize((w, h), Image.LANCZOS)
            self._update_label()

    def crop(self):
        if not self.image:
            return

        rect = self._selection_rect
        if rect is None or rect.width() < 2 or rect.height() < 2:
            # fall back to centre crop
            w, h = self.image.size
            rect_img = (w // 4, h // 4, w * 3 // 4, h * 3 // 4)
        else:
            # convert label coords → image coords (consider scaling)
            pix = self.label.pixmap()
            scale_x = self.image.width / pix.width()
            scale_y = self.image.height / pix.height()
            x1 = int(rect.left()   * scale_x)
            y1 = int(rect.top()    * scale_y)
            x2 = int(rect.right()  * scale_x)
            y2 = int(rect.bottom() * scale_y)
            rect_img = (x1, y1, x2, y2)

        self._push_undo()
        self.image = self.image.crop(rect_img)
        self.label.clear_selection()
        self._update_label()

    # ------------------ Adjustments (Brightness etc.) ------------------ #

    def adjust(self, kind: str):
        if not self.image:
            return
        factor, ok = QtWidgets.QInputDialog.getDouble(
            self, kind.title(), f"{kind.title()} factor (0.0–4.0)",
            decimals=2, min=0.0, max=4.0, value=1.2)
        if not ok:
            return
        self._push_undo()
        enhancer_cls = (ImageEnhance.Brightness
                        if kind == "brightness"
                        else ImageEnhance.Contrast)
        enhancer = enhancer_cls(self.image)
        self.image = enhancer.enhance(factor)
        self._update_label()

    # -------------------------- Filters -------------------------- #

    def blur(self):
        if self.image:
            self._push_undo()
            import numpy as np
            cv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGBA2BGRA)
            cv_img = cv2.GaussianBlur(cv_img, (7, 7), sigmaX=0)
            self.image = Image.fromarray(
                cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))
            self._update_label()

    def sharpen(self):
        if self.image:
            self._push_undo()
            self.image = self.image.filter(ImageFilter.SHARPEN)
            self._update_label()

    def grayscale(self):
        if self.image:
            self._push_undo()
            self.image = self.image.convert("L").convert("RGBA")
            self._update_label()

    def edge_detect(self):
        if self.image:
            self._push_undo()
            self.image = self.image.filter(ImageFilter.FIND_EDGES)
            self._update_label()

    # ------------------- New features: OCR & insert ------------------- #

    def ocr_edit(self):
        """
        Run Windows OCR on the entire image, let the user correct the text,
        and optionally burn the corrected string back onto the picture.
        """
        if not self.image:
            return

        try:
            text = run_windows_ocr(self.image)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "OCR Error", str(e))
            return

        # 2) Ask user to edit text
        dlg = QtWidgets.QDialog(self, windowTitle="OCR – edit recognised text")
        dlg.setMinimumSize(600, 400)
        layout = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QPlainTextEdit(text, dlg)
        layout.addWidget(edit)
        chk = QtWidgets.QCheckBox("Draw the corrected text onto the image", dlg)
        chk.setChecked(True)
        layout.addWidget(chk)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                             QtWidgets.QDialogButtonBox.Cancel,
                                             parent=dlg)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        corrected = edit.toPlainText()
        if not chk.isChecked():
            # user only wanted the text (copy to clipboard for convenience)
            QtWidgets.QApplication.clipboard().setText(corrected)
            return

        # 3) Burn text back onto image (simple overlay top-left)
        self._push_undo()
        draw = ImageDraw.Draw(self.image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        margin = 10

        # --- FIX: use draw.textbbox instead of deprecated getsize_multiline ---
        bbox = draw.textbbox((0, 0), corrected, font=font)   # (left, top, right, bottom)
        height = (bbox[3] - bbox[1]) + 2 * margin
        # ---------------------------------------------------------------------

        draw.rectangle([(0, 0), (self.image.width, height)],
                       fill=(255, 255, 255, 200))
        draw.multiline_text((margin, margin), corrected, fill="black",
                            font=font)
        self._update_label()

    def insert_image(self):
        """Paste another image in the centre of the canvas (simple layer)."""
        if not self.image:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Insert Image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif);;All files (*)")
        if not path:
            return
        ins = Image.open(path).convert("RGBA")

        # Ask for scale percentage
        pct, ok = QtWidgets.QInputDialog.getDouble(self, "Insert",
                     "Scale new image (% of original):", value=100.0,
                     min=10.0, max=400.0, decimals=1)
        if not ok:
            return
        if pct != 100.0:
            w = int(ins.width * pct / 100.0)
            h = int(ins.height * pct / 100.0)
            ins = ins.resize((w, h), Image.LANCZOS)

        # Paste at centre
        self._push_undo()
        base = self.image.copy()
        x = (base.width  - ins.width)  // 2
        y = (base.height - ins.height) // 2
        base.paste(ins, (x, y), ins)
        self.image = base
        self._update_label()

    # ---------------- Internal helpers ---------------- #

    def _update_label(self):
        if self.image:
            qimg = pil_to_qimage(self.image)
            pix = QtGui.QPixmap.fromImage(qimg)
            if self.scale_factor != 1.0:
                w = int(self.image.width * self.scale_factor)
                h = int(self.image.height * self.scale_factor)
                pix = pix.scaled(w, h, QtCore.Qt.KeepAspectRatio,
                                 QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pix)
            self.label.resize(pix.size())

    def _on_selection_change(self):
        self._selection_rect = self.label.selection_rect()


# ---------------------------- Application entry ---------------------------- #

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ImageEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

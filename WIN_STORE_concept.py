import sys
import os
import base64
import asyncio
from pathlib import Path

from PIL import Image, ImageQt, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import cv2
from reportlab.pdfgen import canvas

# --- Windows OCR ----------------------------------------------------------- #
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import (
        SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
    )
    from winsdk.windows.security.cryptography import CryptographicBuffer
except ImportError:
    OcrEngine = None

# --- WinUI (python-winrt) imports ----------------------------------------- #
try:
    from winsdk.windows.ui.xaml import Application, Window
    from winsdk.windows.ui.xaml.controls import (
        Page, Grid, MenuBar, MenuBarItem, MenuFlyoutItem, MenuFlyoutSeparator,
        MenuBarItemFlyout, CommandBar, AppBarButton, TextBlock, Image as XamlImage,
        ScrollViewer, Button, Slider, CheckBox
    )
    from winsdk.windows.ui.xaml.media import SolidColorBrush
    from winsdk.windows.ui.xaml import HorizontalAlignment, VerticalAlignment
    from winsdk.windows.storage.pickers import FileOpenPicker, FileSavePicker
    from winsdk.windows.storage import FileAccessMode
    from winsdk.windows.ui.xaml.media.imaging import BitmapImage
    from winsdk.windows.foundation import Uri
    from winsdk.windows.ui.xaml.input import PointerRoutedEventArgs
except ImportError:
    class _ApplicationStub:
        Current = None
        def __init__(self):
            _ApplicationStub.Current = self
        @staticmethod
        def Start(cb):
            cb()
        def Exit(self):
            sys.exit(0)
    class _WindowStub:
        def __init__(self):
            self.Title = ""
            self.Content = None
        def Activate(self): ...
    Application = _ApplicationStub
    Window = _WindowStub
    Page = Grid = MenuBar = MenuBarItem = MenuFlyoutItem = MenuFlyoutSeparator = \
    MenuBarItemFlyout = CommandBar = AppBarButton = TextBlock = XamlImage = \
    ScrollViewer = Button = Slider = CheckBox = object
    SolidColorBrush = object
    HorizontalAlignment = VerticalAlignment = None
    FileOpenPicker = FileSavePicker = object
    FileAccessMode = object
    BitmapImage = object
    Uri = object
    PointerRoutedEventArgs = object

# --------------------- HELPER: WinRT IBuffer from bytes -------------------- #
def _ibuffer(data: bytes):
    return CryptographicBuffer.decode_from_base64_string(
        base64.b64encode(data).decode("ascii")
    )

def _swbmp_from_pil(img: Image.Image) -> "SoftwareBitmap":
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = _ibuffer(img.tobytes())
    return SoftwareBitmap.create_copy_from_buffer(
        buf, BitmapPixelFormat.RGBA8, img.width, img.height, BitmapAlphaMode.STRAIGHT
    )

def run_windows_ocr(img: Image.Image, lang_tag: str = "en") -> str:
    if OcrEngine is None:
        raise RuntimeError("Windows OCR requires ‘winsdk’.")
    lang = Language(lang_tag)
    if not OcrEngine.is_language_supported(lang):
        raise RuntimeError(f"Language “{lang_tag}” is not installed.")
    engine = OcrEngine.try_create_from_language(lang)
    sbmp = _swbmp_from_pil(img)
    async def _do_ocr():
        res = await engine.recognize_async(sbmp)
        return res.text
    return asyncio.run(_do_ocr())

# ---------------------- STUBS: "Rubber-band" selection --------------------- #
class ImageSelectionCanvas:
    def __init__(self):
        self._selection = None
        self._origin = None
        self._is_selecting = False
    def pointer_pressed(self, x, y):
        self._origin = (x, y); self._is_selecting = True
    def pointer_moved(self, x, y):
        if self._is_selecting and self._origin:
            ox, oy = self._origin
            self._selection = (min(ox, x), min(oy, y), abs(x-ox), abs(y-oy))
    def pointer_released(self, x, y):
        self.pointer_moved(x, y); self._is_selecting = False
    def get_selection_rect(self):
        return self._selection
    def clear_selection(self):
        self._selection = None

# --------------------- Image Editor: The main logic ------------------------ #
class ImageEditor(Page):
    def __init__(self):
        super().__init__()
        self.image = None
        self.undo_stack, self.redo_stack = [], []
        self.scale_factor = 1.0
        self.selection_canvas = ImageSelectionCanvas()
        self.root_grid = Grid(); self.Content = self.root_grid
        self._create_menubar(); self._create_main_view(); self._create_command_bar()

    # -------------------- UI Construction ---------------------------------- #
    def _create_menubar(self):
        self.menubar = MenuBar(); self.root_grid.Children.append(self.menubar)
        file_item = MenuBarItem(); file_item.Label = "File"; file_flyout = MenuBarItemFlyout()
        open_menu = MenuFlyoutItem(); open_menu.Text = "Open…"; open_menu.Click = self.open_image
        save_menu = MenuFlyoutItem(); save_menu.Text = "Save As…"; save_menu.Click = self.save_image
        export_pdf_menu = MenuFlyoutItem(); export_pdf_menu.Text = "Export PDF…"; export_pdf_menu.Click = self.export_pdf
        exit_menu = MenuFlyoutItem(); exit_menu.Text = "Quit"; exit_menu.Click = self.close_application
        file_flyout.Items.extend([open_menu, save_menu, export_pdf_menu, MenuFlyoutSeparator(), exit_menu])
        file_item.Flyout = file_flyout
        edit_item = MenuBarItem(); edit_item.Label = "Edit"; edit_flyout = MenuBarItemFlyout()
        undo_menu = MenuFlyoutItem(); undo_menu.Text = "Undo"; undo_menu.Click = self.undo
        redo_menu = MenuFlyoutItem(); redo_menu.Text = "Redo"; redo_menu.Click = self.redo
        edit_flyout.Items.extend([undo_menu, redo_menu]); edit_item.Flyout = edit_flyout
        view_item = MenuBarItem(); view_item.Label = "View"; view_flyout = MenuBarItemFlyout()
        zoom_in_menu = MenuFlyoutItem(); zoom_in_menu.Text = "Zoom In"; zoom_in_menu.Click = lambda s,e: self.zoom(1.25)
        zoom_out_menu = MenuFlyoutItem(); zoom_out_menu.Text = "Zoom Out"; zoom_out_menu.Click = lambda s,e: self.zoom(0.8)
        zoom_reset_menu = MenuFlyoutItem(); zoom_reset_menu.Text = "Actual Size"; zoom_reset_menu.Click = lambda s,e: self.set_zoom(1.0)
        view_flyout.Items.extend([zoom_in_menu, zoom_out_menu, zoom_reset_menu]); view_item.Flyout = view_flyout
        image_item = MenuBarItem(); image_item.Label = "Image"; image_flyout = MenuBarItemFlyout()
        rotate_left_menu = MenuFlyoutItem(); rotate_left_menu.Text = "Rotate ⟲"; rotate_left_menu.Click = lambda s,e: self.rotate(-90)
        rotate_right_menu = MenuFlyoutItem(); rotate_right_menu.Text = "Rotate ⟳"; rotate_right_menu.Click = lambda s,e: self.rotate(90)
        flip_h_menu = MenuFlyoutItem(); flip_h_menu.Text = "Flip Horiz"; flip_h_menu.Click = lambda s,e: self.flip(True)
        flip_v_menu = MenuFlyoutItem(); flip_v_menu.Text = "Flip Vert"; flip_v_menu.Click = lambda s,e: self.flip(False)
        resize_menu = MenuFlyoutItem(); resize_menu.Text = "Resize…"; resize_menu.Click = self.resize_image
        crop_menu = MenuFlyoutItem(); crop_menu.Text = "Crop Selection"; crop_menu.Click = self.crop
        insert_img_menu = MenuFlyoutItem(); insert_img_menu.Text = "Insert Image…"; insert_img_menu.Click = self.insert_image
        image_flyout.Items.extend([
            rotate_left_menu, rotate_right_menu, flip_h_menu, flip_v_menu,
            resize_menu, crop_menu, insert_img_menu
        ]); image_item.Flyout = image_flyout
        adjust_item = MenuBarItem(); adjust_item.Label = "Adjust"; adjust_flyout = MenuBarItemFlyout()
        brightness_menu = MenuFlyoutItem(); brightness_menu.Text = "Brightness…"; brightness_menu.Click = lambda s,e: self.adjust("brightness")
        contrast_menu = MenuFlyoutItem(); contrast_menu.Text = "Contrast…"; contrast_menu.Click = lambda s,e: self.adjust("contrast")
        adjust_flyout.Items.extend([brightness_menu, contrast_menu]); adjust_item.Flyout = adjust_flyout
        filter_item = MenuBarItem(); filter_item.Label = "Filters"; filter_flyout = MenuBarItemFlyout()
        blur_menu = MenuFlyoutItem(); blur_menu.Text = "Blur"; blur_menu.Click = self.blur
        sharpen_menu = MenuFlyoutItem(); sharpen_menu.Text = "Sharpen"; sharpen_menu.Click = self.sharpen
        grayscale_menu = MenuFlyoutItem(); grayscale_menu.Text = "Grayscale"; grayscale_menu.Click = self.grayscale
        edge_menu = MenuFlyoutItem(); edge_menu.Text = "Edge Detect"; edge_menu.Click = self.edge_detect
        filter_flyout.Items.extend([blur_menu, sharpen_menu, grayscale_menu, edge_menu]); filter_item.Flyout = filter_flyout
        tools_item = MenuBarItem(); tools_item.Label = "Tools"; tools_flyout = MenuBarItemFlyout()
        ocr_menu = MenuFlyoutItem(); ocr_menu.Text = "OCR (Edit)…"; ocr_menu.Click = self.ocr_edit
        tools_flyout.Items.append(ocr_menu); tools_item.Flyout = tools_flyout
        self.menubar.Items.extend([file_item, edit_item, view_item, image_item, adjust_item, filter_item, tools_item])

    def _create_main_view(self):
        self.scroll_viewer = ScrollViewer(); self.root_grid.Children.append(self.scroll_viewer)
        self.image_control = XamlImage()
        self.image_control.HorizontalAlignment = HorizontalAlignment.Center
        self.image_control.VerticalAlignment = VerticalAlignment.Center
        self.image_control.PointerPressed += self._on_pointer_pressed
        self.image_control.PointerMoved += self._on_pointer_moved
        self.image_control.PointerReleased += self._on_pointer_released
        self.scroll_viewer.Content = self.image_control

    def _create_command_bar(self):
        self.command_bar = CommandBar(); self.root_grid.Children.append(self.command_bar)
        def _btn(label, cb): b = AppBarButton(); b.Label = label; b.Click = cb; return b
        buttons = [
            _btn("Open", self.open_image), _btn("Save", self.save_image),
            _btn("Undo", self.undo), _btn("Redo", self.redo),
            _btn("⟲", lambda s,e: self.rotate(-90)), _btn("⟳", lambda s,e: self.rotate(90)),
            _btn("Crop", self.crop), _btn("Insert", self.insert_image),
            _btn("Brightness", lambda s,e: self.adjust("brightness")),
            _btn("Contrast", lambda s,e: self.adjust("contrast")),
            _btn("Zoom In", lambda s,e: self.zoom(1.25)),
            _btn("Zoom Out", lambda s,e: self.zoom(0.8)),
            _btn("Actual Size", lambda s,e: self.set_zoom(1.0)),
            _btn("OCR", self.ocr_edit)
        ]
        for b in buttons: self.command_bar.PrimaryCommands.append(b)

    # ----------------------------- File ops --------------------------------- #
    def open_image(self, sender=None, args=None):
        picker = FileOpenPicker()
        future = picker.PickSingleFileAsync()
        file = asyncio.run(future)
        if file:
            self.image = Image.open(Path(file.Path))
            self.scale_factor = 1.0; self._push_undo(); self.selection_canvas.clear_selection()
            self._update_image_control()

    def save_image(self, sender=None, args=None):
        if not self.image: return
        picker = FileSavePicker()
        future = picker.PickSaveFileAsync()
        file = asyncio.run(future)
        if file: self.image.save(file.Path)

    def export_pdf(self, sender=None, args=None):
        if not self.image: return
        picker = FileSavePicker()
        future = picker.PickSaveFileAsync()
        file = asyncio.run(future)
        if not file: return
        pdf_path = file.Path
        c = canvas.Canvas(pdf_path)
        tmp = Path(pdf_path).with_suffix(".png")
        self.image.save(tmp); c.drawImage(str(tmp), 0, 0, width=self.image.width, height=self.image.height)
        c.showPage(); c.save(); tmp.unlink(missing_ok=True)

    def close_application(self, sender=None, args=None):
        Application.Current.Exit()

    # ------------------------ Undo / Redo logic ----------------------------- #
    def _push_undo(self):
        if self.image: self.undo_stack.append(self.image.copy()); self.redo_stack.clear()
    def undo(self, sender=None, args=None):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.image = self.undo_stack[-1].copy(); self._update_image_control()
    def redo(self, sender=None, args=None):
        if self.redo_stack:
            self.image = self.redo_stack.pop(); self.undo_stack.append(self.image.copy()); self._update_image_control()

    # ---------------------------- Zooming ----------------------------------- #
    def zoom(self, factor: float):
        if self.image:
            self.scale_factor = max(0.1, min(8.0, self.scale_factor * factor)); self._update_image_control()
    def set_zoom(self, factor: float):
        if self.image: self.scale_factor = factor; self._update_image_control()

    # -------------------- Basic image transforms ---------------------------- #
    def rotate(self, angle: int):
        if self.image:
            self._push_undo(); self.image = self.image.rotate(angle, expand=True); self._update_image_control()
    def flip(self, horizontal: bool):
        if self.image:
            self._push_undo()
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM)
            self._update_image_control()
    def resize_image(self, sender=None, args=None):
        if not self.image: return
        new_width = self._prompt_for_int("Resize Width", self.image.width)
        if not new_width: return
        new_height = self._prompt_for_int("Resize Height", self.image.height)
        if not new_height: return
        self._push_undo(); self.image = self.image.resize((new_width, new_height), Image.LANCZOS); self._update_image_control()
    def crop(self, sender=None, args=None):
        if not self.image: return
        rect = self.selection_canvas.get_selection_rect()
        if rect is None or rect[2] < 2 or rect[3] < 2:
            w, h = self.image.size; rect_img = (w//4, h//4, w*3//4, h*3//4)
        else:
            x1, y1, w, h = rect; rect_img = (x1, y1, x1+w, y1+h)
        self._push_undo(); self.image = self.image.crop(rect_img); self.selection_canvas.clear_selection(); self._update_image_control()

    # ------------------ Adjustments (Brightness etc.) ----------------------- #
    def adjust(self, kind: str):
        if not self.image: return
        factor = self._prompt_for_float(f"{kind.title()} factor", 1.2, 0.0, 4.0)
        if factor is None: return
        self._push_undo()
        enhancer = (ImageEnhance.Brightness if kind == "brightness" else ImageEnhance.Contrast)(self.image)
        self.image = enhancer.enhance(factor); self._update_image_control()

    # -------------------------- Filters ------------------------------------- #
    def blur(self, sender=None, args=None):
        if self.image:
            import numpy as np
            self._push_undo()
            cv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGBA2BGRA)
            cv_img = cv2.GaussianBlur(cv_img, (7, 7), 0)
            self.image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))
            self._update_image_control()
    def sharpen(self, sender=None, args=None):
        if self.image:
            self._push_undo(); self.image = self.image.filter(ImageFilter.SHARPEN); self._update_image_control()
    def grayscale(self, sender=None, args=None):
        if self.image:
            self._push_undo(); self.image = self.image.convert("L").convert("RGBA"); self._update_image_control()
    def edge_detect(self, sender=None, args=None):
        if self.image:
            self._push_undo(); self.image = self.image.filter(ImageFilter.FIND_EDGES); self._update_image_control()

    # ------------------- OCR & Insert Image --------------------------------- #
    def ocr_edit(self, sender=None, args=None):
        if not self.image: return
        try: text = run_windows_ocr(self.image)
        except Exception as e: self._show_message(f"OCR Error: {e}"); return
        corrected = self._prompt_for_text("OCR Result", text)
        if corrected is None: return
        burn_back = True
        if burn_back:
            self._push_undo(); draw = ImageDraw.Draw(self.image)
            try: font = ImageFont.truetype("arial.ttf", 20)
            except IOError: font = ImageFont.load_default()
            margin = 10
            lines = corrected.splitlines() or [""]
            height = sum(font.getsize(line)[1] for line in lines) + 2*margin
            draw.rectangle([(0,0),(self.image.width,height)], fill=(255,255,255,200))
            draw.multiline_text((margin, margin), corrected, fill="black", font=font)
            self._update_image_control()

    def insert_image(self, sender=None, args=None):
        if not self.image: return
        picker = FileOpenPicker()
        file = asyncio.run(picker.PickSingleFileAsync())
        if not file: return
        ins = Image.open(Path(file.Path)).convert("RGBA")
        pct = self._prompt_for_float("Scale new image (%)", 100.0, 10.0, 400.0)
        if pct is None: return
        if pct != 100.0:
            w = int(ins.width*pct/100.0); h = int(ins.height*pct/100.0)
            ins = ins.resize((w,h), Image.LANCZOS)
        self._push_undo(); base = self.image.copy()
        x = (base.width-ins.width)//2; y = (base.height-ins.height)//2
        base.paste(ins, (x,y), ins); self.image = base; self._update_image_control()

    # ---------------------- Internal UI Helpers ----------------------------- #
    def _update_image_control(self):
        if self.image:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                self.image.save(tmp, format="PNG"); temp_path = tmp.name
            bitmap = BitmapImage(); bitmap.UriSource = Uri("file:///"+temp_path.replace("\\","/"))
            self.image_control.Source = bitmap

    def _on_pointer_pressed(self, sender, e): p = e.GetCurrentPoint(sender).Position; self.selection_canvas.pointer_pressed(p.X, p.Y)
    def _on_pointer_moved(self, sender, e): p = e.GetCurrentPoint(sender).Position; self.selection_canvas.pointer_moved(p.X, p.Y)
    def _on_pointer_released(self, sender, e): p = e.GetCurrentPoint(sender).Position; self.selection_canvas.pointer_released(p.X, p.Y)

    def _prompt_for_int(self, title, default_val): return default_val
    def _prompt_for_float(self, prompt, default_val, min_val, max_val): return default_val
    def _prompt_for_text(self, title, initial): return initial
    def _show_message(self, msg): print(msg)

# ---------------------------- Application entry ---------------------------- #
class MyWinUIApplication(Application):
    def OnLaunched(self, args):
        self.window = Window(); self.window.Title = "PDFSage Photo Editor (WinUI port)"
        self.window.Content = ImageEditor(); self.window.Activate()

def main():
    app = MyWinUIApplication()
    if hasattr(Application, "Start"):
        Application.Start(lambda: None)

if __name__ == "__main__":
    main()

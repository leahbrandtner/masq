"""
CZI ROI Mask Analyzer

Install in Thonny:
Tools > Manage packages:
numpy
pillow
scipy
scikit-image
aicspylibczi

If aicspylibczi does not work, try:
czifile
"""

# =============================================================================
# Imports
# =============================================================================

import csv
import os
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

try:
    from skimage import filters, measure, morphology, segmentation
except Exception:
    filters = measure = morphology = segmentation = None

from PIL import Image, ImageDraw, ImageFont, ImageTk

    
# =============================================================================
# Theme
# =============================================================================

BG = "#1f2023"
PANEL_BG = "#24262b"
CANVAS_BG = "#17191d"
ENTRY_BG = "#2b2d31"
TEXT = "#eeeeee"
MUTED_TEXT = "#b8bcc7"
INACTIVE_TEXT = "#686c75"
ACCENT = "#18c7b7"
ACCENT_DARK = "#0f8f84"
BUTTON_BG = "#343740"
BUTTON_ACTIVE = "#4a4f5c"

BASE_FONT = ("Segoe UI", 10)
TITLE_FONT = ("Segoe UI", 11, "bold")
SUBTITLE_FONT = ("Segoe UI", 10, "bold")
BUTTON_FONT = ("Segoe UI", 10, "bold")

# =============================================================================
# Settings
# =============================================================================

@dataclass
class Settings:
    root_folder: str
    output_csv: str
    folder_level_names: list
    channel_names: dict
    roi_method: str
    roi_width: int
    roi_height: int
    roi_channel: int
    mask_mode: str
    intensity_channels: list
    mask_channel_1: int
    use_second_mask: bool
    mask_channel_2: int
    target_channel: int
    gaussian_sigma: float
    threshold_method: str
    manual_threshold: float
    use_separate_thresholds: bool
    mask1_threshold_method: str
    mask1_manual_threshold: float
    mask2_threshold_method: str
    mask2_manual_threshold: float
    target_threshold_method: str
    target_manual_threshold: float
    min_object_area: int
    max_object_area: int
    min_overlap_fraction: float
    fill_holes: bool
    clear_border: bool

# =============================================================================
# CZI Loading
# =============================================================================

def load_czi_as_cyx(path):
    """Load .czi image and return array as Channel, Y, X."""
    errors = []

    try:
        from aicspylibczi import CziFile
        czi = CziFile(path)
        data, _shape = czi.read_image()
        dims = getattr(czi, "dims", "")
        return standardize_to_cyx(np.asarray(data), dims)
    except Exception as exc:
        errors.append("aicspylibczi: " + str(exc))

    try:
        import czifile
        data = czifile.imread(path)
        return standardize_to_cyx(np.asarray(data), "")
    except Exception as exc:
        errors.append("czifile: " + str(exc))

    raise RuntimeError(
        "Could not read CZI file. Install aicspylibczi or czifile.\n"
        + "\n".join(errors)
    )


def standardize_to_cyx(data, dims=""):
    """Convert common microscope array layouts to C, Y, X."""
    original = np.asarray(data)
    arr = np.squeeze(original)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :].astype(np.float32)

    if dims and len(dims) == original.ndim and "Y" in dims and "X" in dims:
        indexer = []
        kept = []
        for _axis_number, axis_name in enumerate(dims):
            if axis_name in ("C", "Y", "X"):
                indexer.append(slice(None))
                kept.append(axis_name)
            else:
                indexer.append(0)

        reduced = np.asarray(original[tuple(indexer)])
        kept = "".join(kept)

        if "C" in kept:
            order = [kept.index("C"), kept.index("Y"), kept.index("X")]
            return np.transpose(reduced, order).astype(np.float32)

        return reduced[np.newaxis, :, :].astype(np.float32)

    while arr.ndim > 3:
        small_axes = [i for i, size in enumerate(arr.shape) if size <= 8]
        channel_axis = small_axes[0] if small_axes else 0
        indexer = [0] * arr.ndim
        indexer[channel_axis] = slice(None)
        indexer[-2] = slice(None)
        indexer[-1] = slice(None)
        arr = arr[tuple(indexer)]

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        small_axes = [i for i, size in enumerate(arr.shape) if size <= 8]
        channel_axis = small_axes[0] if small_axes else 0
        arr = np.moveaxis(arr, channel_axis, 0)

    return arr.astype(np.float32)


# =============================================================================
# Image Processing
# =============================================================================

def normalize_for_display(image):
    """Convert one channel to 8-bit preview image."""
    img = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(img)

    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)

    low, high = np.percentile(img[finite], (1, 99.8))

    if high <= low:
        low = float(np.min(img[finite]))
        high = float(np.max(img[finite])) or 1.0

    scaled = np.clip((img - low) / (high - low + 1e-9), 0, 1)
    return (scaled * 255).astype(np.uint8)


def make_rgb_preview(image_cyx, preview_channel_index=0):
    """Create a grayscale RGB preview from the selected channel."""
    channels = image_cyx.shape[0]
    channel_index = clamp_channel(preview_channel_index, channels)
    base = normalize_for_display(image_cyx[channel_index])
    return np.dstack([base, base, base])


def subtract_background(image, sigma):
    """Gaussian background subtraction."""
    img = np.asarray(image, dtype=np.float32)

    if sigma <= 0 or ndi is None:
        return img

    background = ndi.gaussian_filter(img, sigma=sigma)
    corrected = img - background
    corrected[corrected < 0] = 0
    return corrected


def threshold_image(image, method, manual_threshold):
    """Threshold with selectable scientific methods."""
    if thresholding_is_disabled(method):
        return np.zeros(np.asarray(image).shape, dtype=bool)

    if filters is None:
        raise RuntimeError("scikit-image is required for thresholding.")

    img = np.asarray(image, dtype=np.float32)
    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.zeros(img.shape, dtype=bool)

    method = method.lower()

    if method == "manual":
        threshold = manual_threshold
    elif method == "yen":
        threshold = filters.threshold_yen(finite)
    elif method == "triangle":
        threshold = filters.threshold_triangle(finite)
    elif method == "li":
        threshold = filters.threshold_li(finite)
    else:
        threshold = filters.threshold_otsu(finite)

    return img > threshold


def thresholding_is_disabled(method):
    return str(method).lower().startswith("none")


def clean_mask(mask, settings):
    """Object filtering and mask cleanup."""
    if morphology is None or segmentation is None or measure is None:
        raise RuntimeError("scikit-image is required for object filtering.")

    cleaned = mask.astype(bool)

    if settings.fill_holes and ndi is not None:
        cleaned = ndi.binary_fill_holes(cleaned)

    if settings.clear_border:
        cleaned = segmentation.clear_border(cleaned)

    if settings.min_object_area > 0:
        cleaned = morphology.remove_small_objects(cleaned, settings.min_object_area)

    if settings.max_object_area > 0:
        labels = measure.label(cleaned)
        keep = np.zeros(cleaned.shape, dtype=bool)

        for region in measure.regionprops(labels):
            if region.area <= settings.max_object_area:
                keep[labels == region.label] = True

        cleaned = keep

    return cleaned


def build_mask(channel_image, settings, method=None, manual_threshold=None):
    """Create one cleaned mask from one channel."""
    corrected = subtract_background(channel_image, settings.gaussian_sigma)
    raw_mask = threshold_image(
        corrected,
        method if method is not None else settings.threshold_method,
        manual_threshold if manual_threshold is not None else settings.manual_threshold,
    )
    return clean_mask(raw_mask, settings)


def get_mask1_threshold_settings(settings):
    if settings.use_separate_thresholds:
        return settings.mask1_threshold_method, settings.mask1_manual_threshold
    return settings.threshold_method, settings.manual_threshold


def get_mask2_threshold_settings(settings):
    if settings.use_separate_thresholds:
        return settings.mask2_threshold_method, settings.mask2_manual_threshold
    return settings.threshold_method, settings.manual_threshold


def get_target_threshold_settings(settings):
    if settings.use_separate_thresholds:
        return settings.target_threshold_method, settings.target_manual_threshold
    return settings.threshold_method, settings.manual_threshold


# =============================================================================
# ROI Selection
# =============================================================================

def calculate_roi(image_cyx, method, width, height, roi_channel):
    """Return ROI as x, y, width, height."""
    channels, image_height, image_width = image_cyx.shape

    width = max(1, min(int(width), image_width))
    height = max(1, min(int(height), image_height))

    if method == "Whole image":
        return 0, 0, image_width, image_height

    if method == "Centered ROI":
        x = (image_width - width) // 2
        y = (image_height - height) // 2
        return x, y, width, height

    channel_index = clamp_channel(roi_channel, channels)
    signal = image_cyx[channel_index]

    if method == "Lowest signal in selected channel":
        return best_window_by_mean(signal, width, height, want_high=False)

    if method == "Highest signal in selected channel":
        return best_window_by_mean(signal, width, height, want_high=True)

    return (image_width - width) // 2, (image_height - height) // 2, width, height


def best_window_by_mean(image, width, height, want_high=True):
    """Find ROI with high/low signal; if similar, prefer the most centered ROI."""
    img = np.asarray(image, dtype=np.float32)
    image_height, image_width = img.shape

    width = min(width, image_width)
    height = min(height, image_height)

    center_x = image_width / 2
    center_y = image_height / 2

    step_y = max(1, height // 10)
    step_x = max(1, width // 10)

    best_score = None
    best_distance = None
    best_xy = (0, 0)

    for y in range(0, image_height - height + 1, step_y):
        for x in range(0, image_width - width + 1, step_x):
            roi = img[y:y + height, x:x + width]
            score = float(np.mean(roi))

            roi_center_x = x + width / 2
            roi_center_y = y + height / 2
            distance = ((roi_center_x - center_x) ** 2 + (roi_center_y - center_y) ** 2) ** 0.5

            if best_score is None:
                best_score = score
                best_distance = distance
                best_xy = (x, y)
                continue

            score_difference = abs(score - best_score)
            tolerance = abs(best_score) * 0.03 + 1e-9

            better_signal = score > best_score if want_high else score < best_score
            similar_but_more_centered = score_difference <= tolerance and distance < best_distance

            if better_signal or similar_but_more_centered:
                best_score = score
                best_distance = distance
                best_xy = (x, y)

    return best_xy[0], best_xy[1], width, height


# =============================================================================
# Measurements
# =============================================================================

def analyze_one_image(path, settings):
    """Analyze one CZI image and return one CSV row."""
    image_cyx = load_czi_as_cyx(path)
    channels, _image_height, _image_width = image_cyx.shape

    roi_x, roi_y, roi_w, roi_h = calculate_roi(
        image_cyx,
        settings.roi_method,
        settings.roi_width,
        settings.roi_height,
        settings.roi_channel,
    )

    roi_slice = (slice(roi_y, roi_y + roi_h), slice(roi_x, roi_x + roi_w))
    
    selected_intensity_channels = settings.intensity_channels
    if not selected_intensity_channels:
        selected_intensity_channels = [settings.target_channel]

    intensity_results = {}

    for channel_index in selected_intensity_channels:
        safe_index = clamp_channel(channel_index, channels)
        channel_name = settings.channel_names.get(
            safe_index,
            f"Channel {safe_index + 1}",
        )
        channel_key = clean_column_name(channel_name)

        channel_image = image_cyx[safe_index][roi_slice]
        corrected_channel = subtract_background(channel_image, settings.gaussian_sigma)

        intensity_results[f"mean_intensity_roi_{channel_key}"] = safe_mean(corrected_channel)
        intensity_results[f"median_intensity_roi_{channel_key}"] = safe_median(corrected_channel)

    final_mask = None
    target_binary = None
    target_image = None

    if settings.mask_mode != "No mask - ROI intensity only":
        mask1_method, mask1_manual = get_mask1_threshold_settings(settings)
        mask2_method, mask2_manual = get_mask2_threshold_settings(settings)
        target_method, target_manual = get_target_threshold_settings(settings)

        mask1_image = image_cyx[clamp_channel(settings.mask_channel_1, channels)][roi_slice]
        mask1 = build_mask(mask1_image, settings, mask1_method, mask1_manual)

        if settings.mask_mode == "AND mask from two channels":
            mask2_image = image_cyx[clamp_channel(settings.mask_channel_2, channels)][roi_slice]
            mask2 = build_mask(mask2_image, settings, mask2_method, mask2_manual)
            final_mask = mask1 & mask2
        else:
            final_mask = mask1

        for channel_index in selected_intensity_channels:
            safe_index = clamp_channel(channel_index, channels)
            channel_name = settings.channel_names.get(
                safe_index,
                f"Channel {safe_index + 1}",
            )
            channel_key = clean_column_name(channel_name)

            channel_image = image_cyx[safe_index][roi_slice]
            corrected_channel = subtract_background(channel_image, settings.gaussian_sigma)

            mask_values = corrected_channel[final_mask]

            intensity_results[f"mean_intensity_in_mask_{channel_key}"] = safe_mean(mask_values)
            intensity_results[f"median_intensity_in_mask_{channel_key}"] = safe_median(mask_values)   

        target_raw = image_cyx[clamp_channel(settings.target_channel, channels)][roi_slice]
        target_image = subtract_background(target_raw, settings.gaussian_sigma)

        target_binary = threshold_image(
            target_image,
            target_method,
            target_manual,
        )
        target_binary = clean_mask(target_binary, settings)

        object_count, object_area_px, object_mean_intensity = count_target_objects(
            target_binary,
            target_image,
            final_mask,
            settings.min_overlap_fraction,
        )

        intersection = final_mask & target_binary
        union = final_mask | target_binary
    else:
        target_raw = image_cyx[clamp_channel(settings.target_channel, channels)][roi_slice]
        target_image = subtract_background(target_raw, settings.gaussian_sigma)
        final_mask = np.zeros(target_image.shape, dtype=bool)
        target_binary = np.zeros(target_image.shape, dtype=bool)
        intersection = np.zeros(target_image.shape, dtype=bool)
        union = np.zeros(target_image.shape, dtype=bool)
        object_count = 0
        object_area_px = 0
        object_mean_intensity = 0.0

    row = build_folder_metadata(
        path,
        settings.root_folder,
        settings.folder_level_names,
    )
    
    roi_channel_name = settings.channel_names.get(
        settings.roi_channel,
        f"Channel {settings.roi_channel + 1}",
    )

    if settings.roi_method == "Highest signal in selected channel":
        roi_method_output = f"Highest signal in {roi_channel_name}"
    elif settings.roi_method == "Lowest signal in selected channel":
        roi_method_output = f"Lowest signal in {roi_channel_name}"
    else:
        roi_method_output = settings.roi_method

    mask1_method, mask1_manual = get_mask1_threshold_settings(settings)
    mask2_method, mask2_manual = get_mask2_threshold_settings(settings)
    target_method, target_manual = get_target_threshold_settings(settings)
    
    row.update({
        "image_name": Path(path).name,
        "image_path": str(path),
        "mask_channel_1": settings.channel_names.get(
            settings.mask_channel_1,
            f"Channel {settings.mask_channel_1 + 1}",
        ),
        "mask_channel_2": settings.channel_names.get(
            settings.mask_channel_2,
            f"Channel {settings.mask_channel_2 + 1}",
        ) if settings.use_second_mask else "",
        "target_channel": settings.channel_names.get(
            settings.target_channel,
            f"Channel {settings.target_channel + 1}",
        ),
        "roi_method": roi_method_output,
        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_width": roi_w,
        "roi_height": roi_h,
        "threshold_mode": "separate" if settings.use_separate_thresholds else "shared",
        "mask1_threshold_method": mask1_method,
        "mask1_manual_threshold": mask1_manual,
        "mask2_threshold_method": mask2_method if settings.mask_mode == "AND mask from two channels" else "",
        "mask2_manual_threshold": mask2_manual if settings.mask_mode == "AND mask from two channels" else "",
        "target_threshold_method": target_method,
        "target_manual_threshold": target_manual,
        "object_count": object_count,
        "object_area_px": object_area_px,
        "object_mean_intensity": object_mean_intensity,
        "mean_intensity_in_mask": safe_mean(target_image[final_mask]),
        "median_intensity_in_mask": safe_median(target_image[final_mask]),
        "mean_intensity_roi": safe_mean(target_image),
        "median_intensity_roi": safe_median(target_image),
        "jaccard_index": safe_divide(
            np.count_nonzero(intersection),
            np.count_nonzero(union),
        ),
        "target_signal_fraction_in_mask": safe_divide(
            np.count_nonzero(intersection),
            np.count_nonzero(target_binary),
        ),
        "mask_area_percent": 100.0 * safe_divide(
            np.count_nonzero(final_mask),
            final_mask.size,
        ),
        "target_area_percent_in_mask": 100.0 * safe_divide(
            np.count_nonzero(intersection),
            final_mask.size,
        ),
    })
    
    row.update(intensity_results)

    if settings.mask_mode != "No mask - ROI intensity only":
        row.update({
            "mask_channel_1": settings.channel_names.get(
                settings.mask_channel_1,
                f"Channel {settings.mask_channel_1 + 1}",
            ),
            "mask_channel_2": settings.channel_names.get(
                settings.mask_channel_2,
                f"Channel {settings.mask_channel_2 + 1}",
            ) if settings.mask_mode == "AND mask from two channels" else "",
            "target_channel": settings.channel_names.get(
                settings.target_channel,
                f"Channel {settings.target_channel + 1}",
            ),
            "object_count": object_count,
            "object_area_px": object_area_px,
            "object_mean_intensity": object_mean_intensity,
            "jaccard_index": safe_divide(
                np.count_nonzero(intersection),
                np.count_nonzero(union),
            ),
            "target_signal_fraction_in_mask": safe_divide(
                np.count_nonzero(intersection),
                np.count_nonzero(target_binary),
            ),
            "mask_area_percent": 100.0 * safe_divide(
                np.count_nonzero(final_mask),
                final_mask.size,
            ),
            "target_area_percent_in_mask": 100.0 * safe_divide(
                np.count_nonzero(intersection),
                final_mask.size,
            ),
        })

    return row


def count_target_objects(target_binary, target_image, final_mask, min_overlap_fraction):
    """Count target objects whose overlap with the final mask is sufficient."""
    if measure is None:
        raise RuntimeError("scikit-image is required for object measurement.")

    labels = measure.label(target_binary)

    count = 0
    total_area = 0
    intensities = []

    for region in measure.regionprops(labels, intensity_image=target_image):
        object_pixels = labels == region.label
        overlap_pixels = object_pixels & final_mask

        overlap_fraction = safe_divide(
            np.count_nonzero(overlap_pixels),
            region.area,
        )

        if overlap_fraction >= min_overlap_fraction:
            count += 1
            total_area += int(region.area)

            if np.any(overlap_pixels):
                intensities.append(float(np.mean(target_image[overlap_pixels])))

    return count, total_area, safe_mean(np.asarray(intensities, dtype=np.float32))


def build_folder_metadata(image_path, root_folder, level_names):
    """Create one CSV column for each subfolder level."""
    image_parent = Path(image_path).parent
    root = Path(root_folder)

    try:
        parts = image_parent.relative_to(root).parts
    except ValueError:
        parts = image_parent.parts

    row = {}

    for index, name in enumerate(level_names):
        column = name.strip() or f"folder_level_{index + 1}"
        row[column] = parts[index] if index < len(parts) else ""

    return row

def clamp_channel(channel_index, channel_count):
    return max(0, min(int(channel_index), channel_count - 1))

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)

def safe_mean(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))

def safe_median(values):
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 0.0

    return float(np.median(arr))


# =============================================================================
# GUI
# =============================================================================

class App:
    """Tkinter GUI for CZI ROI Mask Analyzer."""

    def __init__(self, root):
        self.root = root
        self.root.title("CZI ROI Mask Analyzer")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)

        self.files = []
        self.preview_image_cyx = None
        self.preview_index = tk.IntVar(value=0)
        self.preview_channel = tk.StringVar(value="Channel 1")
        self.current_preview_rgb = None

        self.status_queue = queue.Queue()
        self.is_running = False
        self.run_animation_step = 0

        self.create_variables()
        self.create_layout()
        self.poll_status_queue()
        
    # -------------------------------------------------------------------------
    # GUI Variables
    # -------------------------------------------------------------------------

    def create_variables(self):
        self.root_folder_var = tk.StringVar()
        self.output_csv_var = tk.StringVar(
            value=str(Path.cwd() / "czi_roi_mask_results.csv")
        )

        self.folder_levels_var = tk.StringVar(
            value="replicate, construct, coverslip"
        )

        self.channel_name_vars = [
            tk.StringVar(value=f"Channel {i + 1}") for i in range(4)
        ]

        self.roi_method_var = tk.StringVar(value="Centered ROI")
        self.roi_width_var = tk.IntVar(value=512)
        self.roi_height_var = tk.IntVar(value=512)
        self.roi_channel_var = tk.StringVar(value="Channel 1")
        
        self.mask_mode_var = tk.StringVar(value="No mask - ROI intensity only")

        self.intensity_channel_vars = [
            tk.BooleanVar(value=True),
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
        ]
        
        for var in self.intensity_channel_vars:
            var.trace_add("write", lambda *args: self.root.after(300, self.update_preview))

        self.histogram_source_var = tk.StringVar(value="Target channel")
        self.histogram_height_var = tk.IntVar(value=160)
        self.histogram_log_scale_var = tk.BooleanVar(value=True)
        self.overlay_alpha_var = tk.DoubleVar(value=0.60)
        self.show_target_overlay_var = tk.BooleanVar(value=True)
        self.show_mask1_overlay_var = tk.BooleanVar(value=True)
        self.show_mask2_overlay_var = tk.BooleanVar(value=True)
        self.show_final_mask_overlay_var = tk.BooleanVar(value=True)

        self.mask_channel_1_var = tk.StringVar(value="Channel 1")
        self.use_second_mask_var = tk.BooleanVar(value=False)
        self.mask_channel_2_var = tk.StringVar(value="Channel 2")
        self.target_channel_var = tk.StringVar(value="Channel 3")

        self.gaussian_sigma_var = tk.DoubleVar(value=8.0)
        self.threshold_method_var = tk.StringVar(value="Otsu")
        self.manual_threshold_var = tk.DoubleVar(value=100.0)
        self.use_separate_thresholds_var = tk.BooleanVar(value=False)
        self.mask1_threshold_method_var = tk.StringVar(value="Otsu")
        self.mask1_manual_threshold_var = tk.DoubleVar(value=100.0)
        self.mask2_threshold_method_var = tk.StringVar(value="Otsu")
        self.mask2_manual_threshold_var = tk.DoubleVar(value=100.0)
        self.target_threshold_method_var = tk.StringVar(value="Otsu")
        self.target_manual_threshold_var = tk.DoubleVar(value=100.0)

        self.min_object_area_var = tk.IntVar(value=20)
        self.max_object_area_var = tk.IntVar(value=1000)
        self.limit_max_object_area_var = tk.BooleanVar(value=False)
        self.min_overlap_fraction_var = tk.DoubleVar(value=0.10)

        self.fill_holes_var = tk.BooleanVar(value=True)
        self.clear_border_var = tk.BooleanVar(value=False)
        
        for var in [
            self.roi_method_var,
            self.roi_width_var,
            self.roi_height_var,
            self.roi_channel_var,
            self.mask_channel_1_var,
            self.use_second_mask_var,
            self.mask_channel_2_var,
            self.target_channel_var,
            self.gaussian_sigma_var,
            self.threshold_method_var,
            self.manual_threshold_var,
            self.use_separate_thresholds_var,
            self.mask1_threshold_method_var,
            self.mask1_manual_threshold_var,
            self.mask2_threshold_method_var,
            self.mask2_manual_threshold_var,
            self.target_threshold_method_var,
            self.target_manual_threshold_var,
            self.min_object_area_var,
            self.max_object_area_var,
            self.limit_max_object_area_var,
            self.min_overlap_fraction_var,
            self.fill_holes_var,
            self.clear_border_var,
            self.mask_mode_var,
            self.histogram_source_var,
            self.histogram_log_scale_var,
            self.preview_channel,
            self.overlay_alpha_var,
            self.show_target_overlay_var,
            self.show_mask1_overlay_var,
            self.show_mask2_overlay_var,
            self.show_final_mask_overlay_var,
        ]:
            var.trace_add(
                "write",
                lambda *args: self.root.after(300, self.update_preview)
                )

    # -------------------------------------------------------------------------
    # GUI Layout
    # -------------------------------------------------------------------------

    def create_layout(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        settings_outer = ttk.Frame(main_pane, style="Panel.TFrame")
        preview_outer = ttk.Frame(main_pane, style="Panel.TFrame")

        main_pane.add(settings_outer, weight=2)
        main_pane.add(preview_outer, weight=3)

        self.create_settings_panel(settings_outer)
        self.create_preview_panel(preview_outer)

    def create_settings_panel(self, parent):
        canvas = tk.Canvas(
            parent,
            borderwidth=0,
            highlightthickness=0,
            bg=BG,
            bd=0,
        )
        
        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            command=canvas.yview,
            style="Vertical.TScrollbar",
        )

        self.settings_frame = ttk.Frame(canvas)

        self.settings_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.add_input_section()
        self.add_metadata_section()
        self.add_channel_section()
        self.add_roi_section()
        self.add_intensity_section()
        self.add_mask_section()
        self.add_processing_section()
        self.add_output_section()

    def section(self, title):
        frame = ttk.LabelFrame(self.settings_frame, text=title, padding=10)
        frame.pack(fill=tk.X, padx=10, pady=7)
        # Keep a consistent gap between descriptive labels in column 0 and
        # their controls in column 1 throughout the settings panel.
        frame.columnconfigure(0, pad=12)
        return frame

    def add_input_section(self):
        frame = self.section("1. Input folder and image scan")

        ttk.Label(
            frame,
            text="Root folder containing subfolders with .czi files",
        ).grid(row=0, column=0, sticky="w")

        ttk.Entry(
            frame,
            textvariable=self.root_folder_var,
            width=48,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ttk.Button(
            frame,
            text="Browse...",
            command=self.choose_root_folder,
        ).grid(row=1, column=1, sticky="ew")

        ttk.Button(
            frame,
            text="Scan .czi files",
            command=self.scan_files,
            style="Run.TButton",
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        
        self.file_count_label = ttk.Label(frame, text="No folder scanned yet.")
        self.file_count_label.grid(row=2, column=1, sticky="w", padx=(8, 0))

        frame.columnconfigure(0, weight=1)

    def add_metadata_section(self):
        frame = self.section("2. Folder hierarchy and metadata columns")

        ttk.Label(
            frame,
            text=(
                "Map subfolder levels below the root folder to CSV output columns."
                ),
            foreground=MUTED_TEXT,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Label(
            frame,
            text=(
                "Enter column names from top to bottom, separated by commas."
            ),
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="w")

        ttk.Entry(
            frame,
            textvariable=self.folder_levels_var,
            width=58,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(
            frame,
            text=(
                "No fixed limit; missing levels stay empty. Unnamed deeper levels are\n"
                "ignored."
            ),
            foreground=MUTED_TEXT,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky="w", pady=(5, 0))

        frame.columnconfigure(0, weight=1)

    def add_channel_section(self):
        frame = self.section("3. Define channel names")

        ttk.Label(
            frame,
            text="Enter the marker or signal measured in each image channel.",
            foreground=MUTED_TEXT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        for i, var in enumerate(self.channel_name_vars):
            ttk.Label(frame, text=f"Channel {i + 1}").grid(
                row=i + 1,
                column=0,
                sticky="w",
            )

            ttk.Entry(frame, textvariable=var, width=34).grid(
                row=i + 1,
                column=1,
                sticky="ew",
                pady=2,
            )

        frame.columnconfigure(1, weight=1)

    def add_roi_section(self):
        frame = self.section("4. ROI selection")

        methods = [
            "Centered ROI",
            "Highest signal in selected channel",
            "Lowest signal in selected channel",
            "Whole image",
        ]
        channels = [f"Channel {index}" for index in range(1, 5)]

        ttk.Label(frame, text="ROI method").grid(row=0, column=0, sticky="w")

        self.roi_method_box = ttk.Combobox(
            frame,
            textvariable=self.roi_method_var,
            values=methods,
            state="readonly",
        )
        self.roi_method_box.grid(row=0, column=1, sticky="ew")
        self.roi_method_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: (self.update_roi_controls(), self.update_preview()),
        )
        ttk.Button(
            frame,
            text="i",
            width=2,
            style="Info.TButton",
            command=lambda: self.show_info(
                "ROI method",
                "Choose how the region of interest is positioned. Centered ROI uses the "
                "specified width and height. Highest or lowest signal automatically finds "
                "a region in the selected signal channel. Whole image analyzes the full image.",
            ),
        ).grid(row=0, column=2, sticky="w", padx=(6, 0))

        self.roi_channel_label = ttk.Label(
            frame, text="Signal channel for automatic ROI"
        )
        self.roi_channel_label.grid(row=1, column=0, sticky="w")
        self.roi_channel_box = ttk.Combobox(
            frame,
            textvariable=self.roi_channel_var,
            values=channels,
            state="readonly",
        )
        self.roi_channel_box.grid(row=1, column=1, sticky="ew")
        self.roi_channel_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_preview()
        )

        self.roi_width_label = ttk.Label(frame, text="ROI width (px)")
        self.roi_width_label.grid(row=2, column=0, sticky="w")
        self.roi_width_entry = ttk.Entry(frame, textvariable=self.roi_width_var)
        self.roi_width_entry.grid(row=2, column=1, sticky="ew")

        self.roi_height_label = ttk.Label(frame, text="ROI height (px)")
        self.roi_height_label.grid(row=3, column=0, sticky="w")
        self.roi_height_entry = ttk.Entry(frame, textvariable=self.roi_height_var)
        self.roi_height_entry.grid(row=3, column=1, sticky="ew")

        ttk.Button(
            frame,
            text="Update preview overlay",
            command=self.update_preview,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        frame.columnconfigure(1, weight=1)
        self.update_roi_controls()
    
    def add_intensity_section(self):
        frame = self.section("5. Intensity channels")

        ttk.Label(
            frame,
            text="Measure mean ROI intensity for selected channels",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        for i, var in enumerate(self.intensity_channel_vars):
            ttk.Checkbutton(
                frame,
                text=f"Channel {i + 1}",
                variable=var,
                command=self.update_preview,
            ).grid(row=i + 1, column=0, columnspan=2, sticky="w")

        frame.columnconfigure(1, weight=1)
    
    def add_mask_section(self):
        frame = self.section("6. Mask mode and target signal")

        modes = [
            "No mask - ROI intensity only",
            "Single mask channel",
            "AND mask from two channels",
        ]
        channels = [f"Channel {index}" for index in range(1, 5)]

        ttk.Label(frame, text="Mask mode").grid(row=0, column=0, sticky="w")
        self.mask_mode_box = ttk.Combobox(
            frame,
            textvariable=self.mask_mode_var,
            values=modes,
            state="readonly",
        )
        self.mask_mode_box.grid(row=0, column=1, sticky="ew")
        self.mask_mode_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: (self.update_mask_mode_controls(), self.update_preview()),
        )
        ttk.Button(
            frame,
            text="i",
            width=2,
            style="Info.TButton",
            command=lambda: self.show_info(
                "Mask mode",
                "Choose whether the analysis uses no mask, one mask channel, or the "
                "overlap (AND) of two mask channels. The target channel is the signal "
                "measured inside the resulting region.",
            ),
        ).grid(row=0, column=2, sticky="w", padx=(6, 0))

        self.mask_channel_1_label = ttk.Label(frame, text="Mask channel 1")
        self.mask_channel_1_label.grid(row=1, column=0, sticky="w")
        self.mask_channel_1_box = ttk.Combobox(
            frame,
            textvariable=self.mask_channel_1_var,
            values=channels,
            state="readonly",
        )
        self.mask_channel_1_box.grid(row=1, column=1, sticky="ew")
        self.mask_channel_1_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_preview()
        )

        self.mask_channel_2_label = ttk.Label(frame, text="Mask channel 2")
        self.mask_channel_2_label.grid(row=2, column=0, sticky="w")
        self.mask_channel_2_box = ttk.Combobox(
            frame,
            textvariable=self.mask_channel_2_var,
            values=channels,
            state="readonly",
        )
        self.mask_channel_2_box.grid(row=2, column=1, sticky="ew")
        self.mask_channel_2_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_preview()
        )

        ttk.Label(frame, text="Target channel").grid(row=3, column=0, sticky="w")
        self.target_channel_box = ttk.Combobox(
            frame,
            textvariable=self.target_channel_var,
            values=channels,
            state="readonly",
        )
        self.target_channel_box.grid(row=3, column=1, sticky="ew")
        self.target_channel_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_preview()
        )

        ttk.Button(
            frame,
            text="Preview mask and target overlay",
            command=self.update_preview,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        frame.columnconfigure(1, weight=1)
        self.update_mask_mode_controls()

    def add_processing_section(self):
        frame = self.section("7. Object detection")
        methods = ["None - preview only", "Otsu", "Yen", "Triangle", "Li", "Manual"]

        def subgroup(parent, title, row):
            ttk.Label(parent, text=title, style="Subheading.TLabel").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(4 if row == 0 else 12, 5)
            )

        def info_button(parent, row, title, message):
            ttk.Button(
                parent,
                text="i",
                width=2,
                style="Info.TButton",
                command=lambda: self.show_info(title, message),
            ).grid(row=row, column=2, sticky="w", padx=(6, 0))

        # A. Background correction
        subgroup(frame, "Background correction", 0)
        ttk.Label(frame, text="Background smoothing radius (px)").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.gaussian_sigma_var).grid(
            row=1, column=1, sticky="ew"
        )
        info_button(
            frame,
            1,
            "Background smoothing radius",
            "Estimates and removes slowly varying background intensity before thresholding. "
            "Larger values account for broader background variations. Set to 0 to use the "
            "original image without background correction."
        )

        # B. Thresholding
        subgroup(frame, "Thresholding", 2)
        ttk.Label(frame, text="Threshold method").grid(row=3, column=0, sticky="w")
        self.shared_threshold_method_box = ttk.Combobox(
            frame,
            textvariable=self.threshold_method_var,
            values=methods,
            state="readonly",
        )
        self.shared_threshold_method_box.grid(row=3, column=1, sticky="ew")
        self.shared_threshold_method_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_processing_controls()
        )
        info_button(
            frame,
            3,
            "Threshold method",
            "Converts image intensities into detected foreground objects. Otsu, Yen, "
            "Triangle, and Li calculate the threshold automatically. Manual uses the "
            "entered value; None disables object detection for preview purposes.",
        )

        ttk.Label(frame, text="Manual threshold").grid(row=4, column=0, sticky="w")
        self.shared_manual_threshold_entry = ttk.Entry(
            frame, textvariable=self.manual_threshold_var
        )
        self.shared_manual_threshold_entry.grid(row=4, column=1, sticky="ew")

        ttk.Checkbutton(
            frame,
            text="Configure thresholds separately for each channel",
            variable=self.use_separate_thresholds_var,
            command=lambda: (self.update_processing_controls(), self.update_preview()),
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(7, 3))

        self.separate_threshold_widgets = []
        self.separate_threshold_rows = []
        separate_rows = [
            ("Primary mask", self.mask1_threshold_method_var, self.mask1_manual_threshold_var),
            ("Secondary mask", self.mask2_threshold_method_var, self.mask2_manual_threshold_var),
            ("Target", self.target_threshold_method_var, self.target_manual_threshold_var),
        ]

        for offset, (label, method_var, manual_var) in enumerate(separate_rows, start=6):
            label_widget = ttk.Label(frame, text=label)
            label_widget.grid(row=offset, column=0, sticky="w")
            row_frame = ttk.Frame(frame)
            row_frame.grid(row=offset, column=1, sticky="ew")
            row_frame.columnconfigure(0, weight=1)

            method_box = ttk.Combobox(
                row_frame,
                textvariable=method_var,
                values=methods,
                state="readonly",
                width=13,
            )
            method_box.grid(row=0, column=0, sticky="ew")
            manual_entry = ttk.Entry(row_frame, textvariable=manual_var, width=8)
            manual_entry.grid(row=0, column=1, sticky="e", padx=(6, 0))
            method_box.bind(
                "<<ComboboxSelected>>", lambda _event: self.update_processing_controls()
            )
            self.separate_threshold_widgets.extend([method_box, manual_entry])
            self.separate_threshold_rows.append(
                (label_widget, method_box, manual_entry)
            )

        # C. Object filtering
        subgroup(frame, "Object filtering", 9)
        ttk.Label(frame, text="Minimum object size (px²)").grid(row=10, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.min_object_area_var).grid(
            row=10, column=1, sticky="ew"
        )

        ttk.Checkbutton(
            frame,
            text="Limit maximum object size (px²)",
            variable=self.limit_max_object_area_var,
            command=lambda: (
                self.update_processing_controls(),
                self.update_preview(),
            ),
        ).grid(row=11, column=0, sticky="w")

        self.max_object_area_entry = ttk.Entry(
            frame,
            textvariable=self.max_object_area_var,
        )
        self.max_object_area_entry.grid(row=11, column=1, sticky="ew")

        ttk.Label(frame, text="Required mask overlap (%)").grid(row=12, column=0, sticky="w")
        self.min_overlap_percent_var = tk.DoubleVar(
            value=self.min_overlap_fraction_var.get() * 100.0
        )
        self.min_overlap_percent_var.trace_add("write", self.sync_overlap_percentage)
        ttk.Entry(frame, textvariable=self.min_overlap_percent_var).grid(
            row=12, column=1, sticky="ew"
        )
        info_button(
            frame,
            12,
            "Required mask overlap",
            "Sets the minimum percentage of a detected target object that must overlap "
            "the active mask to be counted. 0% requires no overlap; 100% requires the "
            "object to lie completely inside the mask.",
        )

        ttk.Checkbutton(
            frame,
            text="Fill holes inside masks",
            variable=self.fill_holes_var,
            command=self.update_preview,
        ).grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            frame,
            text="Exclude objects touching the ROI border",
            variable=self.clear_border_var,
            command=self.update_preview,
        ).grid(row=14, column=0, columnspan=3, sticky="w")

        frame.columnconfigure(1, weight=1)
        self.update_processing_controls()

    def add_output_section(self):
        frame = self.section("8. CSV export and analysis run")

        ttk.Label(frame, text="Output CSV file").grid(row=0, column=0, sticky="w")

        ttk.Entry(
            frame,
            textvariable=self.output_csv_var,
            width=48,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ttk.Button(
            frame,
            text="Save as...",
            command=self.choose_output_csv,
        ).grid(row=1, column=1, sticky="ew")

        self.run_button = ttk.Button(
            frame,
            text="Run analysis",
            command=self.run_analysis,
            style="Run.TButton",
        )
        
        self.run_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.status_label = ttk.Label(
            frame,
            text="Ready.",
            anchor="w",
            justify=tk.LEFT,
            wraplength=480,
        )
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew")

        frame.columnconfigure(0, weight=1)

    def create_preview_panel(self, parent):
        top = ttk.LabelFrame(parent, text="Live preview", padding=10)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = ttk.Frame(top)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Image").pack(side=tk.LEFT, padx=(0, 4))

        self.image_choice_var = tk.StringVar()
        self.image_choice_box = ttk.Combobox(
            controls,
            textvariable=self.image_choice_var,
            state="readonly",
            width=28,
        )
        self.image_choice_box.pack(side=tk.LEFT)
        self.image_choice_box.bind(
            "<<ComboboxSelected>>", self.choose_preview_from_dropdown
        )

        ttk.Button(
            controls,
            text="Previous",
            command=self.previous_preview,
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(
            controls,
            text="Next",
            command=self.next_preview,
        ).pack(side=tk.LEFT, padx=(5, 0))

        legend = ttk.Frame(top)
        legend.pack(anchor="w", fill=tk.X, pady=(8, 0))

        self.add_legend_item(legend, "#ff3b3b", "Target signal")
        self.add_legend_item(legend, "#00dcff", "Mask channel 1")
        self.add_legend_item(legend, "#aa46ff", "Mask channel 2")
        self.add_legend_item(legend, "#ffe600", "Final active mask")
        self.add_legend_item(legend, "#ffffff", "ROI border")

        ttk.Label(controls, text="Displayed image channel").pack(
            side=tk.LEFT, padx=(16, 4)
        )
        self.preview_channel_box = ttk.Combobox(
            controls,
            textvariable=self.preview_channel,
            values=[f"Channel {index}" for index in range(1, 5)],
            state="readonly",
            width=10,
        )
        self.preview_channel_box.pack(side=tk.LEFT)
        self.preview_channel_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_preview()
        )

        ttk.Button(
            controls,
            text="Export current preview",
            command=self.export_current_preview,
            style="Run.TButton",
        ).pack(side=tk.LEFT, padx=(8, 0))

        overlay_controls = ttk.Frame(top)
        overlay_controls.pack(anchor="w", fill=tk.X, pady=(8, 0))

        ttk.Label(overlay_controls, text="Overlay opacity").pack(side=tk.LEFT)
        ttk.Scale(
            overlay_controls,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.overlay_alpha_var,
            command=lambda _value: self.update_preview(),
            length=130,
        ).pack(side=tk.LEFT, padx=(8, 14))

        ttk.Checkbutton(
            overlay_controls,
            text="Target",
            variable=self.show_target_overlay_var,
            command=self.update_preview,
            style="Overlay.TCheckbutton",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            overlay_controls,
            text="Mask 1",
            variable=self.show_mask1_overlay_var,
            command=self.update_preview,
            style="Overlay.TCheckbutton",
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            overlay_controls,
            text="Mask 2",
            variable=self.show_mask2_overlay_var,
            command=self.update_preview,
            style="Overlay.TCheckbutton",
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            overlay_controls,
            text="Final mask",
            variable=self.show_final_mask_overlay_var,
            command=self.update_preview,
            style="Overlay.TCheckbutton",
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.preview_name_label = ttk.Label(top, text="No image loaded.")
        self.preview_name_label.pack(anchor="w", pady=(8, 4))

        self.preview_canvas = tk.Canvas(
            top,
            bg=CANVAS_BG,
            highlightthickness=0,
            bd=0,
        )
        
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", lambda event: self.update_preview())
        
        # Draggable upper edge for resizing the histogram area.
        self.histogram_resize_handle = tk.Canvas(
            top,
            height=8,
            bg=BG,
            highlightthickness=0,
            bd=0,
            cursor="sb_v_double_arrow",
        )
        self.histogram_resize_handle.pack(fill=tk.X, pady=(6, 0))
        self.histogram_resize_handle.bind(
            "<Configure>", self.draw_histogram_resize_handle
        )
        self.histogram_resize_handle.bind(
            "<ButtonPress-1>", self.start_histogram_resize
        )
        self.histogram_resize_handle.bind("<B1-Motion>", self.resize_histogram)
        self.histogram_resize_handle.bind(
            "<ButtonRelease-1>", self.finish_histogram_resize
        )

        # Threshold histogram
        histogram_frame = ttk.LabelFrame(top, text="Threshold histogram", padding=8)
        histogram_frame.pack(fill=tk.X)

        histogram_controls = ttk.Frame(histogram_frame)
        histogram_controls.pack(fill=tk.X)

        ttk.Label(histogram_controls, text="Displayed signal channel").pack(side=tk.LEFT)

        ttk.Combobox(
            histogram_controls,
            textvariable=self.histogram_source_var,
            values=[
                "ROI signal channel",
                "Mask channel 1",
                "Mask channel 2",
                "Target channel",
            ],
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.threshold_values_label = ttk.Label(
            histogram_controls,
            text="Thresholds: -",
            foreground=MUTED_TEXT,
        )
        self.threshold_values_label.pack(side=tk.LEFT, padx=(12, 0))

        histogram_view_controls = ttk.Frame(histogram_frame)
        histogram_view_controls.pack(fill=tk.X, pady=(8, 0))

        ttk.Checkbutton(
            histogram_view_controls,
            text="Log y",
            variable=self.histogram_log_scale_var,
            command=self.update_preview,
        ).pack(side=tk.LEFT)
        ttk.Button(
            histogram_view_controls,
            text="i",
            width=2,
            style="Info.TButton",
            command=lambda: self.show_info(
                "Log y",
                "Uses a logarithmic vertical axis so both frequent and rare intensity "
                "values remain visible. This changes only the histogram display; it does "
                "not change thresholds or analysis results.",
            ),
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.histogram_canvas = tk.Canvas(
            histogram_frame,
            height=self.histogram_height_var.get(),
            bg=CANVAS_BG,
            highlightthickness=0,
            bd=0,
        )
        self.histogram_canvas.pack(fill=tk.X, expand=False, pady=(8, 0))
    
    def add_legend_item(self, parent, color, text):
        item = ttk.Frame(parent)
        item.pack(side=tk.LEFT, padx=(0, 16), pady=2)

        dot = tk.Canvas(
            item,
            width=13,
            height=13,
            bg=BG,
            highlightthickness=0,
            bd=0,
        )
        dot.pack(side=tk.LEFT, padx=(0, 5))
        dot.create_oval(2, 2, 11, 11, fill=color, outline=color)

        ttk.Label(item, text=text, foreground=MUTED_TEXT).pack(side=tk.LEFT)
    
    def choose_preview_from_dropdown(self, _event=None):
        selected = self.image_choice_var.get()
        for index, path in enumerate(self.files):
            if selected == Path(path).name:
                self.preview_index.set(index)
                self.load_preview_image()
                return
    
    def show_info(self, title, message):
        messagebox.showinfo(title, message)

    @staticmethod
    def channel_number(channel_label):
        """Convert labels such as 'Channel 3' to their one-based number."""
        text = str(channel_label).strip()
        if text.lower().startswith("channel"):
            text = text[len("channel"):].strip()
        return int(text)

    def update_roi_controls(self):
        """Enable only ROI settings used by the selected ROI method."""
        method = self.roi_method_var.get()
        automatic_signal_roi = method in (
            "Highest signal in selected channel",
            "Lowest signal in selected channel",
        )
        size_is_used = method != "Whole image"

        self.roi_channel_label.configure(
            style="TLabel" if automatic_signal_roi else "Inactive.TLabel"
        )
        self.roi_channel_box.configure(
            state="readonly" if automatic_signal_roi else "disabled"
        )

        size_style = "TLabel" if size_is_used else "Inactive.TLabel"
        size_state = "normal" if size_is_used else "disabled"
        self.roi_width_label.configure(style=size_style)
        self.roi_height_label.configure(style=size_style)
        self.roi_width_entry.configure(state=size_state)
        self.roi_height_entry.configure(state=size_state)

    def update_mask_mode_controls(self):
        """Show which mask-channel selectors apply to the current mode."""
        mode = self.mask_mode_var.get()
        mask_1_active = mode in (
            "Single mask channel",
            "AND mask from two channels",
        )
        mask_2_active = mode == "AND mask from two channels"

        self.mask_channel_1_label.configure(
            style="TLabel" if mask_1_active else "Inactive.TLabel"
        )
        self.mask_channel_1_box.configure(
            state="readonly" if mask_1_active else "disabled"
        )

        self.mask_channel_2_label.configure(
            style="TLabel" if mask_2_active else "Inactive.TLabel"
        )
        self.mask_channel_2_box.configure(
            state="readonly" if mask_2_active else "disabled"
        )

    def sync_overlap_percentage(self, *_args):
        """Keep the GUI percentage compatible with the analysis setting."""
        try:
            self.min_overlap_fraction_var.set(
                float(self.min_overlap_percent_var.get()) / 100.0
            )
        except (tk.TclError, ValueError):
            # The entry may temporarily be empty while the user edits it.
            pass

    def update_processing_controls(self):
        """Enable only controls relevant to the selected detection settings."""
        separate = bool(self.use_separate_thresholds_var.get())

        self.shared_threshold_method_box.configure(
            state="disabled" if separate else "readonly"
        )
        self.shared_manual_threshold_entry.configure(
            state="normal"
            if not separate and self.threshold_method_var.get() == "Manual"
            else "disabled"
        )
        
        for label_widget, method_box, manual_entry in self.separate_threshold_rows:
            label_widget.configure(
                style="TLabel" if separate else "Inactive.TLabel"
            )
            method_box.configure(state="readonly" if separate else "disabled")
            manual_entry.configure(
                state="normal"
                if separate and method_box.get() == "Manual"
                else "disabled"
            )

        self.max_object_area_entry.configure(
            state="normal" if self.limit_max_object_area_var.get() else "disabled"
        )

    def update_separate_threshold_controls(self):
        if not hasattr(self, "separate_threshold_rows"):
            return
        self.update_processing_controls()

    def draw_histogram_resize_handle(self, event=None):
        """Draw the subtle horizontal grip above the histogram."""
        handle = self.histogram_resize_handle
        handle.delete("all")
        width = event.width if event is not None else handle.winfo_width()
        handle.create_line(
            0,
            4,
            max(1, width),
            4,
            fill=BUTTON_ACTIVE,
            width=2,
        )

    def start_histogram_resize(self, event):
        self._histogram_resize_start_y = event.y_root
        self._histogram_resize_start_height = self.histogram_canvas.winfo_height()

    def resize_histogram(self, event):
        """Resize upward to enlarge and downward to reduce the histogram."""
        if getattr(self, "_histogram_resize_start_y", None) is None:
            return

        distance = self._histogram_resize_start_y - event.y_root
        new_height = self._histogram_resize_start_height + distance
        new_height = max(100, min(500, int(new_height)))
        self.histogram_height_var.set(new_height)
        self.histogram_canvas.configure(height=new_height)

    def finish_histogram_resize(self, _event=None):
        self._histogram_resize_start_y = None
        self._histogram_resize_start_height = self.histogram_canvas.winfo_height()
        self.update_preview()

    def update_histogram_height(self, _value=None):
        if hasattr(self, "histogram_canvas"):
            self.histogram_canvas.configure(height=int(self.histogram_height_var.get()))
        self.update_preview()
    
    # -------------------------------------------------------------------------
    # File and Preview Actions
    # -------------------------------------------------------------------------

    def choose_root_folder(self):
        folder = filedialog.askdirectory(title="Select root folder with CZI files")
        if folder:
            self.root_folder_var.set(folder)

    def choose_output_csv(self):
        file_path = filedialog.asksaveasfilename(
            title="Choose output CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self.output_csv_var.set(file_path)

    def scan_files(self):
        root_folder = self.root_folder_var.get().strip()

        if not root_folder or not os.path.isdir(root_folder):
            messagebox.showerror("Input folder missing", "Please choose a valid root folder.")
            return

        self.files = sorted(str(path) for path in Path(root_folder).rglob("*.czi"))
        
        if hasattr(self, "image_choice_box"):
            names = [Path(path).name for path in self.files]
            self.image_choice_box["values"] = names
            if names:
                self.image_choice_var.set(names[0])
        
        self.file_count_label.configure(text=f"{len(self.files)} .czi files found.")
        self.preview_index.set(0)

        if self.files:
            self.load_preview_image()
        else:
            self.preview_name_label.configure(text="No .czi files found.")
        
    def load_preview_image(self):
        if not self.files:
            return

        index = max(0, min(self.preview_index.get(), len(self.files) - 1))
        self.preview_index.set(index)

        try:
            self.preview_image_cyx = load_czi_as_cyx(self.files[index])
            channel_count = self.preview_image_cyx.shape[0]
            if hasattr(self, "preview_channel_box"):
                self.preview_channel_box.configure(
                    values=[f"Channel {number}" for number in range(1, channel_count + 1)]
                )
            selected_preview_channel = clamp_channel(
                self.channel_number(self.preview_channel.get()) - 1,
                channel_count,
            ) + 1
            self.preview_channel.set(f"Channel {selected_preview_channel}")
            self.preview_name_label.configure(
                text=f"{index + 1}/{len(self.files)}: "
                f"{Path(self.files[index]).name} "
                f"({channel_count} channels)"
            )
            self.update_preview()
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def previous_preview(self):
        if self.files:
            self.preview_index.set(max(0, self.preview_index.get() - 1))
            self.load_preview_image()

    def next_preview(self):
        if self.files:
            self.preview_index.set(min(len(self.files) - 1, self.preview_index.get() + 1))
            self.load_preview_image()

    def export_current_preview(self):
        if self.preview_image_cyx is None or not self.files:
            messagebox.showerror("No preview", "Please load a preview image first.")
            return

        self.update_preview()

        if self.current_preview_rgb is None:
            messagebox.showerror("No preview", "The current preview could not be exported.")
            return

        output_csv = self.output_csv_var.get().strip()
        if output_csv:
            export_dir = Path(output_csv).parent / "preview_exports"
        else:
            export_dir = Path.cwd() / "preview_exports"

        export_dir.mkdir(parents=True, exist_ok=True)

        index = max(0, min(self.preview_index.get(), len(self.files) - 1))
        image_name = Path(self.files[index]).stem
        if self.use_separate_thresholds_var.get():
            method_name = "separate_thresholds"
        else:
            method_name = clean_filename_part(self.threshold_method_var.get())
        source_name = clean_filename_part(self.histogram_source_var.get())
        channel_name = f"ch{self.channel_number(self.preview_channel.get())}"
        mask_name = f"mask{self.channel_number(self.mask_channel_1_var.get())}"
        if self.mask_mode_var.get() == "AND mask from two channels":
            mask_name += f"_{self.channel_number(self.mask_channel_2_var.get())}"

        filename = f"{image_name}_{channel_name}_{mask_name}_{method_name}_{source_name}_preview.png"
        output_path = export_dir / filename

        try:
            settings = self.collect_settings(validate_paths=False)
            mask1_method, _mask1_manual = get_mask1_threshold_settings(settings)
            mask2_method, _mask2_manual = get_mask2_threshold_settings(settings)
            target_method, _target_manual = get_target_threshold_settings(settings)
            overlays_are_visible = settings.mask_mode != "No mask - ROI intensity only"
            export_image = add_preview_export_footer(
                self.current_preview_rgb,
                settings,
                preview_channel=self.preview_channel.get(),
                overlay_states={
                    "Target": overlays_are_visible and not thresholding_is_disabled(target_method) and self.show_target_overlay_var.get(),
                    "Mask 1": overlays_are_visible and not thresholding_is_disabled(mask1_method) and self.show_mask1_overlay_var.get(),
                    "Mask 2": overlays_are_visible and settings.mask_mode == "AND mask from two channels" and not thresholding_is_disabled(mask2_method) and self.show_mask2_overlay_var.get(),
                    "Final mask": overlays_are_visible and self.show_final_mask_overlay_var.get(),
                },
            )
        except Exception:
            export_image = Image.fromarray(self.current_preview_rgb.astype(np.uint8), mode="RGB")

        export_image.save(output_path)
        self.status_label.configure(text="Preview exported. Ready for further changes.")
        messagebox.showinfo("Preview exported", f"Saved preview image:\n{output_path}")

    def update_preview(self):
        try:
            if self.preview_image_cyx is None:
                return

            settings = self.collect_settings(validate_paths=False)
            image = self.preview_image_cyx
            channels = image.shape[0]

            preview_channel_index = clamp_channel(
                self.channel_number(self.preview_channel.get()) - 1,
                channels,
            )
            rgb = make_rgb_preview(image, preview_channel_index)

            roi_x, roi_y, roi_w, roi_h = calculate_roi(
                image,
                settings.roi_method,
                settings.roi_width,
                settings.roi_height,
                settings.roi_channel,
            )
            
            roi_slice = np.s_[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            overlay = rgb[roi_slice].copy()
            overlay_alpha = max(0.0, min(float(self.overlay_alpha_var.get()), 1.0))

            if (
                settings.mask_mode != "No mask - ROI intensity only"
            ):
                mask1_method, mask1_manual = get_mask1_threshold_settings(settings)
                mask2_method, mask2_manual = get_mask2_threshold_settings(settings)
                target_method, target_manual = get_target_threshold_settings(settings)

                mask1 = build_mask(
                    image[clamp_channel(settings.mask_channel_1, channels)][roi_slice],
                    settings,
                    mask1_method,
                    mask1_manual,
                )

                final_mask = mask1
                mask2 = None

                if settings.mask_mode == "AND mask from two channels":
                    mask2 = build_mask(
                        image[clamp_channel(settings.mask_channel_2, channels)][roi_slice],
                        settings,
                        mask2_method,
                        mask2_manual,
                    )
                    final_mask = mask1 & mask2
    
                target = subtract_background(
                    image[clamp_channel(settings.target_channel, channels)][roi_slice],
                    settings.gaussian_sigma,
                )

                target_binary = threshold_image(
                    target,
                    target_method,
                    target_manual,
                )
                target_binary = clean_mask(target_binary, settings)

                if self.show_target_overlay_var.get():
                    overlay[target_binary] = blend_color(overlay[target_binary], (255, 0, 0), overlay_alpha)

                if self.show_mask1_overlay_var.get():
                    overlay[mask1] = blend_color(overlay[mask1], (0, 220, 255), overlay_alpha)

                if mask2 is not None and self.show_mask2_overlay_var.get():
                    overlay[mask2] = blend_color(overlay[mask2], (170, 70, 255), overlay_alpha)

                if self.show_final_mask_overlay_var.get():
                    overlay[final_mask] = blend_color(overlay[final_mask], (255, 230, 0), overlay_alpha)

            rgb[roi_slice] = overlay

            draw_rectangle(rgb, roi_x, roi_y, roi_w, roi_h, (255, 255, 255), 2)
            self.current_preview_rgb = rgb.copy()
            self.show_rgb_on_canvas(rgb)
            self.update_histogram(settings)

        except Exception as e:
            print("Preview update failed:", e)

            if self.preview_image_cyx is not None:
                channel_index = clamp_channel(
                    self.channel_number(self.preview_channel.get()) - 1,
                    self.preview_image_cyx.shape[0],
                )
                self.show_rgb_on_canvas(make_rgb_preview(self.preview_image_cyx, channel_index))

            try:
                self.update_histogram(self.collect_settings(validate_paths=False))
            except Exception as hist_error:
                print("Histogram update failed:", hist_error)

    def show_rgb_on_canvas(self, rgb):
        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())

        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS

        image.thumbnail((canvas_width, canvas_height), resample)

        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_canvas.delete("all")

        x = (canvas_width - image.width) // 2
        y = (canvas_height - image.height) // 2

        self.preview_canvas.create_image(x, y, anchor="nw", image=self.preview_photo)
    
    def get_histogram_channel_index(self, settings, channel_count):
        source = self.histogram_source_var.get()

        if source == "ROI signal channel":
            return clamp_channel(settings.roi_channel, channel_count)

        if source == "Mask channel 1":
            return clamp_channel(settings.mask_channel_1, channel_count)

        if source == "Mask channel 2":
            return clamp_channel(settings.mask_channel_2, channel_count)

        return clamp_channel(settings.target_channel, channel_count)

    def get_histogram_threshold_settings(self, settings):
        source = self.histogram_source_var.get()

        if source == "Mask channel 1":
            return get_mask1_threshold_settings(settings)

        if source == "Mask channel 2":
            return get_mask2_threshold_settings(settings)

        if source == "Target channel":
            return get_target_threshold_settings(settings)

        return settings.threshold_method, settings.manual_threshold

    def update_histogram(self, settings):
        if self.preview_image_cyx is None:
            return
        if not hasattr(self, "histogram_canvas"):
            return

        image = self.preview_image_cyx
        channels, _h, _w = image.shape

        channel_index = self.get_histogram_channel_index(settings, channels)

        roi_x, roi_y, roi_w, roi_h = calculate_roi(
            image,
            settings.roi_method,
            settings.roi_width,
            settings.roi_height,
            settings.roi_channel,
        )

        roi_slice = (slice(roi_y, roi_y + roi_h), slice(roi_x, roi_x + roi_w))

        channel_image = image[channel_index][roi_slice]
        corrected = subtract_background(channel_image, settings.gaussian_sigma)

        self.draw_histogram(corrected, settings)

    def draw_histogram(self, image, settings):
        canvas = self.histogram_canvas
        canvas.delete("all")

        width = max(10, canvas.winfo_width())
        height = max(10, canvas.winfo_height())

        margin_left = 56
        margin_right = 12
        margin_top = 12
        margin_bottom = 42

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = np.asarray(image, dtype=np.float32)
        values = values[np.isfinite(values)]

        if values.size == 0:
            self.update_threshold_values_label({}, settings.threshold_method)
            return

        selected_method, selected_manual = self.get_histogram_threshold_settings(settings)

        thresholds = calculate_threshold_lines(
            values,
            selected_manual,
        )
        self.update_threshold_values_label(thresholds, selected_method)

        low, high = np.percentile(values, (0.5, 99.5))
        if high <= low:
            low = float(np.min(values))
            high = float(np.max(values)) or 1.0

        clipped = np.clip(values, low, high)
        counts, edges = np.histogram(clipped, bins=80, range=(low, high))

        max_count = max(1, int(np.max(counts)))
        use_log_scale = bool(self.histogram_log_scale_var.get())
        if use_log_scale:
            display_counts = np.log1p(counts)
            display_max = max(1e-9, float(np.max(display_counts)))
            y_axis_label = "Pixel count (log)"
        else:
            display_counts = counts.astype(np.float32)
            display_max = float(max_count)
            y_axis_label = "Pixel count"

        canvas.create_line(
            margin_left,
            margin_top + plot_h,
            margin_left + plot_w,
            margin_top + plot_h,
            fill=MUTED_TEXT,
        )
        canvas.create_line(
            margin_left,
            margin_top,
            margin_left,
            margin_top + plot_h,
            fill=MUTED_TEXT,
        )

        for fraction in (0.0, 0.5, 1.0):
            y = margin_top + plot_h - fraction * plot_h
            if use_log_scale:
                label_value = int(round(np.expm1(display_max * fraction)))
            else:
                label_value = int(round(max_count * fraction))
            canvas.create_line(
                margin_left - 4,
                y,
                margin_left,
                y,
                fill=MUTED_TEXT,
            )
            canvas.create_text(
                margin_left - 7,
                y,
                text=str(label_value),
                fill=MUTED_TEXT,
                anchor="e",
                font=("Segoe UI", 8),
            )

        bar_w = plot_w / len(counts)

        for i, display_count in enumerate(display_counts):
            x1 = margin_left + i * bar_w
            x2 = x1 + max(1, bar_w - 1)
            y2 = margin_top + plot_h
            y1 = y2 - (float(display_count) / display_max) * plot_h

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="#3a3f47",
                outline="",
            )

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = margin_left + fraction * plot_w
            value = low + fraction * (high - low)
            canvas.create_line(
                x,
                margin_top + plot_h,
                x,
                margin_top + plot_h + 4,
                fill=MUTED_TEXT,
            )
            canvas.create_text(
                x,
                margin_top + plot_h + 7,
                text=f"{value:.1f}",
                fill=MUTED_TEXT,
                anchor="n",
                font=("Segoe UI", 8),
            )

        colors = {
            "Otsu": "#18c7b7",
            "Yen": "#7aa2f7",
            "Triangle": "#ffe600",
            "Li": "#aa46ff",
            "Manual": "#ff3b3b",
        }

        for label, value in thresholds.items():
            if value < low:
                x = margin_left
                shown_label = label + "<"
            elif value > high:
                x = margin_left + plot_w
                shown_label = label + ">"
            else:
                x = margin_left + ((value - low) / (high - low)) * plot_w
                shown_label = label

            color = colors.get(label, "#ffffff")

            canvas.create_line(
                x,
                margin_top,
                x,
                margin_top + plot_h,
                fill=color,
                width=2,
            )
            canvas.create_text(
                x + 4,
                margin_top + 8,
                text=shown_label,
                fill=color,
                anchor="nw",
                font=("Segoe UI", 8),
            )

        canvas.create_text(
            margin_left,
            height - 5,
            text="Intensity",
            fill=MUTED_TEXT,
            anchor="sw",
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            6,
            margin_top,
            text=y_axis_label,
            fill=MUTED_TEXT,
            anchor="nw",
            font=("Segoe UI", 8),
        )

    def update_threshold_values_label(self, thresholds, selected_method):
        if not hasattr(self, "threshold_values_label"):
            return

        if not thresholds:
            self.threshold_values_label.configure(text="Thresholds: -")
            return

        parts = []

        for label in ("Otsu", "Yen", "Triangle", "Li", "Manual"):
            if label not in thresholds:
                continue

            value_text = f"{thresholds[label]:.2f}"
            if label == selected_method:
                value_text = f"[{value_text}]"
            parts.append(f"{label}: {value_text}")

        self.threshold_values_label.configure(text="Thresholds: " + "  ".join(parts))

    # -------------------------------------------------------------------------
    # Run Analysis
    # -------------------------------------------------------------------------

    def collect_settings(self, validate_paths=True):
        root_folder = self.root_folder_var.get().strip()
        output_csv = self.output_csv_var.get().strip()

        if validate_paths:
            if not root_folder or not os.path.isdir(root_folder):
                raise ValueError("Please choose a valid root folder.")
            if not output_csv:
                raise ValueError("Please choose an output CSV file.")

        level_names = [
            part.strip()
            for part in self.folder_levels_var.get().split(",")
            if part.strip()
        ]

        channel_names = {
            index: var.get().strip() or f"Channel {index + 1}"
            for index, var in enumerate(self.channel_name_vars)
        }

        return Settings(
            root_folder=root_folder,
            output_csv=output_csv,
            folder_level_names=level_names,
            channel_names=channel_names,
            roi_method=self.roi_method_var.get(),
            roi_width=int(self.roi_width_var.get()),
            roi_height=int(self.roi_height_var.get()),
            mask_mode=self.mask_mode_var.get(),
            intensity_channels=[
                index for index, var in enumerate(self.intensity_channel_vars)
                if var.get()
            ],
            roi_channel=self.channel_number(self.roi_channel_var.get()) - 1,
            mask_channel_1=self.channel_number(self.mask_channel_1_var.get()) - 1,
            use_second_mask=bool(self.use_second_mask_var.get()),
            mask_channel_2=self.channel_number(self.mask_channel_2_var.get()) - 1,
            target_channel=self.channel_number(self.target_channel_var.get()) - 1,
            gaussian_sigma=float(self.gaussian_sigma_var.get()),
            threshold_method=self.threshold_method_var.get(),
            manual_threshold=float(self.manual_threshold_var.get()),
            use_separate_thresholds=bool(self.use_separate_thresholds_var.get()),
            mask1_threshold_method=self.mask1_threshold_method_var.get(),
            mask1_manual_threshold=float(self.mask1_manual_threshold_var.get()),
            mask2_threshold_method=self.mask2_threshold_method_var.get(),
            mask2_manual_threshold=float(self.mask2_manual_threshold_var.get()),
            target_threshold_method=self.target_threshold_method_var.get(),
            target_manual_threshold=float(self.target_manual_threshold_var.get()),
            min_object_area=int(self.min_object_area_var.get()),
            max_object_area=(
                int(self.max_object_area_var.get())
                if self.limit_max_object_area_var.get()
                else 0
            ),
            min_overlap_fraction=float(self.min_overlap_fraction_var.get()),
            fill_holes=bool(self.fill_holes_var.get()),
            clear_border=bool(self.clear_border_var.get()),
        )

    def run_analysis(self):
        if self.is_running:
            return

        try:
            settings = self.collect_settings(validate_paths=True)
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            return

        if not self.files:
            self.scan_files()

        if not self.files:
            messagebox.showerror("No images", "No .czi files were found.")
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.run_animation_step = 0
        self.animate_run_button()
        self.progress.configure(maximum=len(self.files), value=0)
        self.status_label.configure(text="Analysis running...")

        worker = threading.Thread(
            target=self.analysis_worker,
            args=(settings, list(self.files)),
            daemon=True,
        )
        worker.start()

    def analysis_worker(self, settings, files):
        rows = []

        try:
            total_files = len(files)
            for index, path in enumerate(files, start=1):
                self.status_queue.put(
                    f"PROGRESS|{index - 1}|Analyzing image {index} of {total_files}..."
                )

                rows.append(analyze_one_image(path, settings))

                self.status_queue.put(
                    f"PROGRESS|{index}|Finished image {index} of {total_files}."
                )

            write_csv(settings.output_csv, rows)

            self.status_queue.put(
                f"DONE|Analysis complete. CSV saved: {settings.output_csv}"
            )

        except Exception:
            self.status_queue.put("ERROR|" + traceback.format_exc())

    def poll_status_queue(self):
        try:
            while True:
                message = self.status_queue.get_nowait()
                parts = message.split("|", 2)

                if parts[0] == "PROGRESS":
                    self.progress.configure(value=int(parts[1]))
                    self.status_label.configure(text=parts[2])

                elif parts[0] == "DONE":
                    self.is_running = False
                    self.run_button.configure(state="normal", text="Run analysis")
                    self.run_animation_step = 0
                    self.progress.configure(value=self.progress["maximum"])
                    self.status_label.configure(
                        text="Analysis complete. Ready for a new run."
                    )
                    messagebox.showinfo("Analysis complete", parts[1])

                elif parts[0] == "ERROR":
                    self.is_running = False
                    self.run_button.configure(state="normal", text="Run analysis")
                    self.run_animation_step = 0
                    self.status_label.configure(text="Analysis failed.")
                    messagebox.showerror("Analysis error", parts[1])

        except queue.Empty:
            pass

        self.root.after(200, self.poll_status_queue)
    
    def animate_run_button(self):
        if not self.is_running:
            self.run_button.configure(text="Run analysis")
            return

        dots = "." * ((self.run_animation_step % 3) + 1)
        self.run_button.configure(text="Analyzing" + dots)
        self.run_animation_step += 1
        self.root.after(450, self.animate_run_button)


# =============================================================================
# Drawing and CSV Export
# =============================================================================


def add_preview_export_footer(rgb, settings, preview_channel, overlay_states):
    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")

    footer_height = 150
    output = Image.new(
        "RGB",
        (base.width, base.height + footer_height),
        (24, 25, 29),
    )
    output.paste(base, (0, 0))

    draw = ImageDraw.Draw(output)

    try:
        export_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        export_font = ImageFont.load_default()

    y0 = base.height

    draw.rectangle(
        [0, y0, base.width, y0 + footer_height],
        fill=(24, 25, 29),
    )
    draw.line(
        [0, y0, base.width, y0],
        fill=(238, 238, 238),
        width=1,
    )

    mask_channels = f"Mask channels: 1={settings.mask_channel_1 + 1}"

    if settings.mask_mode == "AND mask from two channels":
        mask_channels += f", 2={settings.mask_channel_2 + 1}"
    else:
        mask_channels += ", 2=not used"

    if settings.use_separate_thresholds:
        threshold_text = (
            "Thresholds: "
            f"M1={settings.mask1_threshold_method}, "
            f"M2={settings.mask2_threshold_method}, "
            f"Target={settings.target_threshold_method}"
        )
    else:
        threshold_text = f"Threshold method: {settings.threshold_method}"

    lines = [
        f"Preview channel: {preview_channel}   "
        f"{threshold_text}   "
        f"Target channel: {settings.target_channel + 1}",
        f"{mask_channels}   Mask mode: {settings.mask_mode}",
    ]

    text_x = 12
    text_y = y0 + 10

    for line in lines:
        draw.text(
            (text_x, text_y),
            line,
            fill=(238, 238, 238),
            font=export_font,
        )
        text_y += 30

    legend_items = [
        ("Target", (255, 59, 59)),
        ("Mask 1", (0, 220, 255)),
        ("Mask 2", (170, 70, 255)),
        ("Final mask", (255, 230, 0)),
        ("ROI border", (255, 255, 255)),
    ]

    x = text_x
    y = y0 + 90

    for label, color in legend_items:
        if label != "ROI border" and not overlay_states.get(label, False):
            continue

        draw.rectangle(
            [x, y + 2, x + 20, y + 22],
            fill=color,
            outline=color,
        )
        draw.text(
            (x + 26, y),
            label,
            fill=(238, 238, 238),
            font=export_font,
        )

        x += 165

    return output

def blend_color(pixels, color, alpha):
    color_array = np.asarray(color, dtype=np.float32)
    return (
        pixels.astype(np.float32) * (1.0 - alpha)
        + color_array * alpha
    ).astype(np.uint8)


def draw_rectangle(rgb, x, y, width, height, color, thickness):
    h, w, _ = rgb.shape

    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(0, min(x + width - 1, w - 1))
    y2 = max(0, min(y + height - 1, h - 1))

    rgb[y1:y1 + thickness, x1:x2 + 1] = color
    rgb[y2 - thickness + 1:y2 + 1, x1:x2 + 1] = color
    rgb[y1:y2 + 1, x1:x1 + thickness] = color
    rgb[y1:y2 + 1, x2 - thickness + 1:x2 + 1] = color
    
def calculate_threshold_lines(values, manual_threshold):
    result = {}

    if filters is None:
        result["Manual"] = manual_threshold
        return result

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return result

    try:
        result["Otsu"] = float(filters.threshold_otsu(values))
    except Exception:
        pass

    try:
        result["Yen"] = float(filters.threshold_yen(values))
    except Exception:
        pass

    try:
        result["Triangle"] = float(filters.threshold_triangle(values))
    except Exception:
        pass

    try:
        result["Li"] = float(filters.threshold_li(values))
    except Exception:
        pass

    result["Manual"] = float(manual_threshold)

    return result

def clean_column_name(name):
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(name)
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "channel"


def clean_filename_part(name):
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(name)
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "value"

def format_csv_value(value):
    """Format numeric values for German-style CSV import in R/Excel."""
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return ""
        text = f"{float(value):.10g}"
        return text.replace(".", ",")

    if isinstance(value, (np.integer, int)):
        return int(value)

    return value

def format_csv_row(row):
    return {
        key: format_csv_value(value)
        for key, value in row.items()
    }

def write_csv(output_csv, rows):
    if not rows:
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_order = [
        "image_name",
        "image_path",
        "mask_channel_1",
        "mask_channel_2",
        "target_channel",
        "roi_method",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "threshold_mode",
        "mask1_threshold_method",
        "mask1_manual_threshold",
        "mask2_threshold_method",
        "mask2_manual_threshold",
        "target_threshold_method",
        "target_manual_threshold",
        "object_count",
        "object_area_px",
        "object_mean_intensity",
        "mean_intensity_in_mask",
        "median_intensity_in_mask",
        "mean_intensity_roi",
        "median_intensity_roi",
        "jaccard_index",
        "target_signal_fraction_in_mask",
        "mask_area_percent",
        "target_area_percent_in_mask",
    ]

    all_keys = []

    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    folder_keys = [key for key in all_keys if key not in preferred_order]
    
    fieldnames = (
        ["image_name", "image_path"]
        + folder_keys
        + [key for key in preferred_order if key in all_keys and key not in ("image_name", "image_path")]
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(format_csv_row(row) for row in rows)


# =============================================================================
# Program Start
# =============================================================================

def main():
    root = tk.Tk()
    
    try:
        icon_path = Path(__file__).with_name("icon.png")
        icon_image = tk.PhotoImage(file=str(icon_path))
        root.iconphoto(True, icon_image)
        root.icon_image = icon_image
    except Exception:
        pass
    
    root.configure(bg=BG)

    style = ttk.Style(root)

    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        ".",
        background=BG,
        foreground=TEXT,
        fieldbackground=ENTRY_BG,
        font=BASE_FONT,
        borderwidth=0,
        relief="flat",
    )

    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL_BG)

    style.configure(
        "TLabelframe",
        background=BG,
        foreground=TEXT,
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=TEXT,
        font=TITLE_FONT,
    )

    style.configure("TLabel", background=BG, foreground=TEXT, font=BASE_FONT)
    style.configure(
        "Inactive.TLabel",
        background=BG,
        foreground=INACTIVE_TEXT,
        font=BASE_FONT,
    )
    style.configure(
        "Subheading.TLabel",
        background=BG,
        foreground=TEXT,
        font=SUBTITLE_FONT,
    )

    style.configure(
        "TButton",
        background=BUTTON_BG,
        foreground=TEXT,
        padding=(10, 7),
        font=BUTTON_FONT,
        borderwidth=0,
        focusthickness=0,
    )
    style.map(
        "TButton",
        background=[("active", BUTTON_ACTIVE), ("pressed", ACCENT_DARK)],
        foreground=[("active", "#ffffff")],
    )
    style.configure(
        "Info.TButton",
        background=BUTTON_BG,
        foreground=TEXT,
        padding=(4, 2),
        font=("Segoe UI", 9, "bold"),
    )

    style.configure(
        "Run.TButton",
        background=ACCENT_DARK,
        foreground="#ffffff",
        padding=(12, 8),
        font=("Segoe UI", 11, "bold"),
    )
    style.map(
        "Run.TButton",
        background=[("active", ACCENT), ("pressed", ACCENT_DARK)],
    )

    style.configure(
        "TEntry",
        fieldbackground=ENTRY_BG,
        foreground=TEXT,
        insertcolor=TEXT,
        padding=4,
        borderwidth=0,
    )

    style.configure(
        "TCombobox",
        fieldbackground=ENTRY_BG,
        background=ENTRY_BG,
        foreground=TEXT,
        arrowcolor=TEXT,
        padding=4,
        borderwidth=0,
    )

    style.configure(
        "TCheckbutton",
        background=BG,
        foreground=TEXT,
        font=BASE_FONT,
    )
    style.configure(
        "Overlay.TCheckbutton",
        background=BG,
        foreground=TEXT,
        font=BASE_FONT,
        indicatorsize=17,
        padding=(3, 3),
    )

    style.configure(
        "TProgressbar",
        background=ACCENT,
        troughcolor=ENTRY_BG,
        borderwidth=0,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
    )

    style.configure(
        "Vertical.TScrollbar",
        background="#30343a",
        troughcolor=BG,
        bordercolor=BG,
        arrowcolor=TEXT,
        relief="flat",
        borderwidth=0,
        width=14,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", ACCENT), ("pressed", ACCENT_DARK)],
    )

    App(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()
    









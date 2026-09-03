# MasQ: CZI ROI Mask Analyzer

MasQ is a graphical Python application for mask-based quantitative analysis of
multichannel fluorescence microscopy images stored in Carl Zeiss Image (CZI)
format. It was developed for reproducible analysis of fluorescence intensities,
objects, and spatial overlap within user-defined regions of interest (ROIs).

MasQ processes original image data for quantitative measurements. Automatic
contrast adjustment is applied only to the graphical preview and does not alter
the underlying data used for analysis.

## Features

- Recursive discovery of `.czi` files within a selected root directory
- Import of metadata encoded by the input folder hierarchy
- Standardization of image data to channel-by-height-by-width (`C x Y x X`)
- Whole-image, centered, highest-signal, or lowest-signal ROI selection
- Gaussian background subtraction
- Shared or channel-specific threshold settings
- Otsu, Yen, Triangle, Li, and manual thresholding
- Single-channel masks or logical intersections of two channel masks
- Optional hole filling, border clearing, and object-size filtering
- Selection of one target channel and multiple intensity channels
- Measurement of mean and median intensities within the ROI and final mask
- Target-object count, combined object area, and mean object intensity
- Minimum-overlap filtering for target objects associated with the final mask
- Jaccard index and additional mask/target overlap measurements
- Color-coded previews and exportable preview images
- Semicolon-delimited CSV output with decimal commas
- Graphical interface with background processing and progress reporting

## Requirements

MasQ was developed with Python 3.10.11 and the following packages:

- NumPy 2.2.6
- Pillow 11.0.0
- SciPy 1.15.3
- scikit-image 0.25.2
- aicspylibczi 3.3.1

The graphical interface uses Tkinter, which is included with most standard
Python installations. The application was developed and executed in Thonny,
but Thonny is not required; MasQ can also be started with a compatible Python
interpreter.

The program includes a fallback reader based on `czifile`. This optional
package can be installed if `aicspylibczi` cannot read a particular file or is
not available in the local environment:

```text
pip install czifile
```

## Installation

### Installation in Thonny

1. Install Python 3.10.11 and Thonny.
2. Open **Tools > Manage packages** in Thonny.
3. Install `numpy`, `Pillow`, `scipy`, `scikit-image`, and `aicspylibczi`.
4. Open `masq.py` in Thonny and run the script.

### Installation with pip

Download the repository, open a terminal in its directory, and install the
required packages:

```text
python -m pip install -r requirements.txt
```

Then start the application:

```text
python masq.py
```

## Input data

MasQ expects one or more multichannel fluorescence images in CZI format. After
a root directory is selected, all `.czi` files in that directory and its
subdirectories are identified recursively.

The names of folder levels can be configured in the interface. Information
encoded in the folder structure, such as the biological replicate,
experimental construct, or coverslip, is then transferred to separate columns
in the output file.

MasQ analyzes a two-dimensional image plane with separate fluorescence
channels. If a CZI file contains additional dimensions, the program reduces
unsupported dimensions by selecting their first position. Users should verify
the imported image and channel assignment in the preview before analysis.

## Basic workflow

1. Start `masq.py`.
2. Select the root folder containing the CZI images.
3. Select the destination and name of the output CSV file.
4. Define names for the relevant folder levels and fluorescence channels.
5. Select the ROI method and, where applicable, its width, height, and channel.
6. Choose whether the analysis uses no mask, one mask channel, or an AND mask
   generated from two channels.
7. Select the target channel and any additional channels for intensity
   measurements.
8. Configure background correction, thresholding, object-size limits, minimum
   overlap, hole filling, and border clearing.
9. Inspect representative images, masks, thresholds, and overlays in the
   preview.
10. Run the analysis and inspect the exported CSV file.

## ROI selection

MasQ offers the following ROI modes:

- **Whole image**: analyzes the complete image plane.
- **Centered ROI**: places an ROI of the specified dimensions at the image
  center.
- **Highest signal in selected channel**: searches for an ROI with high mean
  intensity in the selected channel.
- **Lowest signal in selected channel**: searches for an ROI with low mean
  intensity in the selected channel.

For signal-based placement, similarly scoring regions are resolved in favor of
the region closer to the image center.

## Background correction and thresholding

Background correction is performed by subtracting a Gaussian-filtered version
of an image from the original channel. Negative corrected values are set to
zero. A Gaussian sigma of zero disables this correction.

Thresholds can be shared by all mask and target channels or configured
separately. Available methods are Otsu, Yen, Triangle, Li, and a manually
specified threshold. Thresholding can also be disabled where the interface
offers that option.

Binary masks can be cleaned by filling holes, removing objects touching the
ROI border, removing objects below a minimum area, and excluding objects above
an optional maximum area. Areas are reported in pixels because the program
does not apply spatial calibration.

## Measurements

Depending on the selected analysis mode, the exported measurements include:

- Mean and median background-corrected intensity within the ROI
- Mean and median background-corrected intensity within the final mask
- Number and combined pixel area of qualifying target objects
- Mean intensity of qualifying target objects
- Area of the final mask as a percentage of the ROI
- Fraction of thresholded target signal located within the final mask
- Target area within the final mask as a percentage of the ROI
- Jaccard index between the final mask and thresholded target signal
- ROI coordinates, channel assignments, threshold settings, image identifiers,
  and folder-derived experimental metadata

A target object is included when at least the configured fraction of its area
overlaps the final analysis mask.

## Output format

MasQ writes one row per input image to a semicolon-delimited CSV file. Numeric
values use decimal commas to facilitate import into German-language versions
of spreadsheet and statistical software. The output also records the source
image path and the settings needed to identify the analysis configuration.

## Important limitations

- MasQ is intended for two-dimensional image planes and is not a general
  multidimensional CZI analysis tool.
- Unsupported CZI dimensions are reduced to their first position.
- Pixel-based areas are not converted into calibrated physical units.
- The validity of segmentation depends on appropriate parameters and visual
  quality control by the user.
- Automatic thresholding methods with the same name may be implemented
  differently in other software and can therefore produce different results.
- Users should validate the workflow for their own images and experimental
  application before drawing biological conclusions.

## Citation

If you use MasQ in academic work, please cite the archived software release.
Citation metadata are provided in `CITATION.cff`. GitHub and Zenodo links and
the DOI will be added when version 1.0.0 is released.

Suggested citation before assignment of the DOI:

> Brandtner, L. (2026). *MasQ: CZI ROI Mask Analyzer* (Version 1.0.0)
> [Computer software]. ORCID: https://orcid.org/0009-0001-8462-8887

## License

MasQ is released under the MIT License. The software may be used, copied,
modified, merged, published, distributed, sublicensed, and sold under the
conditions stated in the `LICENSE` file. Scientific users are additionally
asked to cite the archived software release.

## Author

Leah Brandtner  
ORCID: https://orcid.org/0009-0001-8462-8887


# PixF Channel Analysis

## Description

PixF (Pixel-based Fluorescence Analysis) is a rapid and efficient tool designed for image characterization, enabling users to analyze collections of images with ease. One of its key features, PixF Channel Analysis, performs real-time separation of red, green, and blue (RGB) channels, measuring spatial intensity at each pixel. This capability allows users to accurately assess the distribution of fluorescence markers in images.

A notable application of this tool is the extraction of co-expression data, allowing for the mapping of individual molecular expressions in biological tissue samples. In addition to processing images into primary RGB channels, PixF Channel Analysis can also generate channel images and composite images in colors such as yellow, purple, and teal. It visualizes the data through 3D bar plots, where the x and y axes denote the pixel's spatial position, and the z-axis reflects the absolute fluorescence intensity.

This tool is particularly useful in biological and biomedical research, offering insights into molecular and spatial relationships within image data.

## Getting started / Installation

To install PixF Channal Analysis

1. clone the github repository

```git clone https://github.com/ChowdhuryRatul/pixf-channel-analysis```

2. navigate and install python requirements

```
cd ./pixf-channel-analysis
pip install -r requirements.txt
```

## Usage

To use PixF Channal Analysis

1. follow step in installation

2. run PixF Channel Analysis (below is a demo)

```
python pixf_ca_analysis.py --imagefolder ./demo --channels R,G,Y --output PixF_ca_output --threshold 30 --plot N --image N
```

### Arguments

| argument      | option                    | description                                                                                                  | default        |
| ------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------- |
| --imagefolder | ./path/to/folder          | Input path to the folder containing collection of images for analysis.                                       |                |
| --channels    | R/G/B/Y/T/P               | Input single/multiple 1 letter representation of channel/composite for analysis. Example --channels R,G,Y    | R,G,Y          |
| --output      | Desired output file name. | Output file/images/plot file name.                                                                           | PixF_ca_output |
| --threshold   |                           | The intensity threshold used during analysis, any pixel with intensity below the threshold will be excluded. | 30             |
| --plot        |                           | Y to plot 3d manhattan bar plot, N otherwise                                                                 | N              |
| --image       |                           | Y to generate channel/composite image base on --channels. N otherwise                                        | N              |

## License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

## Acknowledgement

PixF Channel Analysis is developed by
[**Chowdhury Lab**](https://chowdhurylab.github.io/) team.

""" 
usage:
    pixf_ca_analysis.py --imagefolder ./demo --channel R,G,Y
"""

from tqdm import tqdm
from utils import color_analysis_4, get_all_images, getColorKey
import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="pixf_ca_analysis.py analyze all images in a given folder, specify relavent channel for analysis."
    )
    parser.add_argument(
        "--imagefolder",
        type=str,
        required=True,
        help="Input path the folder that contain images for analysis.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="R,G,Y",
        required=False,
        help="Default: R,G,Y. Available options: R/G/B/Y/T/P. Input single/multiple relavant channel for analysis. Example: --channels R,G",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PixF_ca_output",
        required=False,
        help="Default: PixF_ca_output. Desired output file name.",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="30",
        required=False,
        help="Default at: 30. The intensity threshold used during analysis, any pixel with intensity below the threshold will be excluded.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="N",
        required=False,
        help="Default at: N. Available options: Y/N. Y to plot 3d, N otherwise.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="N",
        required=False,
        help="Default at: N. Available options: Y/N. Y to output channel image, N otherwise.",
    )
    args = parser.parse_args()
    return args


def addHeaders(channels):
    header = []
    for c in channels:
        header += generateChannelHeader(c)
    return header


def generateChannelHeader(channel):
    return [
        channel + " - min",
        channel + " - max",
        channel + " - mean",
        channel + " - median",
        channel + " - R total intensity",
        channel + " - G total intensity",
        channel + " - B total intensity",
        channel + " - R (%) intensity",
        channel + " - G (%) intensity",
        channel + " - B (%) intensity",
        channel + " - percentage",
        channel + " - pixels",
    ]


def generateChannelRow(object, channel):
    return [
        str(object[getColorKey(channel)]["min"]),
        str(object[getColorKey(channel)]["max"]),
        str(object[getColorKey(channel)]["mean"]),
        str(object[getColorKey(channel)]["median"]),
        str(object[getColorKey(channel)]["rgbIntensity"][0]),
        str(object[getColorKey(channel)]["rgbIntensity"][1]),
        str(object[getColorKey(channel)]["rgbIntensity"][2]),
        str(object[getColorKey(channel)]["rgbPercentIntensity"][0]),
        str(object[getColorKey(channel)]["rgbPercentIntensity"][1]),
        str(object[getColorKey(channel)]["rgbPercentIntensity"][2]),
        str(object[getColorKey(channel)]["percentage"]),
        str(object[getColorKey(channel)]["colorPixels"]),
    ]


def write_to_result(save_file, str):
    with open(save_file, "a") as f:
        f.write(str)


if __name__ == "__main__":
    args = parser()
    print("Starting PixF Channel Analysis for analytics...")

    images = get_all_images(args.imagefolder)
    output_file = args.output + ".csv"
    channels = [
        c for c in args.channels.split(",") if c in ["R", "G", "B", "Y", "P", "T"]
    ]

    if args.plot not in ["Y", "N"]:
        raise ValueError("valid input for --plot: ['Y', 'N'].")

    if args.image not in ["Y", "N"]:
        raise ValueError("valid input for --channelimage: ['Y', 'N'].")

    plot = True if args.plot == "Y" else False
    channel_image = True if args.image == "Y" else False

    with open(output_file, "w") as f:
        f.write(
            "protein image,total image pixel," + ",".join(addHeaders(channels)) + "\n"
        )

    print("Analyzing all images...")
    for image in tqdm(images):
        analysis_object = color_analysis_4(
            image,
            intensity_threshold=int(args.threshold),
            output_image=channel_image,
            output_plot=plot,
            output_channel=channels,
        )

        write_to_result(
            output_file,
            image.split("/")[-1]
            + ","
            + str(analysis_object[getColorKey("R")]["totalPixels"])
            + ",",
        )
        row = []
        for c in channels:
            row += generateChannelRow(analysis_object, c)
        write_to_result(output_file, ",".join(row))

        write_to_result(output_file, "\n")

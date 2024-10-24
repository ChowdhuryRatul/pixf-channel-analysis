import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# _________________________________________________________________________________________
# main function
def color_analysis_4(
    image_path,
    intensity_threshold=30,
    output_image=False,
    output_plot=False,
    output_channel=["R", "G", "B", "Y", "P", "T"],
):
    main_img = Image.open(image_path)
    main_img = main_img.convert("RGB")

    image_name = image_path.split("/")[-1]
    image_name = ".".join(image_name.split(".")[:-1])

    rgb_color = get_color_information(main_img, intensity_threshold)
    total_pixels = main_img.size[0] * main_img.size[1]

    stats = {}
    for i, c in enumerate(output_channel):
        stats[c] = get_color_statistic(rgb_color[c])
        stats[c]["totalPixels"] = total_pixels
        if output_image:
            color_to_image(rgb_color[c], main_img.size, id=c, save=image_name)

    if output_plot:
        r_gray, g_gray, b_gray = gray_scale(main_img)
        buildPlotData(image_name, r_gray, g_gray)

    analysis_object = {
        "red": stats["R"] if "R" in stats else {},
        "green": stats["G"] if "G" in stats else {},
        "blue": stats["B"] if "B" in stats else {},
        "yellow": stats["Y"] if "Y" in stats else {},
        "purple": stats["P"] if "P" in stats else {},
        "teal": stats["T"] if "T" in stats else {},
    }

    return analysis_object


# _________________________________________________________________________________________
# channel analysis helper functions
def get_color_information(image_obj, intensity_threshold):
    color = {
        "R": [],
        "G": [],
        "B": [],
        "Y": [],
        "P": [],
        "T": [],
    }

    for row in np.array(image_obj):
        for pixel in row:
            pixel = [int(p) for p in pixel]
            if (
                pixel[0] > 0
                and pixel[1] > 0
                and (pixel[0] + pixel[1] // 2) >= intensity_threshold
            ):  # Yellow
                color["Y"].append([pixel[0], pixel[1], 0])
            else:
                color["Y"].append([0, 0, 0])

            if (
                pixel[0] > 0
                and pixel[2] > 0
                and (pixel[0] + pixel[2] // 2) >= intensity_threshold
            ):  # Purple
                color["P"].append([pixel[0], 0, pixel[2]])
            else:
                color["P"].append([0, 0, 0])

            if pixel[0] > 0 and pixel[0] >= intensity_threshold:  # R
                color["R"].append([pixel[0], 0, 0])
            else:
                color["R"].append([0, 0, 0])

            if (
                pixel[1] > 0
                and pixel[2] > 0
                and (pixel[1] + pixel[2] // 2) >= intensity_threshold
            ):  # Teal
                color["T"].append([0, pixel[1], pixel[2]])
            else:
                color["T"].append([0, 0, 0])

            if pixel[1] > 0 and pixel[1] >= intensity_threshold:  # G
                color["G"].append([0, pixel[1], 0])
            else:
                color["G"].append([0, 0, 0])

            if pixel[2] > 0 and pixel[2] >= intensity_threshold:  # B
                color["B"].append([0, 0, pixel[2]])
            else:
                color["B"].append([0, 0, 0])

    return color


def color_to_image(color_arr, size, id=None, save=None):
    image = Image.new("RGB", size)
    flattened_data = [tuple(pixel) for pixel in color_arr]
    image.putdata(flattened_data)

    if save and id:
        image.save(f"{save}_output_image_{id}.png")

    elif save:
        image.save(f"{save}_output_image.png")

    elif id:
        image.save(f"output_image_{id}.png")

    return image


def get_color_statistic(color_arr):
    minn = float("inf")
    maxx = 0
    median = []
    total = 0
    rgb = [0, 0, 0]

    for pixel in color_arr:
        if pixel == [0, 0, 0]:  # skip empty pixel
            continue
        value = sum(pixel) // (len(pixel) - pixel.count(0))

        minn = min(minn, value)
        maxx = max(maxx, value)
        median.append(value)

        rgb = [x + y for x, y in zip(rgb, pixel)]
        total += 1

    minn = 0 if minn == float("inf") else minn

    return {
        "min": minn,
        "max": maxx,
        "mean": (
            cal_percentage(sum(rgb) / (len(rgb) - rgb.count(0)) / 100, total)
            if rgb.count(0) < 3
            else 0
        ),
        "median": statistics.median(median) if median else 0,
        "colorPixels": total,
        "percentage": cal_percentage(total, len(color_arr)),
        "rgbIntensity": rgb,
        "rgbPercentIntensity": [
            cal_percentage(rgb[0], sum(rgb)) if rgb.count(0) < 3 else 0,
            cal_percentage(rgb[1], sum(rgb)) if rgb.count(0) < 3 else 0,
            cal_percentage(rgb[2], sum(rgb)) if rgb.count(0) < 3 else 0,
        ],
    }


def cal_percentage(part, whole):
    return round((float(part) / float(whole)) * 100, 2)


def get_all_images(folder_path):
    done = False
    paths = [folder_path]
    images = []

    while not done:
        path = paths.pop()

        images += [f"{path}/{x}" for x in os.listdir(path)]

        for image in images:
            if (
                image[-5:] != ".tiff"
                and image[-4:] != ".png"
                and image[-4:] != ".jpg"
                and image[-5:] != ".jpeg"
                and not os.path.isfile(image)
            ):
                paths.append(image)

        if not paths:
            done = True

        for path in paths:
            if path in images:
                images.remove(path)

    return images


def getColorKey(letter):
    keys = {
        "R": "red",
        "G": "green",
        "B": "blue",
        "Y": "yellow",
        "P": "purple",
        "T": "teal",
    }
    return keys[letter]


# _________________________________________________________________________________________
# 3d plot helper functions


def buildPlotData(image_name, r_gray, g_gray):
    r_gray_numpy = np.array(r_gray)
    g_gray_numpy = np.array(g_gray)

    matrix = []
    ratio_colors = []  # this will be a already flatten version.
    ratio_max = -1
    for i in range(len(r_gray_numpy)):
        row = []

        color_row = []
        for j in range(len(r_gray_numpy[0])):
            red_intensity = int(r_gray_numpy[i][j][0])
            green_intensity = int(g_gray_numpy[i][j][0])

            # for Ratio calculation here
            ratio = 0  # (this is height)
            ratio_color = [255 / 255, 255 / 255, 255 / 255]

            # We only check ratio when the pixel has both red and green intensity
            # otherwise, not bar for that pixel.
            if red_intensity > 0 and green_intensity > 0:

                # when both equal, we use pure yellow.
                # but this will also have the highest opacity
                # due to scaling based on max ratio we can obtain.
                # tall = more opaque, short = less opaque
                # that way tall bar are not blocking shorter bar behind.
                if red_intensity == green_intensity:
                    ratio = 1
                    ratio_color = [255 / 255, 255 / 255, 0 / 255]

                elif red_intensity > green_intensity:
                    ratio = red_intensity // green_intensity
                    ratio_color = [255 / 255, 0 / 255, 0 / 255]

                elif green_intensity > red_intensity:
                    ratio = green_intensity // red_intensity
                    ratio_color = [0 / 255, 255 / 255, 0 / 255]

            ratio_max = max(ratio_max, ratio)
            row.append(ratio)
            color_row.append(ratio_color)

        ratio_colors.append(color_row)
        matrix.append(row)

    matrix = matrix[::-1]
    ratio_colors = ratio_colors[::-1]

    matrix = pad_matrix(matrix)
    ratio_colors = pad_matrix(ratio_colors)

    # once we got all the height(ratios), we can scale the opacity
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if ratio_colors[x][y] == 0:
                ratio_colors[x][y] = [255 / 255, 255 / 255, 255 / 255]

            height = matrix[x][y]
            r = height / (ratio_max + 25)  # <-- 2 // 280

            # add 25 here because if there is a height of 255
            # and the ratio_max is 255, we will get 1.
            # and ended up with opacity 1 - 1 = 0 (which means we can't see it.)
            if r > 1:
                print(r)
                print(height / (ratio_max + 25))
                print(height)
                print((ratio_max + 25))
                raise ValueError(
                    f"Opacity ratio (to be minused) > 1, value: {r}, ratio: {height}, max ratio: {ratio_max}"
                )

            ratio_colors[x][y].append(min(0.4, 1 - r))

    ratio_colors = flatten_matrix(ratio_colors)

    plot_3d_barchart(matrix=matrix, save=f"{image_name} (3d plot)", colors=ratio_colors)


def plot_3d_barchart(
    matrix=[[1, 2, 3, 4], [4, 5, 6, 7]],
    save="",
    colors=[
        [255 / 255, 0 / 255, 0 / 255, 1],
        [255 / 255, 0 / 255, 0 / 255, 1],
        [255 / 255, 0 / 255, 0 / 255, 0.1],
        [255 / 255, 0 / 255, 255 / 255, 0.5],
        [0 / 255, 255 / 255, 0 / 255, 1],
        [0 / 255, 255 / 255, 0 / 255, 1],
        [0 / 255, 255 / 255, 0 / 255, 1],
        [255 / 255, 255 / 255, 0 / 255, 1],
    ],
):
    # Create a sample 2D array
    data = np.array(matrix)

    # Create a figure and 3D axes
    mul = 20
    fig = plt.figure(figsize=(4 * mul, 3 * mul))
    ax = fig.add_subplot(111, projection="3d")

    # Generate x and y coordinates
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # Flatten the arrays
    x = x.ravel()
    y = y.ravel()
    z = data.ravel()

    #                           width, depth
    ax.bar3d(x, y, np.zeros_like(z), 1, 1, z, color=colors)

    # Set labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Set ticks
    ax.set_zticks(np.arange(0, 256, 50))

    # Custom z ticks
    # Customize Z ticks with increased font size and Arial font
    ax.tick_params(axis="x", labelsize=10 * mul, pad=2 * mul)
    ax.tick_params(axis="y", labelsize=10 * mul, pad=4 * mul)

    ax.tick_params(axis="z", labelsize=10 * mul, pad=8 * mul)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontname("Arial")

    # Show the plot
    if save:
        fig.savefig(f"{save}.png")

    plt.close()


def gray_scale(image):
    r, g, b = image.split()

    r_gray = r.convert("RGB")
    g_gray = g.convert("RGB")
    b_gray = b.convert("RGB")

    return r_gray, g_gray, b_gray


def flatten_matrix(matrix):
    return [element for row in matrix for element in row]


def pad_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Calculate new size for square matrix
    max_dim = max(rows, cols)

    # Initialize new square matrix with zeros
    square_matrix = [[0 for _ in range(max_dim)] for _ in range(max_dim)]

    # Copy original matrix into the center of the square matrix
    for i in range(rows):
        for j in range(cols):
            square_matrix[i][j] = matrix[i][j]

    return square_matrix

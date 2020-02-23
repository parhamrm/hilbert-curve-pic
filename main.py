import numpy as np
import cv2
import sys
import argparse
from progress.bar import IncrementalBar

LINE_COLOR = (100, 100, 100)
WINDOW_NAME = "Hilbert Image"

parser = argparse.ArgumentParser(description='Gets an Image as input and saves the hilbert curved image. Works best on squared images. use -p for preview.')
parser.add_argument('image', help="Path to image.", type=str)
parser.add_argument('--save-to', help="Path to save location", type=str, default="./res.jpg")
parser.add_argument('--order', help="Order of dense hilbert curve.", type=int, default=7)
parser.add_argument('--size', help="Final image size.", type=int, default=1024)
parser.add_argument('--thickness', help="Line thickness value.", type=int, default=4)
parser.add_argument('-p', help="Only see preview.", action='store_true')

def hilbert(order: int, length: int, total: int) -> tuple:
    bar = IncrementalBar('Making Points', max = total)
    options = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=np.uint8)
    for i in range(total):
        bar.next()
        index = i & 3
        point = options[index].copy()

        for j in range(1, order):
            i = i >> 2
            index = i & 3
            l = 2 ** j
            if index == 0:
                point[0], point[1] = point[1], point[0]
            elif index == 1:
                point[1] += l
            elif index == 2:
                point += l
            elif index == 3:
                point[0], point[1] = 2 * l - 1 - point[1], l - 1 - point[0]
        
        yield tuple(map(int, point * length + length / 2))
    bar.finish()

def draw(img: np.array, points: tuple, background_size: tuple, thickness: int) -> np.array:
    hilbert_curve = np.zeros(background_size, np.uint8)
    for i in range(1, len(points)):
        cv2.line(hilbert_curve, points[i-1], points[i], LINE_COLOR, thickness)
    img = cv2.resize(img, background_size[:2]) 
    img[hilbert_curve!=LINE_COLOR] *= 0

    return img

if __name__ == "__main__":
    args = parser.parse_args()

    image_path = args.image
    save_path = args.save_to
    order = args.order
    size = args.size
    thickness = args.thickness
    p = args.p

    n = 2 ** order
    total = n * n
    length = size / n

    background_size = (size, size, 3)

    img = cv2.imread(image_path)

    points = tuple(point for point in hilbert(order, length, total))

    hilbert_image = draw(img, points, background_size, thickness)

    if p:
        cv2.imshow(WINDOW_NAME, hilbert_image)
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, hilbert_image)

    
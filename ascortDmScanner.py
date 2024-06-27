from pylibdmtx.pylibdmtx import decode
import argparse
import cv2
import pyperclip
import sys

# Constants
windowName = "Ascort:DM scanner"
version = 2


def renderPlain(image):
    image = cv2.putText(
        image,
        'Поиск DM',
        (30, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )
    image = cv2.putText(
        image,
        'для завершения нажмите любую кнопку',
        (30, 450),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )
    image = cv2.putText(
        image,
        'Версия ' + str(version),
        (520, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )


def renderWithUIN(image, UIN):
    overlay = image.copy()
    if rectWork:
        cv2.rectangle(
            overlay,
            (int(image.shape[1] / 2 - aimSize), int(image.shape[0] / 2 - aimSize / 4)),
            (int(image.shape[1] / 2 + aimSize), int(image.shape[0] / 2 + aimSize / 4)),
            (135, 255, 169),
            -1
        )
    else:
        cv2.rectangle(
            overlay,
            (int(image.shape[1] / 2 - aimSize / 2), int(image.shape[0] / 2 - aimSize / 2)),
            (int(image.shape[1] / 2 + aimSize / 2), int(image.shape[0] / 2 + aimSize / 2)),
            (135, 255, 169),
            -1
        )
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    image = cv2.putText(
        image,
        UIN,
        (30, 30),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1,
        (132, 255, 56),
        2
    )
    return image


def render(plain, image, points, UIN):
    if plain:
        renderPlain(image)
    else:
        image = renderWithUIN(image, UIN)
    if rectWork:
        image = cv2.rectangle(
            image,
            (int(image.shape[1] / 2 - aimSize), int(image.shape[0] / 2 - aimSize / 4)),
            (int(image.shape[1] / 2 + aimSize), int(image.shape[0] / 2 + aimSize / 4)),
            (135, 255, 169),
            3
        )
    else:
        image = cv2.rectangle(
            image,
            (int(image.shape[1] / 2 - aimSize / 2), int(image.shape[0] / 2 - aimSize / 2)),
            (int(image.shape[1] / 2 + aimSize / 2), int(image.shape[0] / 2 + aimSize / 2)),
            (135, 255, 169),
            3
        )
    cv2.imshow(windowName, image)


def morphology(binary):
    rez = []
    structuring_elements = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ]

    for element in structuring_elements:
        open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element, iterations=1)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, element, iterations=1)
        dilate = cv2.dilate(close, element, iterations=1)
        erode = cv2.erode(dilate, element, iterations=1)
        rez.append(erode)

    return rez


def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img


def binarize_image(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def prepareImage(image):
    imagesForDebug = []
    if rectWork:
        cropAim = image[int(image.shape[0] / 2 - aimSize / 3.5): int(image.shape[0] / 2 + aimSize / 3.5),
                  int(image.shape[1] / 2 - aimSize): int(image.shape[1] / 2 + aimSize)]
    else:
        cropAim = image[int(image.shape[0] / 2 - aimSize / 2): int(image.shape[0] / 2 + aimSize / 2),
                  int(image.shape[1] / 2 - aimSize / 2): int(image.shape[1] / 2 + aimSize / 2)]

    imagesForDebug.append(cropAim)
    gray = cv2.cvtColor(cropAim, cv2.COLOR_BGR2GRAY)
    imagesForDebug.append(gray)

    contrast_image = increase_contrast(cropAim)
    imagesForDebug.append(contrast_image)

    gray = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2GRAY)
    imagesForDebug.append(gray)

    binary = binarize_image(gray)
    imagesForDebug.append(binary)

    morph = morphology(binary)

    for t in morph:
        imagesForDebug.append(t)

    if debug:
        cnt = 0
        for i in imagesForDebug:
            cv2.imshow('debug ' + str(cnt), i)
            cnt += 1

    return morph


def proccessDMCode(image):
    imagesForScan = prepareImage(image)

    for img in imagesForScan:
        # Пробуем декодировать с таймаутом
        data = decode(img, timeout=1000)

        # Проверка на успешность декодирования
        if not data:
            continue

        for decodedObject in data:
            points = [
                (decodedObject.rect.left, img.shape[0] - decodedObject.rect.top),
                (decodedObject.rect.left + decodedObject.rect.width,
                 img.shape[0] - decodedObject.rect.top - decodedObject.rect.height)
            ]
            print(decodedObject.data.decode("utf-8"))
            return (True, points, decodedObject.data.decode("utf-8"))

    return (False, None, None)


# Main
parser = argparse.ArgumentParser(
    prog='Ascort:DM scanner ' + str(version),
    description='datamatrix codes scanners with USB-cameras',
    epilog='''Usages examples: \n
    ascortDmScanner // open the program \n
    ascortDmScanner --camID=1 --clipboard=True // open and scan with camerID = 1 \n
    ascortDmScanner --readFromFile='qr.jpg' --resultFile='result.txt'  // open file with image and save result to  result.txt'''
)

parser.add_argument("--resultFile", "-r", help="file with result", type=str, default='')
parser.add_argument("--camID", "-c", help="USB camera ID", dest="camID", type=int, default=0)
parser.add_argument("--accuracy", "-a", help="accuracy in milliseconds", type=int, default=70)
parser.add_argument("--clipboard", "-C", help="copy resul to clipboard", type=bool, default=False)
parser.add_argument("--readFromFile", "-f", help="the image file to read", type=str, default='')
parser.add_argument("--aimSize", "-as", help="the aim box size in pixelx", type=int, default=140)
parser.add_argument("--debug", "-d", help="show debugging view", type=bool, default=False)
parser.add_argument("--rect", "-rt", help="scan for rects", type=bool, default=False)

args = parser.parse_args()

resultFile = args.resultFile
camID = args.camID
accuracy = args.accuracy
copyToClipboard = args.clipboard
readFromFile = args.readFromFile
debug = args.debug
aimSize = args.aimSize
rectWork = args.rect

cap = cv2.VideoCapture(camID)

# Настройка экспозиции
cap.set(cv2.CAP_PROP_EXPOSURE, -60)

# Event loop
while True:
    if readFromFile != '':
        image = cv2.imread(readFromFile)
    else:
        rez, image = cap.read()
        if not rez:
            print('can\'t attach to camera')
            sys.exit(-1)

    (found, points, UIN) = proccessDMCode(image)
    render(not found, image, points, UIN)

    if found:
        if resultFile != '':
            with open(resultFile, 'w') as f:
                f.write(UIN)
                f.close()
                break
        elif copyToClipboard:
            pyperclip.copy(UIN)

    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break

cv2.destroyWindow(windowName)
cap.release()
sys.exit(0)

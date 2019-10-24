import configparser
import numpy as np
import pydicom
import datetime
import sys
import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.filters import threshold_multiotsu

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']

# CONSTANTES

SIZE = 512

HEMORRAGE_MIN=65
HEMORRAGE_MAX=95

VENTRICULO_MIN=0
VENTRICULO_MAX=15

PARENCHYMA_MIN=20
PARENCHYMA_MAX=45

SKULL_MIN=1000
SKULL_MAX=2048

DICOM_MIN=-1024
DICOM_MAX=2048


# FUNCOES PARA SEGMENTACAO

def hemorrage_threshold(image):
    img = image.copy()
    img[img < HEMORRAGE_MIN] = DICOM_MIN
    img[img > HEMORRAGE_MAX] = DICOM_MIN
    return img

def ventriculo_threshold(image):
    img = image.copy()
    img[img < VENTRICULO_MIN] = DICOM_MIN
    img[img > VENTRICULO_MAX] = DICOM_MIN
    return img

def parenchyma_threshold(image):
    img = image.copy()
    img[img < PARENCHYMA_MIN] = DICOM_MIN
    img[img > PARENCHYMA_MAX] = DICOM_MIN
    return img

def skull_threshold(image):
    img = image.copy()
    img[img < SKULL_MIN] = DICOM_MIN
    img[img > SKULL_MAX] = DICOM_MIN
    return img

def normalize(image, min, max):
    image = abs(image - min) / abs(max - min)
    return image

def multiotsu(image, regions):
    thresholds = threshold_multiotsu(image, classes=regions)
    regions = np.digitize(image, bins=thresholds)
    regions_colorized = label2rgb(regions)
    return (regions_colorized, regions, thresholds)

# plota um histograma da imagem
def histogram(image, remove_min=False):
    max = np.max(image)
    min = np.min(image)
    if remove_min:
        min = np.min(image[image > min])
    plt.hist(image.ravel(), 256, [min, max])
    plt.title("histogram")
    plt.show()

# obtem classificacao de um elemento dos dados de treinamento
def get_classification(filename):
    train_file = "{}/stage_1_train.csv".format(config['TrainPath'])
    classes = np.zeros(6)
    id = filename[:12]
    try:
        idx = 0
        with open(train_file) as myfile:
            for line in myfile:
                pos = line.find(id)
                if (pos == 0):
                    pos = line.find(",")
                    value = int(line[pos + 1:pos + 2])
                    classes[idx] = value
                    idx += 1
                    if idx == 6:
                        break
    except:
        e = sys.exc_info()[0]
        log("erro ao abrir arquivo de treinamento: {}".format(e))
    return classes

# FUNCOES PARA MANIPULAR DICOM

def read_image(filename):
    ds = pydicom.dcmread(filename)
    b = ds.RescaleIntercept
    m = ds.RescaleSlope
    image = m * ds.pixel_array + b
    return image

def update_dicom(path_original, path_alterado, data):
    ds = pydicom.dcmread(path_original)
    ds.PixelData = data.tostring()
    ds.Rows, ds.Columns = data.shape
    ds.save_as(path_alterado)

# FUNCOES PARA DEFINIR REGIAO DE INTERESSE

def get_roi_cabeca(image):

    # segmenta os ossos e outros objetos com maior densidade da imagem
    colorized, otsu, thresholds = multiotsu(image, 3)
    mask_ossos = np.zeros((SIZE, SIZE)).astype('uint8')
    mask_ossos[otsu == 2] = 1

    #obtem contornos para excluir estruturas densas fora do cranio
    cnts = cv2.findContours(mask_ossos, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    contornos = []
    for contorno in cnts:
        area = cv2.contourArea(contorno)
        if area > 100: # estruturas com area menor que este valor nao sao consideradas
            contornos.append(contorno)
    mask  = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask , contornos, -1, 255, -1)
    # new_img = cv2.bitwise_and(image, image, mask=mask)

    # obtem as fronteiras da região de interesse
    # TODO: definir fronteira mais anatômica para o ROI ao inves de bound box (testar método snake).
    top = getTop(mask)
    bottom = getBottom(mask)
    left = getLeft(mask)
    right = getRight(mask)
    # cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    # faz o recorte do retangulo englobando a cabeça
    roi = image
    roi[:top,:] = DICOM_MIN
    roi[bottom:SIZE, :] = DICOM_MIN
    roi[:, :left] = DICOM_MIN
    roi[:, right:SIZE] = DICOM_MIN

    return roi

def define_roi(imagem, mascara):
    left   = getLeft(mascara)
    right  = getRight(mascara)
    top    = getTop(mascara)
    bottom = getBottom(mascara)
    imagem[:left,:] = 0
    imagem[right:,:] = 0
    imagem[:,:top] = 0
    imagem[:,bottom:] = 0
    return imagem

def getTop(img):
    h, w = img.shape
    for i in range(h-1):
        for j in range(w-1):
            if img[i,j] > 0:
                return i

def getBottom(img):
    h, w = img.shape
    for i in range(h-1, 0, -1):
        for j in range(w):
            if img[i,j] > 0:
                return i

def getLeft(imgx):
    h, w = imgx.shape
    for j in range(w-1):
        for i in range(h-1):
            if imgx[i,j] > 0:
                return j

def getRight(imgx):
    h, w = imgx.shape
    for j in range(w-1, 0, -1):
        for i in range(h-1, 0, -1):
            if imgx[i,j] > 0:
                return j

# FUNCOES UTILITARIAS

def log(mensagem):
    now = datetime.datetime.now()
    print("[INFO] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def error(mensagem):
    now = datetime.datetime.now()
    print("[ERROR] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def plot(title, image, color_map=plt.cm.bone):
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.show()

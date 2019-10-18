import configparser
import numpy as np
import pydicom
import datetime
import sys
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.filters import threshold_multiotsu

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']

# CONSTANTES

HEMORRAGE_MIN=65
HEMORRAGE_MAX=95

VENTRICULO_MIN=0
VENTRICULO_MAX=15

PARENCHYMA_MIN=20
PARENCHYMA_MAX=45

SKULL_MIN=1000
SKULL_MAX=2000

# FUNCOES PARA SEGMENTACAO

def hemorrage_threshold(image):
    img = image.copy()
    img[img < HEMORRAGE_MIN] = 0
    img[img > HEMORRAGE_MAX] = 0
    return img

def ventriculo_threshold(image):
    img = image.copy()
    img[img < VENTRICULO_MIN] = 0
    img[img > VENTRICULO_MAX] = 0
    return img

def parenchyma_threshold(image):
    img = image.copy()
    img[img < PARENCHYMA_MIN] = 0
    img[img > PARENCHYMA_MAX] = 0
    return img

def skull_threshold(image):
    img = image.copy()
    img[img < SKULL_MIN] = 0
    img[img > SKULL_MAX] = 0
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
            if img[i,j] == 1:
                return i

def getBottom(img):
    h, w = img.shape
    for i in range(h-1, 0, -1):
        for j in range(w):
            if img[i,j] == 1:
                return i

def getLeft(imgx):
    h, w = imgx.shape
    for j in range(w-1):
        for i in range(h-1):
            if imgx[i,j] == 1:
                return j

def getRight(imgx):
    h, w = imgx.shape
    for j in range(w-1, 0, -1):
        for i in range(h-1, 0, -1):
            if imgx[i,j] == 1:
                return j

# FUNCOES UTILITARIAS

def log(mensagem):
    now = datetime.datetime.now()
    print("{} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def plot(title, image, color_map=plt.cm.bone):
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.show()

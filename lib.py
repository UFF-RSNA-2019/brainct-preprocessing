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
try:
    config = configparser['Geral']
except KeyError:
    now = datetime.datetime.now()
    msg = "Problema ao obter configuração, verifique se você está rodando o programa no diretório principal do projeto."
    print("[ERROR] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), msg), flush=True)
    sys.exit()

# CONSTANTES

DEFAULT_SIZE = 512

HEMORRAGE_MIN=48 #65
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
    train_file = config['GroundTruth']
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

def obtem_imagem(path, id):
    input_filepath = "{}/{}.dcm".format(path, id)
    try:
        # carrega a imagem a partir do filesystem
        image = read_image(input_filepath)
        return image
    except ValueError:
        error("arquivo dicom corrompido: {}".format(id))
        return np.zeros((DEFAULT_SIZE, DEFAULT_SIZE))

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

# FUNCOES UTILITARIAS

def log(mensagem):
    now = datetime.datetime.now()
    print("[INFO] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def error(mensagem):
    now = datetime.datetime.now()
    print("[ERROR] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)

def debug(mensagem):
    now = datetime.datetime.now()
    print("[DEBUG] {} : {}".format(now.strftime("%Y-%m-%d %H:%M:%S.%f"), mensagem), flush=True)


def plot(title, image, color_map=plt.cm.bone):
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.show()


def show(imagem):
    cv2.imshow("uff", imagem)
    cv2.waitKey(0)


# FUNCOES PARA TESTE DOS EXTRATORES

def get_train_images():
    import configparser
    import os
    configparser = configparser.ConfigParser()
    configparser.read('config.ini')
    config = configparser['Geral']
    train_path = config['TrainPath']
    train_folder = os.fsencode(train_path)
    files = os.listdir(train_folder)
    imagens = []
    for file in files:
        filename = os.fsdecode(file)
        id = filename[:12]
        image = obtem_imagem(train_path, id)
        if (image.any()):
            imagens.append((id, image))
    return imagens

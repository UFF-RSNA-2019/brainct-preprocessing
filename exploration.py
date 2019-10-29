from matplotlib import pyplot as plt
import lib
import os
import numpy as np
import configparser

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']

input_path  = config['TrainPath']
input_folder = os.fsencode(input_path)
files = os.listdir(input_folder)
files.sort()

# classes e seus índices
EPIDURAL=0
INTRAPARENCHYMAL=1
INTRAVENTRICULAR=2
SUBARACHNOID=3
SUBDURAL=4
ANY=5

# indica quais classes quer ver
match = np.zeros(6)
match[EPIDURAL] = 0
match[INTRAPARENCHYMAL] = 0
match[INTRAVENTRICULAR] = 0
match[SUBARACHNOID] = 0
match[SUBDURAL] = 0
match[ANY] = 0

for file in files:
    filename = os.fsdecode(file)

    # verifica se esse está selecionado para processamento
    classification = lib.get_classification(filename)
    processa = np.sum(classification * match) > 0
    if not processa and (np.sum(match) == 0) and (np.sum(classification) == 0): processa = True

    processa = (filename == "ID_559b1d8f7.dcm")

    if (processa):

        # le o arquivo
        filepath = "{}/{}".format(input_path, filename)
        image = lib.read_image(filepath)

        roi = image
        lib.plot("original: {} \n {}".format(filename, classification), image)

        # segmentacao via limiarizacao

        # ossos com otsu multinivel
        colorized, otsu, thresholds = lib.multiotsu(image, 3)
        lib.plot("otsu: {} \n {} \n {}".format(filename, classification, thresholds), colorized)
        mask_ossos = np.zeros((512,512))
        mask_ossos[otsu==2] = 1
        ossos = image * mask_ossos
        lib.plot("ossos: {} \n classes: {} \n thresholds {}".format(filename, classification, thresholds), ossos, color_map=plt.cm.bone)

        # limiarizacao parenchyma
        parenchyma = lib.parenchyma_threshold(image)
        lib.plot("parenchyma: {} \n {}".format(filename, classification), parenchyma, color_map=plt.cm.pink)


        # hemorragia
        hemorrage = lib.hemorrage_threshold(roi)
        lib.plot("hemorrage: {} \n {}".format(filename, classification), hemorrage, color_map=plt.cm.tab20)

        # ventriculo
        ventriculo = lib.ventriculo_threshold(roi)
        lib.plot("ventriculo: {} \n {}".format(filename, classification), ventriculo, color_map=plt.cm.tab20)

        # append = np.append(hemorrage, ventriculo)
        # lib.plot("append: {} \n {}".format(filename, classification), append, color_map=plt.cm.bone)

        # intersecao
        threshold = hemorrage + ventriculo

        # normaliza
        normalized = lib.normalize(threshold, lib.VENTRICULO_MIN, lib.HEMORRAGE_MAX)
        lib.plot("hemorragia + venticulo: {} \n {}".format(filename, classification), normalized, color_map=plt.cm.tab20)

        print("{}".format(filename))


print("Done")
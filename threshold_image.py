import lib
import os
import numpy as np
import configparser

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']

files = os.listdir(config['TrainPath'])

contador = 0

# processa todos os arquivos
for file in files:
    filename = os.fsdecode(file)

    # le arquivo original
    input_filepath = "{}/{}".format(config['TrainPath'], filename)
    image = lib.read_image(input_filepath)

    # todo: definir area de interesse com snake
    roi = image

    # limiarizacao
    hemorrage = lib.hemorrage_threshold(roi)
    ventriculo = lib.ventriculo_threshold(roi)

    # intersecao
    threshold = hemorrage + ventriculo

    # normaliza
    normalized = lib.normalize(threshold, lib.VENTRICULO_MIN, lib.HEMORRAGE_MAX)

    # salva arquivo
    output_filepath = "{}/{}.npz".format(config['SegmentationPath'], filename[:12])
    np.savez_compressed(output_filepath, data=normalized)
    lib.log("{} arquivo: {}".format(contador, output_filepath))
    contador += 1

# faz o flip do epidural
with open(config['EpiduralSetFile']) as f:
    for line in f:
        id = line[:12]
        input_epidural_filepath = "{}/{}.npz".format(config['SegmentationPath'], id)
        image = np.load(input_epidural_filepath)["data"]
        flipped = np.fliplr(image)
        output_epidural_filepath = "{}/{}_flip.npz".format(config['SegmentationPath'], id)
        np.savez_compressed(output_epidural_filepath, data=flipped)
        lib.log("{} arquivo epidural: {}".format(contador, output_epidural_filepath))

print("Done")
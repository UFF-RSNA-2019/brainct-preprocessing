import lib
import os
import configparser
import csv
import numpy as np

from features import tem_ventriculo
from features import qtd_hemorragia
from features import area_hemorragia

# Programa que extrai caracteristicas de imagens DICOM com tomografias de cranio e coloca em arquivo texto
# estes arquivos tem por finalidade serem processados por métodos como: KNN, XGBoost, MLP e RandomForest.
# Será utilizada saída multiclass (SKLearn contempla saída multilabel para os classificadores acima).
# referencia:http://scikit.ml/index.html
#            https://xang1234.github.io/multi-label/
#            https://github.com/scikit-multilearn/scikit-multilearn

# inicializacao
configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']
gt_file_path = config['GroundTruth']
input_path = config['TrainPath']
output_path = config['FeaturesPath']
input_folder = os.fsencode(input_path)
image_files = os.listdir(input_folder)

def calcula_features(id):
    feature_vector = []
    input_filepath = "{}/{}.dcm".format(config['TrainPath'], id)
    try:
        # carrega a imagem a partir do filesystem
        image = lib.read_image(input_filepath)
    except ValueError:
        lib.error("arquivo dicom corrompido: {}".format(id))

    # obtem as features

    # debug
    # print("******************")
    # lib.plot("imagem {} - {}".format(contador, filename), image)
    # debug

    # 1. verifica se tem ventriculo
    feature_vector.append(tem_ventriculo.extract(image))

    # 2. quantidade de hemorragias
    feature_vector.append(qtd_hemorragia.extract(image))

    # 3. área média da hemorragia
    feature_vector.append(area_hemorragia.extract(image))

    # 4. distancia média da hemorragia para o osso (osso mais perto)

    # debug
    print ("imagem {}, feature:{}".format(id, feature_vector))
    # debug

    return feature_vector


# declara as matrizes
stage1_x = []
stage1_y = []

# preenche as matrizes
with open(gt_file_path) as f:
    reader = csv.DictReader(f, delimiter=',')
    n_row=1
    labels = []
    for row in reader:
        id = row['ID'][:12]
        sample = n_row // 6
        print("n_row {} sample {}".format(n_row, sample))
        labels.append(row['Label'])
        if (n_row % 6) == 0:
            # cacula feature
            features = calcula_features(id)
            if (len(features) > 0):
                stage1_x.append(features)
                stage1_y.append(labels)
                # TODO: Falta gravar os vetores
            labels = []
        n_row += 1

print("Done")
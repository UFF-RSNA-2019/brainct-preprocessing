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

# configuracao
gt_file_path = lib.config['GroundTruth']
train_path = lib.config['TrainPath']
test_path = lib.config['TestPath']
output_path = lib.config['FeaturesPath']


# gera vetor de features a partir de todas as features criadas
def extrai_features(image):
    feature_vector = []

    # 1. verifica se tem ventriculo
    feature_vector.append(tem_ventriculo.extract(image))

    # 2. quantidade de hemorragias
    feature_vector.append(qtd_hemorragia.extract(image))

    # 3. área média da hemorragia
    feature_vector.append(area_hemorragia.extract(image))

    # 4. distancia média da hemorragia para o osso (osso mais perto)

    return feature_vector

# PROCESSAMENTO PRINCIPAL

# 1. define os vetores de features e labels que serão gerados
# vetores de entrada para o treinamento do SKlearn
stage1_x = []  # features
stage1_y = []  # labels
# vetor de teste e vetor com respectivos ids para serem usados posteriormente na predição/submissão
stage1_test_id = []
stage1_test_x = []

# 2. processa os dados de treinamento
# para cada ID do dataset de treinamento gera um registro nos vetores stage1_x e stage1_y
contador = 0
with open(gt_file_path) as f:
    reader = csv.DictReader(f, delimiter=',')
    n_row=1
    labels = []
    tem_epidural = False
    for row in reader:
        # monta vetor de labels para cada ID
        id = row['ID'][:12]
        sample = n_row // 6
        labels.append(int(row['Label']))

        # verifica este ID tem epidural
        if  (row['ID'].find('epidural') != -1) and (row['Label'] == '1'): tem_epidural = True

        if (n_row % 6) == 0:
        # já concluiu vetor de labels do ID agora pode extrair as features
            image = lib.obtem_imagem(train_path, id)
            if (image.any()):
                features = extrai_features(image)
                if (len(features) > 0):
                    stage1_x.append(features)
                    stage1_y.append(labels)
                    contador += 1
                    print("train image {}, {}, feature:{}".format(id, contador, features))
                # se for epidural aumenta os dados de treinamento com imagem espelhada
                if (tem_epidural):
                    image = np.fliplr(image)
                    features = extrai_features('flipped_epid', image)
                    if (len(features) > 0):
                        stage1_x.append(features)
                        stage1_y.append(labels)
                        contador += 1
                        print("train image flipped_epid, {}, feature:{}".format(contador, features))

            labels = []
            tem_epidural = False
        n_row += 1
# salva arquivos csv
np.savetxt('{}/stage1_x.csv'.format(output_path), stage1_x, fmt='%1.10f', delimiter=',')
np.savetxt('{}/stage1_y.csv'.format(output_path), stage1_y, fmt='%1.0f', delimiter=',')

# 2. processa os dados de teste
# para cada ID do arquivo de teste gera um registro nos vetores stage1_test_x e no stage1_test_id
contador = 0
test_folder = os.fsencode(test_path)
files = os.listdir(test_folder)
for file in files:
    filename = os.fsdecode(file)
    id = filename[:12]
    image = lib.obtem_imagem(test_path, id)
    if (image.any()):
        features = extrai_features(image)
        if (len(features) > 0):
            stage1_test_x.append(features)
            stage1_test_id.append(id)
            contador += 1
            print("test image {}, {}, feature:{}".format(id, contador, features))
# salva arquivos csv
np.savetxt('{}/stage1_test_x.csv'.format(output_path), stage1_test_x, fmt='%1.10f', delimiter=',')
np.savetxt('{}/stage1_test_id.csv'.format(output_path), stage1_test_id, fmt='%s', delimiter=',')

print("Done")
from keras.applications import VGG16
import lib
import csv
import os
import numpy as np

import feature_tem_ventriculo
import feature_qtd_hemorragia
import feature_area_hemorragia

test_path = lib.config["TestPath"]
output_path = lib.config['FeaturesPath']

# gera vetor de features a partir de todas as features criadas
def extrai_features(image):
    feature_vector = []

    # 1. verifica se tem ventriculo
    feature_vector.append(feature_tem_ventriculo.extract(image))

    # 2. quantidade de hemorragias
    feature_vector.append(feature_qtd_hemorragia.extract(image))

    # 3. área média da hemorragia
    feature_vector.append(feature_area_hemorragia.extract(image))

    # 4. distancia média da hemorragia para o osso (osso mais perto)

# inclui o resultado da extracao de features nos arquivos
def append_train(features, labels):
    # salva arquivos csv
    # features
    features_str = ",".join(['{:1.10f}'.format(x) for x in features])
    with open("{}/{}".format(output_path, "stage1_x.csv"), "a") as resultado_file:
        resultado_file.write("{}\n".format(features_str))

    # labels
    labels_str = ",".join([str(x) for x in labels])
    with open("{}/{}".format(output_path, "stage1_y.csv"), "a") as resultado_file:
        resultado_file.write("{}\n".format(labels_str))

def append_test(features, id):
    # salva arquivos csv
    # features
    features_str = ",".join(['{:1.10f}'.format(x) for x in features])
    with open("{}/{}".format(output_path, "stage1_test_x.csv"), "a") as resultado_file:
        resultado_file.write("{}\n".format(features_str))

    # ID
    with open("{}/{}".format(output_path, "stage1_test_id.csv"), "a") as resultado_file:
        resultado_file.write("{}\n".format(id))

train_path = lib.config["TrainPath"]

conv_base = VGG16(weights='imagenet',
                  include_top=False)
features = np.zeros(shape=(12, 512, 512, 1))
labels = np.zeros(shape=(12))

gt_file_path = lib.config['GroundTruth']

# 1. processa os dados de treinamento
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
            lib.log("train image {}, {}".format(id, contador))
            image = lib.obtem_imagem(train_path, id)
            if (image.any() and image.shape == (512, 512)):
                try:
                    features_batch = conv_base.predict(image)
                    features_manual = extrai_features(image)
                    if (len(features_manual) > 0):
                        append_train(features_manual, labels)
                        contador += 1
                    # se for epidural aumenta os dados de treinamento com imagem espelhada
                    if (tem_epidural):
                        image = np.fliplr(image)
                        features_manual = extrai_features(image)
                        if (len(features_manual) > 0):
                            append_train(features_manual, labels)
                            contador += 1
                except Exception as e:
                    print(str(e))


            labels = []
            tem_epidural = False
        n_row += 1

# # 2. processa os dados de teste
# # para cada ID do arquivo de teste gera um registro nos vetores stage1_test_x e no stage1_test_id
# contador = 0
# test_folder = os.fsencode(test_path)
# files = os.listdir(test_folder)
# for file in files:
#     lib.log("test image {}, {}".format(id, contador))
#     filename = os.fsdecode(file)
#     id = filename[:12]
#     image = lib.obtem_imagem(test_path, id)
#     if (image.any()):
#         features_manual = extrai_features(image)
#         if (len(features_manual) > 0):
#             append_test(features_manual, id)
#             contador += 1

print("Done")
import lib
import os
import configparser
import csv
import numpy as np
import segmentation
import extractor

# Programa que extrai caracteristicas de imagens DICOM com tomografias de cranio e coloca em arquivo texto
# estes arquivos tem por finalidade serem processados por métodos como: KNN, XGBoost, MLP e RandomForest.
# Será utilizada saída multiclass (SKLearn contempla saída multilabel para os classificadores acima).
# referencia:http://scikit.ml/index.html
#            https://xang1234.github.io/multi-label/
#            https://github.com/scikit-multilearn/scikit-multilearn

# configuracao
MAX_EXAMES_NORMAL=288000
gt_file_path = lib.config['GroundTruth']
train_path = lib.config['TrainPath']
test_path = lib.config['TestPath']
output_path = lib.config['FeaturesPath']

# gera vetor de features a partir de todas as features criadas
def extrai_features(image):
    ossos, hemorragias, contornos = segmentation.execute(image)
    return extractor.extract(ossos, hemorragias, contornos)

number_append_train = 0
features_buffer = ""
labels_buffer = ""

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

# PROCESSAMENTO PRINCIPAL

if __name__ == "__main__":

    # 1. processa os dados de treinamento
    # para cada ID do dataset de treinamento gera um registro nos vetores stage1_x e stage1_y
    with open(gt_file_path) as f:
        reader = csv.DictReader(f, delimiter=',')
        labels = []
        contador = 0
        contador_normal = 0
        e_epidural = False
        e_normal = False
        n_row=1
        for row in reader:
            # monta vetor de labels para cada ID
            id = row['ID'][:12]
            sample = n_row // 6
            labels.append(int(row['Label']))

            # verifica este ID tem epidural
            if  (row['ID'].find('epidural') != -1) and (row['Label'] == '1'): e_epidural = True

            # verifica este ID é normal
            if  (row['ID'].find('any') != -1) and (row['Label'] == '0'):
                e_normal = True
                contador_normal += 1

            if (n_row % 6) == 0:
            # já concluiu vetor de labels do ID agora pode extrair as features
                # se for normal só processa até uma quantidade predefinida (undersampling)
                if (e_normal) and (contador_normal < MAX_EXAMES_NORMAL):
                    image = lib.obtem_imagem(train_path, id)
                    if (image.any() and image.shape == (512,512)):
                        lib.log("train image {}, {}".format(id, contador))
                        feat = np.ravel(extrai_features(image))
                        if (len(feat) > 0):
                            append_train(feat, labels)
                            contador += 1
                        # se for epidural aumenta os dados de treinamento com imagem espelhada
                        if (e_epidural):
                            image = np.fliplr(image)
                            feat = extrai_features(image)
                            if (len(feat) > 0):
                                append_train(feat, labels)
                        # TODO: salva imagem segmentada
                    labels = []
                    e_epidural = False
            n_row += 1

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
            lib.log("test image {}, {}".format(id, contador))
            feat = extrai_features(image)
            if (len(feat) > 0):
                append_test(feat, id)
                contador += 1
    print("Done")
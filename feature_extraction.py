import lib
import os
import configparser
from features import tem_ventriculo
from features import qtd_hemorragia

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
input_path = config['TrainPath']
folder = os.fsencode(input_path)
files = os.listdir(folder)

# Feature Vector
# 0: tem_ventriculo
# 1: qtd_hemorragia
# 2:
# 3:

contador = 0
max_records = 100 # variavel para limitar qtd registros processados

contador = 0

for file in files:
    feature_vector = []
    filename = os.fsdecode(file)
    input_filepath = "{}/{}".format(config['TrainPath'], filename)
    try:
        # carrega a imagem a partir do filesystem
        image = lib.read_image(input_filepath)
    except ValueError:
        lib.error("{} arquivo dicom corrompido: {}".format(contador, file))
        continue

    lib.plot("imagem {}".format(contador), image)

    # obtem as features
    feature = tem_ventriculo.extract(image)
    feature_vector.append(feature)
    print("imagem {} \n Tem ventriculo: {}".format(contador, feature))

    # feature_vector.append(qtd_hemorragia.extract(image))

    contador += 1
    if (contador >= max_records): exit()
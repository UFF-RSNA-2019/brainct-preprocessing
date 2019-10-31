import feature_tem_ventriculo
import feature_qtd_hemorragia
import feature_area_hemorragia
import lib

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
    return feature_vector

imagens = lib.get_train_images()
for image in imagens:
    features = extrai_features(image[1])
    print("image: {}, features: {}".format(image[0], features))



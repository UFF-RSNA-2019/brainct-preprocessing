import lib
import cv2
import imutils
import numpy as np
import sys
import segmentation
from scipy.spatial.distance import directed_hausdorff

def extract(ossos, hemorragias, contornos):

    # variaveis_calculo_features
    MIN_DISTANCIA = 0.0
    MAX_DISTANCIA = 400.0
    distancia_media = 0.0
    numerador_area_media_hemo = 0.0
    numerador_dist_media_hemo = 0.0
    denominador = 0.0
    area_media = 0.0
    distancia_minima = MAX_DISTANCIA # invertido intencionalmente para permitir o cálculo abaixo
    distancia_maxima = MIN_DISTANCIA # invertido intencionalmente para permitir o cálculo abaixo

    # computa as variáveis para calculo das features
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        pontos_contornos = np.vstack(contorno).squeeze()
        pontos_ossos = np.column_stack(np.where(ossos > 0))
        distancia = directed_hausdorff(pontos_ossos, pontos_contornos)[0]
        if distancia > distancia_maxima: distancia_maxima = distancia
        if distancia < distancia_minima: distancia_minima = distancia
        numerador_dist_media_hemo += distancia
        numerador_area_media_hemo += area
        denominador += 1
    if denominador > 0:
        area_media = numerador_area_media_hemo / denominador
        distancia_media = numerador_dist_media_hemo / denominador

    # feature_qtde_hemorragias
    quantidade_hemorragias = len(contornos)
    MIN_HEMORRAGIAS = 0.0
    MAX_HEMORRAGIAS = 20.0
    if quantidade_hemorragias > MAX_HEMORRAGIAS:
        quantidade_hemorragias = MAX_HEMORRAGIAS
    feat_qtde_hemorragias = (quantidade_hemorragias - MIN_HEMORRAGIAS) / (MAX_HEMORRAGIAS - MIN_HEMORRAGIAS)

    # feature_area_total_hemorragias
    area_hemorragias = np.sum(hemorragias, axis=None)
    MIN_AREA_TOTAL = 0.0
    MAX_AREA_TOTAL = 20000.0
    if area_hemorragias > MAX_AREA_TOTAL:
        area_hemorragias = MAX_AREA_TOTAL
    feat_area_hemorragias = (area_hemorragias - MIN_AREA_TOTAL) / (MAX_AREA_TOTAL - MIN_AREA_TOTAL)

    # feature_area_media # desativado pois esta retornando valores nao confiáveis
    # MIN_MEDIA = 0.0
    # MAX_MEDIA = 4000.0
    # print(area_media)
    # if area_media > MAX_MEDIA:
    #     area_media = MAX_MEDIA
    # feat_area_media = (area_media - MIN_MEDIA) / (MAX_MEDIA - MIN_MEDIA)

    # feature_distancia_media (distancia media da hemorragia para o osso)
    if distancia_media > MAX_DISTANCIA:
        distancia_media = MAX_DISTANCIA
    feat_dist_media = (distancia_media - MIN_DISTANCIA) / (MAX_DISTANCIA - MIN_DISTANCIA)

    # feature_distancia_minima (distancia minima da hemorragia para o osso)
    if distancia_minima < MIN_DISTANCIA:
        distancia_minima = MIN_DISTANCIA
    feat_dist_minima = (distancia_minima - MIN_DISTANCIA) / (MAX_DISTANCIA - MIN_DISTANCIA)

    # feature_distancia_maxima (distancia maxima da hemorragia para o osso)
    if distancia_maxima > MAX_DISTANCIA:
        distancia_maxima = MAX_DISTANCIA
    feat_dist_maxima = (distancia_maxima - MIN_DISTANCIA) / (MAX_DISTANCIA - MIN_DISTANCIA)

    return [feat_qtde_hemorragias, feat_area_hemorragias, feat_dist_media, feat_dist_minima, feat_dist_maxima]


if __name__ == "__main__":

    imagens = lib.get_train_images()
    for image in imagens:
        if image[1].shape == (512,512):
            ossos, hemorragias, contornos = segmentation.execute(image[1])
            features = extract(ossos, hemorragias, contornos)
            print("image: {}, feature: {}".format(image[0], features))
    print("Done")

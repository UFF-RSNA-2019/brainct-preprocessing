import lib
import cv2
import imutils
import numpy as np
import sys

def execute(image):

    # 1. hemorragia
    # limiarizacao
    hemorragia = lib.hemorrage_threshold(image)

    # verifica as áreas em que encontrou hemorragia
    mask_hemorragia = np.zeros_like(hemorragia).astype('uint8')
    mask_hemorragia[hemorragia > lib.DICOM_MIN] = 1
    cnts = cv2.findContours(mask_hemorragia, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # mantém as áreas de hemorragia com tamanho mínimo
    contornos = []
    for contorno in cnts:
        area = cv2.contourArea(contorno)
        if area > 100: # estruturas com area menor que este valor nao sao consideradas
            # print(area) # debug
            contornos.append(contorno)


    # 2. ventriculos
    # # limiarizacao
    # ventriculo = lib.ventriculo_threshold(image)
    #
    # # verifica as áreas em que encontrou ventriculo
    # mask_ventriculo = np.zeros_like(ventriculo).astype('uint8')
    # mask_ventriculo[ventriculo > lib.DICOM_MIN] = 1
    # cnts = cv2.findContours(mask_ventriculo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #
    # # mantém as áreas de ventriculo com tamanho mínimo
    # contornos_ventriculo = []
    # for contorno in cnts:
    #     area = cv2.contourArea(contorno)
    #     if area > 300: # estruturas com area menor que este valor nao sao consideradas
    #         # print(area) # debug
    #         contornos_ventriculo.append(contorno)


    # 3. cranio
    # segmenta área do crânio
    colorized, otsu, thresholds = lib.multiotsu(image, 3)
    mask_osso = np.zeros((512, 512))
    mask_osso[otsu == 2] = 1
    ossos = cv2.erode(mask_osso, None, iterations=2)
    # thresh = cv2.dilate(mask_osso, None, iterations=10)
    # ossos = image * mask_ossos # debug

    # obtem extremidade do cranio:
    mask_osso_t = np.transpose(ossos)
    cnt = np.column_stack(np.where(mask_osso_t > 0))
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # debug
    # lib.plot("bound box".format(id), image) # debug

    # mantém apenas os contornos que estao dentro das extremidades do cranio
    contornos_hemorragia = []
    for contorno in contornos:
        min = np.min(contorno, axis=0)
        max = np.max(contorno, axis=0)
        inclui = True
        if (min[0][0] < x): inclui = False
        if (min[0][1] < y): inclui = False
        if (max[0][0] > x + w): inclui = False
        if (max[0][1] > y + h): inclui = False
        if (inclui):
            contornos_hemorragia.append(contorno)
    hemorragias = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(hemorragias, contornos_hemorragia, -1, 1, -1)
    # lib.plot("{} - hemorragias do cranio".format(id), hemorragias)

    # contornos_ventriculo_contidos = []
    # for contorno in contornos_ventriculo:
    #     min = np.min(contorno, axis=0)
    #     max = np.max(contorno, axis=0)
    #     inclui = True
    #     if (min[0][0] < x): inclui = False
    #     if (min[0][1] < y): inclui = False
    #     if (max[0][0] > x + w): inclui = False
    #     if (max[0][1] > y + h): inclui = False
    #     if (inclui):
    #         contornos_ventriculo_contidos.append(contorno)
    # ventriculos = np.zeros(image.shape[:2], np.uint8)
    # cv2.drawContours(ventriculos, contornos_ventriculo_contidos, -1, 1, -1)

    # debug
    # lib.plot("{} - original".format(id), image)
    # lib.plot("{} - ossos".format(id), ossos)
    # lib.plot("{} - hemorragias".format(id), mask_hemorragia)
    # hemorragias = np.zeros(image.shape[:2], np.uint8)
    # cv2.drawContours(hemorragias, contornos_hemorragia_contidos, -1, 255, -1)
    # lib.plot("{} - hemorragias do cranio".format(id), mask)
    # debug

    return ossos, hemorragias, contornos_hemorragia



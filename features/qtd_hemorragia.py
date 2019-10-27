import lib
import cv2
import imutils
import numpy as np

def extract(image):

    roi = lib.get_roi_cabeca(image)

    # limiarizacao
    hemorragia = lib.hemorrage_threshold(roi)

    # verifica as áreas em que encontrou hemorragia
    mask_hemorragia = np.zeros((lib.SIZE, lib.SIZE)).astype('uint8')
    mask_hemorragia[hemorragia > lib.DICOM_MIN] = 1
    cnts = cv2.findContours(mask_hemorragia, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    contornos = []
    for contorno in cnts:
        area = cv2.contourArea(contorno)
        if area > 100: # estruturas com area menor que este valor nao sao consideradas
            # print(area) # debug
            contornos.append(contorno)

    # debug
    # mask  = np.zeros(image.shape[:2], np.uint8)
    # cv2.drawContours(mask , contornos, -1, 255, -1)
    # lib.plot("hemorragias", mask_hemorragia) # debug
    # lib.plot("contornos hemorragias", mask) # debug
    # debug

    qtd_hemorragias = len(contornos)

    # NORMALIZA
    MIN_HEMORRAGIA = 0.0
    MAX_HEMORRAGIA = 10.0
    if qtd_hemorragias > MAX_HEMORRAGIA:
        qtd_hemorragias = MAX_HEMORRAGIA

    # normaliza
    normalized = (qtd_hemorragias - MIN_HEMORRAGIA) / (MAX_HEMORRAGIA - MIN_HEMORRAGIA)

    return normalized

# todo: segmentar crânio e só considerar hemorragias dentro dele


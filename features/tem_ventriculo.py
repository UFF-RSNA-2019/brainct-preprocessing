import lib
import cv2
import imutils
import numpy as np

def extract(image):

    roi = lib.get_roi_cabeca(image)

    # limiarizacao
    ventriculo = lib.ventriculo_threshold(roi)

    # verifica as áreas em que encontrou ventriculo
    mask_ventriculo = np.zeros((lib.SIZE, lib.SIZE)).astype('uint8')
    mask_ventriculo[ventriculo > lib.DICOM_MIN] = 1
    cnts = cv2.findContours(mask_ventriculo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    contornos = []
    for contorno in cnts:
        area = cv2.contourArea(contorno)
        if area > 300: # estruturas com area menor que este valor nao sao consideradas
            # print(area) # debug
            contornos.append(contorno)

    # debug
    # mask  = np.zeros(image.shape[:2], np.uint8)
    # cv2.drawContours(mask , contornos, -1, 255, -1)
    # lib.plot("ventriculos", ventriculo) # debug
    # lib.plot("contornos ventriculo", mask) # debug
    # debug

    if (len(contornos) > 0):
        return 1
    else:
        return 0

# todo: segmentar crânio e só considerar ventrículo se estiver dentro dele
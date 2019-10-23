import lib

def extract(image):

    roi = lib.get_roi_cabeca(image)

    # limiarizacao
    # TODO: buscar a maior area utilizando findcountors se tiver area maior que x é porque tem o ventriculo
    ventriculo = lib.ventriculo_threshold(roi)
    ventriculo[ventriculo>0] = 1
    lib.plot("vent", ventriculo)
    print (ventriculo.sum())

    if (ventriculo.sum() > 0):
        return 1
    else:
        return 0

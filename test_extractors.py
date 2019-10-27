from features import tem_ventriculo
from features import qtd_hemorragia
from features import area_hemorragia
import lib

lib.test_extractor(tem_ventriculo.extract())
lib.test_extractor(qtd_hemorragia.extract())
lib.test_extractor(area_hemorragia.extract())
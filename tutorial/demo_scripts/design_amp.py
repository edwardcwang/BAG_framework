# -*- coding: utf-8 -*-

import pprint

from demo_package.demo_dsn.core import design_amp_cs

from ckt_dsn_ec.mos.core import MOSDBDiscrete


if __name__ == '__main__':
    vdd = 1.0
    vout = 0.7
    cload = 40e-15
    fbw = 4e9
    gain_min = 4

    w_list = [4]
    nch_conf_list = ['BAG_framework/tutorial/data/mos_char_nch']
    pch_conf_list = ['BAG_framework/tutorial/data/mos_char_pch']

    nch_db = MOSDBDiscrete(w_list, nch_conf_list, 1)
    pch_db = MOSDBDiscrete(w_list, pch_conf_list, 1)

    best_op = design_amp_cs(nch_db, pch_db, vdd, vout, cload, fbw, gain_min)
    pprint.pprint(best_op)
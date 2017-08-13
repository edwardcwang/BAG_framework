# -*- coding: utf-8 -*-

import math

from typing import Callable

import numpy as np


def design_amp_cs(nch_db, pch_db, vdd, vin, vout, cload, fbw, gain_min, env='tt', vgs_res=2e-3):
    wbw = 2 * np.pi * fbw
    n_intent_list = nch_db.get_dsn_param_values('intent')
    p_intent_list = pch_db.get_dsn_param_values('intent')

    best_sol = None
    best_op = None
    for n_intent in n_intent_list:
        nch_db.set_dsn_params(intent=n_intent)

        arg = [0, vout, vin]

        ibn = float(nch_db.get_function('ibias', env=env)(arg))
        gmn = float(nch_db.get_function('gm', env=env)(arg) / ibn)
        gdsn = float(nch_db.get_function('gds', env=env)(arg) / ibn)
        cddn = float(nch_db.get_function('cdd', env=env)(arg) / ibn)

        for p_intent in p_intent_list:
            print(n_intent, p_intent)
            pch_db.set_dsn_params(intent=p_intent)
            vgs_idx = pch_db.get_fun_arg_index('vgs')

            ibpf = pch_db.get_function('ibias', env=env)
            gdspf = pch_db.get_function('gds', env=env) / ibpf
            cddpf = pch_db.get_function('cdd', env=env) / ibpf

            vgs_min, vgs_max = ibpf.get_input_range(vgs_idx)

            gainf = gmn / (gdsn + gdspf)
            ibiasf = wbw * cload / (gdspf + gdsn - wbw * (cddpf + cddn))  # type: Callable

            num_vgs = int(math.ceil((vgs_max - vgs_min) / vgs_res))
            vgsp_vec = np.linspace(vgs_min, vgs_max, num_vgs)
            arg = np.zeros([num_vgs, 3])
            arg[:, 1] = vout - vdd
            arg[:, 2] = vgsp_vec

            gain_vals = gainf(arg)
            ibias_vals = ibiasf(arg)

            idx_vec = np.logical_and(gain_vals >= gain_min, ibias_vals >= 0)
            vgsp_vec = vgsp_vec[idx_vec]
            ibias_vals = ibias_vals[idx_vec]
            gain_vals = gain_vals[idx_vec]
            if ibias_vals.size > 0:
                opt_idx = np.argmin(ibias_vals)
                vgsp_sol = vgsp_vec[opt_idx]
                cur_ibias = ibias_vals[opt_idx]
                cur_gain = gain_vals[opt_idx]
                cur_op = n_intent, p_intent, vgsp_sol, cur_ibias, cur_gain
                print('sol found')
                print(cur_op)
                if best_sol is None or cur_ibias < best_sol:
                    best_sol = cur_ibias
                    best_op = cur_op
            else:
                print('sol not found')

    print(best_sol)
    print(best_op)

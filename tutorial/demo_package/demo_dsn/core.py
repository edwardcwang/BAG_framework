# -*- coding: utf-8 -*-

import math

from typing import Callable

import numpy as np
import scipy.interpolate as interp


def change_x_to_ibias(mos_db, xmat, num_samp=200):
    ib_mat = mos_db.get_function('ibias')(xmat)

    min_ibias = np.max(np.min(ib_mat, axis=0))
    max_ibias = np.min(np.max(ib_mat, axis=0))

    ib_vec = np.linspace(min_ibias, max_ibias, num_samp)
    num_corners = ib_mat.shape[1]
    results = {}
    new_shape = (num_samp, num_corners)
    for fun_name in ('gm', 'gds', 'cdd', 'css'):
        fun_mat = mos_db.get_function(fun_name)(xmat)
        new_mat = np.empty(new_shape)
        for c_idx in range(num_corners):
            cur_f = interp.interp1d(ib_mat[:, c_idx], fun_mat[:, c_idx])
            new_mat[:, c_idx] = cur_f(ib_vec)

        results[fun_name] = new_mat

    for idx, x_name in enumerate(('vbs', 'vds', 'vgs')):
        cur_x = xmat[:, idx]
        new_mat = np.empty(new_shape)
        for c_idx in range(num_corners):
            cur_f = interp.interp1d(ib_mat[:, c_idx], cur_x)
            new_mat[:, c_idx] = cur_f(ib_vec)

        results[x_name] = new_mat

    results['ibias'] = ib_vec
    return results


def get_xmat_vgs(mos_db, vgs_res, vbs, vds):
    vgs_idx = mos_db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = mos_db.get_function('ibias').get_input_range(vgs_idx)
    num_samp = int(math.ceil((vgs_max - vgs_min) / vgs_res))
    xmat = np.empty((num_samp, 3))
    xmat[:, 0] = vbs
    xmat[:, 1] = vds
    xmat[:, 2] = np.linspace(vgs_min, vgs_max, num_samp)

    return xmat


def design_amp_cs(nch_db, pch_db, vdd, vout, cload, fbw, gain_min, vgs_res=2e-3, num_ib_samp=200):
    wbw = 2 * np.pi * fbw
    n_intent_list = nch_db.get_dsn_param_values('intent')
    p_intent_list = pch_db.get_dsn_param_values('intent')

    best_sol = None
    best_op = None
    mat_shape = (num_ib_samp, num_ib_samp)
    for n_intent in n_intent_list:
        nch_db.set_dsn_params(intent=n_intent)
        nch_dict = change_x_to_ibias(nch_db, get_xmat_vgs(nch_db, vgs_res, 0, vout), num_samp=num_ib_samp)

        ibn_vec = nch_dict['ibias'].reshape(num_ib_samp, 1)
        gmn_mat = nch_dict['gm'] / ibn_vec
        gdsn_mat = nch_dict['gds'] / ibn_vec
        cddn_mat = nch_dict['cdd'] / ibn_vec
        num_corners = gmn_mat.shape[1]

        ibn_mat = np.broadcast_to(ibn_vec, mat_shape)

        vgsn_mat = np.broadcast_to(nch_dict['vgs'][:, np.newaxis, :], (num_ib_samp, num_ib_samp, num_corners))

        for p_intent in p_intent_list:
            pch_db.set_dsn_params(intent=p_intent)
            pch_dict = change_x_to_ibias(pch_db, get_xmat_vgs(pch_db, vgs_res, 0, vout - vdd), num_samp=num_ib_samp)

            ibp_vec = pch_dict['ibias'].reshape(num_ib_samp, 1)
            gdsp_mat = pch_dict['gds'] / ibp_vec
            cddp_mat = pch_dict['cdd'] / ibp_vec

            ibp_mat = np.broadcast_to(ibp_vec.reshape(1, num_ib_samp), mat_shape)

            # calculate gain and total current across corners and ibp/ibn
            gain_mat = np.empty((num_ib_samp, num_ib_samp, num_corners))
            itot_mat = np.empty((num_ib_samp, num_ib_samp, num_corners))
            for idx in range(num_corners):
                # get SS parameteres for this corner.  Reshape to enable numpy broadcasting tricks.
                gmn_cur = gmn_mat[:, idx].reshape(num_ib_samp, 1)
                gdsn_cur = gdsn_mat[:, idx].reshape(num_ib_samp, 1)
                cddn_cur = cddn_mat[:, idx].reshape(num_ib_samp, 1)
                gdsp_cur = gdsp_mat[:, idx].reshape(1, num_ib_samp)
                cddp_cur = cddp_mat[:, idx].reshape(1, num_ib_samp)

                gain_mat[..., idx] = gmn_cur / (gdsn_cur + gdsp_cur)
                itot_mat[..., idx] = wbw * cload / (gdsp_cur + gdsn_cur - wbw * (cddp_cur + cddn_cur))

            # get minimum gain across corners
            gain_mat = np.min(gain_mat, axis=2)
            # get maximum/minimum total current across corners
            imin_mat = np.min(itot_mat, axis=2)
            imax_mat = np.max(itot_mat, axis=2)

            # get indices that satisfies constants
            idx_mat = (gain_mat >= gain_min) & (imin_mat >= 0)
            if np.any(idx_mat):
                # there exists some solutions
                gain_vec = gain_mat[idx_mat]
                imax_vec = imax_mat[idx_mat]
                vgsn_vec = vgsn_mat[idx_mat, :]
                ibp_vec = ibp_mat[idx_mat]
                ibn_vec = ibn_mat[idx_mat]

                opt_idx = np.argmin(imax_vec)
                cur_vgsn = vgsn_vec[opt_idx, :]
                cur_ibn = ibn_vec[opt_idx]
                cur_ibp = ibp_vec[opt_idx]
                cur_ibias = imax_vec[opt_idx]
                cur_gain = gain_vec[opt_idx]
                cur_op = n_intent, p_intent, cur_vgsn, cur_ibn, cur_ibp, cur_ibias, cur_gain
                print('sol_found')
                print(cur_op)
                if best_sol is None or cur_ibias < best_sol:
                    best_sol = cur_ibias
                    best_op = cur_op
            else:
                print('sol not found')

    print(best_sol)
    print(best_op)

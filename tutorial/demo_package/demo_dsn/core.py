# -*- coding: utf-8 -*-

import math

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt


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


def change_x_to_ibias_mirror(mos_db, vgs_res, vbs, vds, num_ib_samp=200):
    vgs_idx = mos_db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = mos_db.get_function('ibias').get_input_range(vgs_idx)
    num_samp = int(math.ceil((vgs_max - vgs_min) / vgs_res))
    xmat = np.empty((num_samp, 3))
    vgs_vec = np.linspace(vgs_min, vgs_max, num_samp)
    xmat[:, 0] = vbs
    xmat[:, 1] = vgs_vec
    xmat[:, 2] = vgs_vec

    ib_mat = mos_db.get_function('ibias')(xmat)
    min_ibias = np.max(np.min(ib_mat, axis=0))
    max_ibias = np.min(np.max(ib_mat, axis=0))
    ib_vec = np.linspace(min_ibias, max_ibias, num_ib_samp)

    num_corners = ib_mat.shape[1]
    new_shape = (num_ib_samp, num_corners)
    results = {}
    vgs_mat = np.empty(new_shape)
    for c_idx in range(num_corners):
        cur_f = interp.interp1d(ib_mat[:, c_idx], vgs_vec)
        vgs_mat[:, c_idx] = cur_f(ib_vec)

    results['vgs'] = vgs_mat

    xmat = np.empty((num_ib_samp, 3))
    xmat[:, 1] = vds
    for fun_name in ('gm', 'gds', 'cdd', 'css'):
        new_mat = np.empty(new_shape)
        for c_idx, cur_fun in enumerate(mos_db.get_function_list(fun_name)):
            xmat[:, 2] = vgs_mat[:, c_idx]
            new_mat[:, c_idx] = cur_fun(xmat)

        results[fun_name] = new_mat

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
    intent_n_list = nch_db.get_dsn_param_values('intent')
    intent_p_list = pch_db.get_dsn_param_values('intent')

    best_sol = None
    best_op = None
    mat_shape = (num_ib_samp, num_ib_samp)
    for intent_n in intent_n_list:
        nch_db.set_dsn_params(intent=intent_n)
        nch_dict = change_x_to_ibias(nch_db, get_xmat_vgs(nch_db, vgs_res, 0, vout), num_samp=num_ib_samp)

        ibn_vec = nch_dict['ibias'].reshape(num_ib_samp, 1)
        gmn_mat = nch_dict['gm'] / ibn_vec
        gdsn_mat = nch_dict['gds'] / ibn_vec
        cddn_mat = nch_dict['cdd'] / ibn_vec
        num_corners = gmn_mat.shape[1]

        ibn_mat = np.broadcast_to(ibn_vec, mat_shape)

        for intent_p in intent_p_list:
            pch_db.set_dsn_params(intent=intent_p)
            pch_dict = change_x_to_ibias_mirror(pch_db, vgs_res, 0, vout - vdd, num_ib_samp=num_ib_samp)

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
            worst_gain_mat = np.min(gain_mat, axis=2)
            # get maximum/minimum total current across corners
            imin_mat = np.min(itot_mat, axis=2)
            imax_mat = np.max(itot_mat, axis=2)

            # get indices that satisfies constants
            idx_mat = (worst_gain_mat >= gain_min) & (imin_mat >= 0)
            if np.any(idx_mat):
                # there exists some solutions
                imax_vec = imax_mat[idx_mat]
                ibp_vec = ibp_mat[idx_mat]
                ibn_vec = ibn_mat[idx_mat]

                opt_idx = np.argmin(imax_vec)
                cur_ibn = ibn_vec[opt_idx]
                cur_ibp = ibp_vec[opt_idx]
                cur_ibias = imax_vec[opt_idx]
                if best_sol is None or cur_ibias < best_sol:
                    best_sol = cur_ibias
                    best_op = intent_n, intent_p, cur_ibn, cur_ibp, cur_ibias

    # got optimal itot, compute sizing
    intent_n, intent_p, iunit_n, iunit_p, itot = best_op

    fgn = int(math.ceil(itot / iunit_n / 2)) * 2
    fgp = int(math.ceil(itot / iunit_p / 2)) * 2

    # compute final performance
    nch_db.set_dsn_params(intent=intent_n)
    pch_db.set_dsn_params(intent=intent_p)

    # compute pmos SS parameters across corners
    iref = iunit_p * fgp
    vgs_idx = pch_db.get_fun_arg_index('vgs')
    vgsp_list, gdsp_list, cddp_list, ibias_list = [], [], [], []
    for ibpf, gdspf, cddpf in zip(pch_db.get_function_list('ibias'),
                                  pch_db.get_function_list('gds'),
                                  pch_db.get_function_list('cdd')):
        vgs_min, vgs_max = ibpf.get_input_range(vgs_idx)

        def zero_fun(vgs):
            arg = pch_db.get_fun_arg(vbs=0, vds=vgs, vgs=vgs)
            return ibpf(arg) - iunit_p

        vgs_sol = sciopt.brentq(zero_fun, vgs_min, vgs_max)
        vgsp_list.append(vgs_sol)
        parg = pch_db.get_fun_arg(vbs=0, vds=vout - vdd, vgs=vgs_sol)
        gdsp_list.append(float(gdspf(parg)) * fgp)
        cddp_list.append(float(cddpf(parg)) * fgp)
        ibias_list.append(float(ibpf(parg)) * fgp)

    # compute nmos SS parameters across corners
    vgs_idx = nch_db.get_fun_arg_index('vgs')
    vgsn_list, gmn_list, gdsn_list, cddn_list = [], [], [], []
    for itarg, ibnf, gmnf, gdsnf, cddnf in zip(ibias_list, nch_db.get_function_list('ibias'),
                                               nch_db.get_function_list('gm'),
                                               nch_db.get_function_list('gds'),
                                               nch_db.get_function_list('cdd')):
        vgs_min, vgs_max = ibnf.get_input_range(vgs_idx)

        def zero_fun(vgs):
            arg = nch_db.get_fun_arg(vbs=0, vds=vout, vgs=vgs)
            return ibnf(arg) * fgn - itarg

        vgs_sol = sciopt.brentq(zero_fun, vgs_min, vgs_max)
        vgsn_list.append(vgs_sol)
        narg = nch_db.get_fun_arg(vbs=0, vds=vout, vgs=vgs_sol)
        gmn_list.append(float(gmnf(narg)) * fgn)
        gdsn_list.append(float(gdsnf(narg)) * fgn)
        cddn_list.append(float(cddnf(narg)) * fgn)

    # compute amplifier parameters
    gain_list, bw_list, ro_list = [], [], []
    for gmn, gdsn, cddn, gdsp, cddp in zip(gmn_list, gdsn_list, cddn_list,
                                           gdsp_list, cddp_list):
        gain_list.append(gmn / (gdsn + gdsp))
        bw_list.append((gdsn + gdsp) / (cddn + cddp + cload) / 2 / np.pi / 1e9)
        ro_list.append(1 / (gdsn + gdsp))

    return dict(
        iref=iref,
        ibias=ibias_list,
        gain=gain_list,
        bw=bw_list,
        ro=ro_list,
        intent_n=intent_n,
        fgn=fgn,
        vgsn=vgsn_list,
        gmn=gmn_list,
        gdsn=gdsn_list,
        cddn=cddn_list,
        intent_p=intent_p,
        fgp=fgp,
        vgsp=vgsp_list,
        gdsp=gdsp_list,
        cddp=cddp_list,
    )

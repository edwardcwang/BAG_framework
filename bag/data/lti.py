# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""This module defines functions and classes useful for characterizing linear time-invariant circuits.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
from scipy.signal.ltisys import StateSpaceContinuous, TransferFunctionContinuous


class LTICircuit(object):
    """A class that models a linear-time-invariant circuit.

    This class computes AC transfer functions for linear-time-invariant circuits.

    Note: Since this class work with AC transfer functions, 'gnd' in this circuit is AC ground.
    """

    _float_min = np.finfo(np.float64).eps

    def __init__(self):
        # type: (List[str]) -> None
        self._num_n = 0
        self._gmat_data = {}
        self._cmat_data = {}
        self._vcvs_list = []
        self._ind_data = {}
        self._node_id = {'gnd': -1}

    def _get_node_id(self, name):
        # type: (str) -> int
        if name not in self._node_id:
            ans = self._num_n
            self._node_id[name] = ans
            self._num_n += 1
            return ans
        else:
            return self._node_id[name]

    @staticmethod
    def _add(mat, key, val):
        if key in mat:
            mat[key] += val
        else:
            mat[key] = val

    def add_res(self, res, p_name, n_name):
        # type: (float, str, str) -> None
        """Adds a resistor to the circuit.

        Parameters
        ----------
        res : float
            the resistance value, in Ohms.
        p_name : str
            the positive terminal net name.
        n_name : str
            the negative terminal net name.
        """
        node_p = self._get_node_id(p_name)
        node_n = self._get_node_id(n_name)

        if node_p == node_n:
            return
        if node_p < node_n:
            node_p, node_n = node_n, node_p

        # avoid 0 resistance.
        res_sgn = 1 if res >= 0 else -1
        g = res_sgn / max(abs(res), self._float_min)
        self._add(self._gmat_data, (node_p, node_p), g)
        if node_n >= 0:
            self._add(self._gmat_data, (node_p, node_n), -g)
            self._add(self._gmat_data, (node_n, node_p), -g)
            self._add(self._gmat_data, (node_n, node_n), g)

    def add_vccs(self, gm, p_name, n_name, cp_name, cn_name='gnd'):
        # type: (float, str, str, str, str) -> None
        """Adds a voltage controlled current source to the circuit.

        Parameters
        ----------
        gm : float
            the gain of the voltage controlled current source, in Siemens.
        p_name : str
            the terminal that the current flows out of.
        n_name : str
            the terminal that the current flows in to.
        cp_name : str
            the positive voltage control terminal.
        cn_name : str
            the negative voltage control terminal.  Defaults to 'gnd'.
        """
        node_p = self._get_node_id(p_name)
        node_n = self._get_node_id(n_name)
        node_cp = self._get_node_id(cp_name)
        node_cn = self._get_node_id(cn_name)

        if node_p == node_n or node_cp == node_cn:
            return

        if node_cp >= 0:
            if node_p >= 0:
                self._add(self._gmat_data, (node_p, node_cp), gm)
            if node_n >= 0:
                self._add(self._gmat_data, (node_n, node_cp), -gm)
        if node_cn >= 0:
            if node_p >= 0:
                self._add(self._gmat_data, (node_p, node_cn), -gm)
            if node_n >= 0:
                self._add(self._gmat_data, (node_n, node_cn), gm)

    def add_vcvs(self, gain, p_name, n_name, cp_name, cn_name='gnd'):
        # type: (float, str, str, str, str) -> None
        """Adds a voltage controlled voltage source to the circuit.

        Parameters
        ----------
        gain : float
            the gain of the voltage controlled voltage source.
        p_name : str
            the positive terminal of the output voltage source.
        n_name : str
            the negative terminal of the output voltage source.
        cp_name : str
            the positive voltage control terminal.
        cn_name : str
            the negative voltage control terminal.  Defaults to 'gnd'.
        """
        node_p = self._get_node_id(p_name)
        node_n = self._get_node_id(n_name)
        node_cp = self._get_node_id(cp_name)
        node_cn = self._get_node_id(cn_name)

        if node_p == node_n:
            raise ValueError('positive and negative terminal of a vcvs cannot be the same.')
        if node_cp == node_cn:
            raise ValueError('positive and negative control terminal of a vcvs cannot be the same.')
        if node_p < node_n:
            # flip nodes so we always have node_p > node_n, to guarantee node_p >= 0
            node_p, node_n, node_cp, node_cn = node_n, node_p, node_cn, node_cp

        self._vcvs_list.append((node_p, node_n, node_cp, node_cn, gain))

    def add_cap(self, cap, p_name, n_name):
        # type: (float, str, str) -> None
        """Adds a capacitor to the circuit.

        Parameters
        ----------
        cap : float
            the capacitance value, in Farads.
        p_name : str
            the positive terminal net name.
        n_name : str
            the negative terminal net name.
        """
        node_p = self._get_node_id(p_name)
        node_n = self._get_node_id(n_name)

        if node_p == node_n:
            return
        if node_p < node_n:
            node_p, node_n = node_n, node_p

        self._add(self._cmat_data, (node_p, node_p), cap)
        if node_n >= 0:
            self._add(self._cmat_data, (node_p, node_n), -cap)
            self._add(self._cmat_data, (node_n, node_p), -cap)
            self._add(self._cmat_data, (node_n, node_n), cap)

    def add_ind(self, ind, p_name, n_name):
        # type: (float, str, str) -> None
        """Adds an inductor to the circuit.

        Parameters
        ----------
        ind : float
            the inductance value, in Henries.
        p_name : str
            the positive terminal net name.
        n_name : str
            the negative terminal net name.
        """
        node_p = self._get_node_id(p_name)
        node_n = self._get_node_id(n_name)

        if node_p == node_n:
            return
        if node_p < node_n:
            key = node_n, node_p
        else:
            key = node_p, node_n

        if key not in self._ind_data:
            self._ind_data[key] = ind
        else:
            self._ind_data[key] = 1.0 / (1.0 / ind + 1.0 / self._ind_data[key])

    def add_transistor(self, tran_info, d_name, g_name, s_name, b_name='gnd', fg=1):
        # type: (Dict[str, np.ndarray], str, str, str, str, int) -> None
        """Adds a small signal transistor model to the circuit.

        Parameters
        ----------
        tran_info : Dict[str, np.ndarray]
            a dictionary of 1-finger transistor small signal parameters.  Should contain gm, gds, gb,
            cgd, cgs, cgb, cds, cdb, and csb.
        d_name : str
            drain net name.
        g_name : str
            gate net name.
        s_name : str
            source net name.
        b_name : str
            body net name.  Defaults to 'gnd'.
        fg : int
            number of transistor fingers.
        """
        gm = tran_info['gm'][0] * fg
        ro = 1 / (tran_info['gds'][0] * fg)
        gb = tran_info['gb'][0] * fg
        cgd = tran_info['cgd'][0] * fg
        cgs = tran_info['cgs'][0] * fg
        cgb = tran_info['cgb'][0] * fg
        cds = tran_info['cds'][0] * fg
        cdb = tran_info['cdb'][0] * fg
        csb = tran_info['csb'][0] * fg

        self.add_vccs(gm, d_name, s_name, g_name, s_name)
        self.add_res(ro, d_name, s_name)
        self.add_vccs(gb, d_name, s_name, b_name, s_name)
        self.add_cap(cgd, g_name, d_name)
        self.add_cap(cgs, g_name, s_name)
        self.add_cap(cgb, g_name, b_name)
        self.add_cap(cds, d_name, s_name)
        self.add_cap(cdb, d_name, b_name)
        self.add_cap(csb, s_name, b_name)

    @classmethod
    def _count_rank(cls, diag):
        # type: (np.ndarray) -> int
        diag_abs = np.abs(diag)
        float_min = cls._float_min
        rank_tol = diag_abs[0] * diag.size * float_min
        rank_cnt = diag_abs > rank_tol  # type: np.ndarray
        return np.count_nonzero(rank_cnt)

    @classmethod
    def _solve_gx_bw(cls, g, b):
        # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Solve the equation G*x + B*[w, w', ...].T = 0 for x.

        Finds matrix Ka, Kw such that x = Ka * a + Kw * [w, w', ...].T solves
        the given equation for any value of a.

        Parameters
        ----------
        g : np.ndarray
            the G matrix, with shape (M, N) and M < N.
        b : np.ndarray
            the B matrix.

        Returns
        -------
        ka : np.ndarray
            the Ky matrix.
        kw : np.ndarray
            the Kw matrix.
        """
        # G = U*S*Vh
        u, s, vh = scipy.linalg.svd(g, full_matrices=True, overwrite_a=True)
        # let B=Uh*B, so now S*Vh*x + B*w = 0
        b = u.T.dot(b)
        # let y = Vh*x, or x = V*y, so now S*y + U*B*w = 0
        v = vh.T
        # truncate the bottom 0 part of S, now S_top*y_top + B_top*w = 0
        rank = cls._count_rank(s)
        # check bottom part of B.  If not 0, there's no solution
        b_abs = np.abs(b)
        zero_tol = np.amax(b_abs) * cls._float_min
        if np.count_nonzero(b_abs[rank:, :] > zero_tol) > 0:
            raise ValueError('B matrix bottom is not zero.  This circuit has no solution.')
        b_top = b[:rank, :]
        s_top_inv = 1 / s[:rank]  # type: np.ndarray
        s_top_inv = np.diag(s_top_inv)
        # solving, we get y_top = -S_top^-1*B_top*w = Ku*w
        kw = s_top_inv.dot(-b_top)
        # now x = V*y = Vl*y_top + Vr*y_bot = Vr*y_bot + Vl*Kw*w = Ky*y_bot = Kw*w
        vl = v[:, :rank]
        vr = v[:, rank:]
        kw = vl.dot(kw)
        return vr, kw

    @classmethod
    def _reduce_state_space(cls, g, c, b, d, e, ndim_w):
        """Reduce state space variables.

        Given the state equation G*x + C*x' + B*[w, w', w'', ...].T = 0, and
        y = D*x + E*[w, w', w'', ...].T, check if C is full rank.  If not,
        we compute new G, C, and B matrices with reduced dimensions.
        """
        # step 0: perform QR factorization of C, and obtain rank
        q, r, p = scipy.linalg.qr(c, pivoting=True)
        rank = cls._count_rank(np.diag(r))
        # step 0A: multiply through so that c is upper triangular
        qh = q.T
        c = r
        g = qh.dot(g[:, p])
        d = d[:, p]
        b = qh.dot(b)
        while rank < r.shape[0]:
            # step 2: eliminate x' term by looking at bottom part of matrices
            ctop = c[:rank, :]
            gtop = g[:rank, :]
            gbot = g[rank:, :]
            btop = b[:rank, :]
            bbot = b[rank:, :]
            # step 3: find ka and kw from bottom
            ka, kw = cls._solve_gx_bw(gbot, bbot)
            # step 4: substitute x = ka * a + kw * [w, w', w'', ...].T
            g = gtop.dot(ka)
            c = ctop.dot(ka)
            b = np.zeros((btop.shape[0], btop.shape[1] + ndim_w))
            b[:, :btop.shape[1]] = btop + gtop.dot(kw)
            b[:, ndim_w:] += ctop.dot(kw)
            enew = np.zeros((e.shape[0], e.shape[1] + ndim_w))
            enew[:, :-ndim_w] = e + d.dot(kw)
            e = enew
            d = d.dot(ka)
            # step 5: update QR factorization
            q, r, p = scipy.linalg.qr(c, pivoting=True)
            rank = cls._count_rank(np.diag(r))
            qh = q.T
            c = r
            g = qh.dot(g[:, p])
            d = d[:, p]
            b = qh.dot(b)

        g, c, b, d, e = cls._simplify(g, c, b, d, e, ndim_w)
        return g, c, b, d, e

    @classmethod
    def _simplify(cls, g, c, b, d, e, ndim_w):
        """Eliminate input derivatives by re-defining state variables.
        """
        while b.shape[1] > ndim_w:
            kw = scipy.linalg.solve_triangular(c, b[:, ndim_w:])
            bnew = np.dot(g, -kw)
            bnew[:, :ndim_w] += b[:, :ndim_w]
            b = bnew
            e[:, :kw.shape[1]] -= d.dot(kw)
        return g, c, b, d, e

    def _build_mna_matrices(self, inputs, outputs, in_type='v'):
        # type: (Union[str, List[str]], Union[str, List[str]], str) -> Tuple[np.ndarray, ...]
        """Create and return MNA matrices representing this circuit.

        Parameters
        ----------
        inputs : Union[str, List[str]]
            the input voltage/current node name(s).
        outputs : Union[str, List[str]]
            the output voltage node name(s).
        in_type : str
            set to 'v' for input voltage sources.  Otherwise, current sources.

        Returns
        -------
        g : np.ndarray
            the conductance matrix
        c : np.ndarray
            the capacitance/inductance matrix.
        b : np.ndarray
            the input-to-state matrix.
        d : np.ndarray
            the state-to-output matrix.
        e : np.ndarray
            the input-to-output matrix.
        """
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            node_ins = [self._node_id[name] for name in inputs]
        else:
            node_ins = [self._node_id[inputs]]
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            node_outs = [self._node_id[name] for name in outputs]
        else:
            node_outs = [self._node_id[outputs]]

        is_voltage = (in_type == 'v')

        # step 1: construct matrices
        gdata, grows, gcols = [], [], []
        cdata, crows, ccols = [], [], []
        # step 1A: gather conductors/vccs
        for (ridx, cidx), gval in self._gmat_data.items():
            gdata.append(gval)
            grows.append(ridx)
            gcols.append(cidx)
        # step 1B: gather capacitors
        for (ridx, cidx), cval in self._cmat_data.items():
            cdata.append(cval)
            crows.append(ridx)
            ccols.append(cidx)
        # step 1C: gather inductors
        num_states = self._num_n
        for (node_p, node_n), lval in self._ind_data.items():
            gdata.append(1)
            grows.append(node_p)
            gcols.append(num_states)
            gdata.append(1)
            grows.append(num_states)
            gcols.append(node_p)
            if node_n >= 0:
                gdata.append(-1)
                grows.append(node_n)
                gcols.append(num_states)
                gdata.append(-1)
                grows.append(num_states)
                gcols.append(node_n)
            cdata.append(-lval)
            crows.append(num_states)
            ccols.append(num_states)
            num_states += 1
        # step 1D: add currents from vcvs
        for node_p, node_n, node_cp, node_cn, gain in self._vcvs_list:
            # vcvs means vp - vn - A*vcp + A*vcn = 0, and current flows from p to n
            # current flowing out of p
            gdata.append(1)
            grows.append(node_p)
            gcols.append(num_states)
            # voltage of p
            gdata.append(1)
            grows.append(num_states)
            gcols.append(node_p)
            if node_n >= 0:
                # current flowing into n
                gdata.append(-1)
                grows.append(node_n)
                gcols.append(num_states)
                # voltage of n
                gdata.append(-1)
                grows.append(num_states)
                gcols.append(node_n)
            if node_cp >= 0:
                # voltage of cp
                gdata.append(-gain)
                grows.append(num_states)
                gcols.append(node_cp)
            if node_cn >= 0:
                # voltage of cn
                gdata.append(gain)
                grows.append(num_states)
                gcols.append(node_cn)
            num_states += 1

        ndim_in = len(node_ins)
        if is_voltage:
            # step 1E: add current/voltage from input voltage source
            b = np.zeros((num_states + ndim_in, ndim_in))
            for in_idx, node_in in enumerate(node_ins):
                gdata.append(1)
                grows.append(node_in)
                gcols.append(num_states)
                gdata.append(-1)
                grows.append(num_states)
                gcols.append(node_in)
                b[num_states + in_idx, in_idx] = 1
            num_states += ndim_in
        else:
            # inject current to node_in
            b = np.zeros((num_states, ndim_in))
            for in_idx, node_in in enumerate(node_ins):
                b[node_in, in_idx] = -1

        # step 2: create matrices
        shape = (num_states, num_states)
        g = scipy.sparse.csc_matrix((gdata, (grows, gcols)), shape=shape).todense().A
        c = scipy.sparse.csc_matrix((cdata, (crows, ccols)), shape=shape).todense().A
        ndim_out = len(node_outs)
        d = scipy.sparse.csc_matrix((np.ones(ndim_out), (np.arange(ndim_out), node_outs)),
                                    shape=(ndim_out, num_states)).todense().A
        e = np.zeros((ndim_out, ndim_in))

        return g, c, b, d, e

    def get_state_space(self, inputs, outputs, in_type='v'):
        # type: (Union[str, List[str]], Union[str, List[str]], str) -> StateSpaceContinuous
        """Compute the state space model from the given inputs to outputs.

        Parameters
        ----------
        inputs : Union[str, List[str]]
            the input voltage/current node name(s).
        outputs : Union[str, List[str]]
            the output voltage node name(s).
        in_type : str
            set to 'v' for input voltage sources.  Otherwise, current sources.

        Returns
        -------
        system : StateSpaceContinuous
            the scipy state space object.  See scipy.signal package on how to use this object.
        """
        g0, c0, b0, d0, e0 = self._build_mna_matrices(inputs, outputs, in_type)
        ndim_in = e0.shape[1]
        g, c, b, d, e = self._reduce_state_space(g0, c0, b0, d0, e0, ndim_in)
        amat = scipy.linalg.solve_triangular(c, -g)
        bmat = scipy.linalg.solve_triangular(c, -b)
        cmat = d
        e_abs = np.abs(e)
        tol = np.amax(e_abs) * self._float_min
        if np.count_nonzero(e_abs[:, ndim_in:] > tol) > 0:
            print('WARNING: output depends on input derivatives.  Ignored.')
        dmat = e[:, :ndim_in]

        return StateSpaceContinuous(amat, bmat, cmat, dmat)

    def get_transfer_function(self, in_name, out_name, in_type='v', atol=0.0):
        # type: (str, str, str, float) -> TransferFunctionContinuous
        """Compute the transfer function between the two given nodes.

        Parameters
        ----------
        in_name : str
            the input voltage/current node name.
        out_name : Union[str, List[str]]
            the output voltage node name.
        in_type : str
            set to 'v' for input voltage sources.  Otherwise, current sources.
        atol : float
            absolute tolerance for checking zeros in the numerator.  Used to filter out scipy warnings.

        Returns
        -------
        system : TransferFunctionContinuous
            the scipy transfer function object.  See scipy.signal package on how to use this object.
        """
        state_space = self.get_state_space(in_name, out_name, in_type=in_type)
        num, den = scipy.signal.ss2tf(state_space.A, state_space.B, state_space.C, state_space.D)
        num = num[0, :]
        # check if numerator has leading zeros.
        # this makes it so the user have full control over numerical precision, and
        # avoid scipy bad conditioning warnings.
        while abs(num[0]) <= atol:
            num = num[1:]
        return TransferFunctionContinuous(num, den)

    def get_impedance(self, node_name, freq, atol=0.0):
        # type: (str, float, float) -> complex
        """Computes the impedance looking into the given node.

        Parameters
        ----------
        node_name : str
            the node to compute impedance for.  We will inject a current into this node and measure the voltage
            on this node.
        freq : float
            the frequency to compute the impedance at, in Hertz.
        atol : float
            absolute tolerance for checking zeros in the numerator.  Used to filter out scipy warnings.

        Returns
        -------
        impedance : complex
            the impedance value, in Ohms.
        """
        sys = self.get_transfer_function(node_name, node_name, in_type='i', atol=atol)
        w_test = 2 * np.pi * freq
        _, zin_vec = sys.freqresp(w=[w_test])
        return zin_vec[0]

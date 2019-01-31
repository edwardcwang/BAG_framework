
"""This module tests AnalogBase."""

from abs_templates_ec.analog_core.base import AnalogBase


class SingleRowTest(AnalogBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            mos_type="transistor type, either 'pch' or 'nch'.",
            lch='channel length, in meters.',
            w='transistor width, in meters/number of fins.',
            intent='transistor threshold flavor.',
            fg='number of fingers.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        mos_type = self.params['mos_type']
        lch = self.params['lch']
        w = self.params['w']
        intent = self.params['intent']
        fg = self.params['fg']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']

        w_list = [w]
        th_list = [intent]

        nw_list = pw_list = []
        nth_list = pth_list = []
        if mos_type == 'nch':
            nw_list = w_list
            nth_list = th_list
        else:
            pw_list = w_list
            pth_list = th_list

        self.draw_base(lch, fg, ptap_w, ntap_w, nw_list, nth_list, pw_list, pth_list)

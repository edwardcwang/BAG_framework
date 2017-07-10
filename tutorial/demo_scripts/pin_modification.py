# -*- coding: utf-8 -*-

import os

from bag import BagProject

prj = BagProject()

impl_lib = 'AAATB'

dut_lib = 'demo_testbenches'
dut_cell = 'stimuli_pwl_pinmod'
fbase = os.path.join(os.environ['BAG_FRAMEWORK'], 'tutorial', 'demo_scripts')

fname_list = [os.path.join(fbase, 'a.data'),
              os.path.join(fbase, 'b.data'),
              os.path.join(fbase, 'c.data')]
sig_list = ['a', 'b', 'c']

print('create DUT module')
dsn = prj.create_design_module(dut_lib, dut_cell)
print("design DUT")
dsn.design(fname_list=fname_list, sig_list=sig_list)
print('create DUT schematic')
dsn.implement_design(impl_lib, erase=True)

# -*- coding: utf-8 -*-

import os

from bag import BagProject
from bag.data import load_sim_results

prj = BagProject()

impl_lib = 'AAATB'

dut_lib = 'demo_templates'
dut_cell = 'rc_lowpass'
res = 500
cap_var = 'cload'
cap_swp_list = [100e-15, 500e-15]

tb_lib = 'demo_testbenches'
tb_cell = 'tb_tran_pwl'
fname = os.path.join(os.environ['BAG_FRAMEWORK'], 'tutorial', 'demo_scripts', 'tb_tran_pwl.data')

print('create DUT module')
dsn = prj.create_design_module(dut_lib, dut_cell)
print("design DUT")
dsn.design(res=res, cap=cap_var)
print('create DUT schematic')
dsn.implement_design(impl_lib, erase=True)

print('create TB module')
tb_sch = prj.create_design_module(tb_lib, tb_cell)
print('design TB')
tb_sch.design(fname=fname, dut_lib=impl_lib, dut_cell=dut_cell)
print('create TB schematic')
tb_sch.implement_design(impl_lib, top_cell_name=tb_cell)

print('configure TB state')
tb = prj.configure_testbench(impl_lib, tb_cell)
tb.set_sweep_parameter(cap_var, values=cap_swp_list)
print('update TB state')
tb.update_testbench()
print('done')
tb.run_simulation()

results = load_sim_results(tb.save_dir)

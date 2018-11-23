# -*- coding: utf-8 -*-

"""Generate setup yaml files for various netlist outputs
"""

from typing import Dict, Any, Tuple, List

import os
import argparse

import yaml

from pybag.enum import DesignOutput

mos_default = {
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['B', 'D', 'G', 'S'],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
        'nf': [3, ''],
    },
}

mos_cdl_fmt = """.SUBCKT {cell_name} B D G S
*.PININFO B:B D:B G:B S:B
MM0 D G S B {model_name} {l_str}=l {w_str}=w {nf_str}=1*nf {other}
.ENDS
"""

mos_verilog_fmt = """module {cell_name}(
    inout B,
    inout D,
    inout G,
    inout S
);

endmodule
"""

supported_formats = {
    DesignOutput.CDL: {
        'fname': 'bag_prim.cdl',
        'include': '.INCLUDE {fname}',
        'mos': mos_cdl_fmt,
    },
    DesignOutput.VERILOG: {
        'fname': 'bag_prim.v',
        'include': '`include "{fname}"',
        'mos': mos_verilog_fmt,
    }
}


def populate_header(config: Dict[str, Any], inc_lines: Dict[DesignOutput, List[str]]) -> None:
    for v, lines in inc_lines.items():
        includes = config[v.name]['includes']
        inc_fmt = supported_formats[v]['include']
        for fname in includes:
            lines.append(inc_fmt.format(fname=fname))


def populate_mos(config: Dict[str, Any], netlist_map: Dict[str, Any],
                 inc_lines: Dict[DesignOutput, List[str]]) -> None:
    for cell_name, model_name in config['types']:
        # populate netlist_map
        cur_info = mos_default.copy()
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            out_config = config[v.name]
            l_str = out_config['l_str']
            w_str = out_config['w_str']
            nf_str = out_config['nf_str']
            other = out_config['other']
            mos_fmt = supported_formats[v]['mos']
            lines.append('')
            lines.append(mos_fmt.format(cell_name=cell_name, model_name=model_name,
                                        l_str=l_str, w_str=w_str, nf_str=nf_str, other=other))


def get_info(config: Dict[str, Any], output_dir) -> Tuple[Dict[str, Any], Dict[int, List[str]]]:
    header_config = config['header']
    mos_config = config['mos']

    netlist_map = {}
    inc_lines = {v: [] for v in supported_formats}

    populate_header(header_config, inc_lines)
    populate_mos(mos_config, netlist_map, inc_lines)

    inc_list = {}
    for v, lines in inc_lines.items():
        fname = os.path.abspath(os.path.join(output_dir, supported_formats[v]['fname']))
        inc_list[v.value] = [fname]
        with open(fname, 'w') as f:
            f.writelines(lines)

    return {'BAG_prim': netlist_map}, inc_list


def parse_options() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description='Generate netlist setup file.')
    parser.add_argument('config_fname', type=str,
                        help='YAML file containing technology information.')
    parser.add_argument('output_dir', type=str,
                        help='Output directory.')
    args = parser.parse_args()
    return args.config_fname, args.output_dir


def main() -> None:
    config_fname, output_dir = parse_options()

    os.makedirs(output_dir, exist_ok=True)

    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    netlist_map, inc_list = get_info(config, output_dir)
    result = {
        'inc_list': inc_list,
        'netlist_map': netlist_map,
    }

    with open(os.path.join(output_dir, 'netlist_setup.yaml'), 'w') as f:
        yaml.dump(result, f)


if __name__ == '__main__':
    main()

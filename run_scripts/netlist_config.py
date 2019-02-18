# -*- coding: utf-8 -*-
"""Generate setup yaml files for various netlist outputs

Please run this script through the generate_netlist_config.sh shell script, which will setup
the PYTHONPATH correctly.
"""

from typing import Dict, Any, Tuple, List

import os
import copy
import argparse

from jinja2 import Environment, DictLoader

from pybag.enum import DesignOutput

from bag.io.file import read_yaml, write_yaml

netlist_map_default = {
    'analogLib': {
        'cap': {
            'lib_name': 'analogLib',
            'cell_name': 'cap',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'c': [3, ''],
                'l': [3, ''],
                'm': [3, ''],
                'w': [3, ''],
            }
        },
        'idc': {
            'lib_name': 'analogLib',
            'cell_name': 'idc',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'acm': [3, ''],
                'acp': [3, ''],
                'idc': [3, ''],
                'pacm': [3, ''],
                'pacp': [3, ''],
                'srcType': [3, 'dc'],
                'xfm': [3, ''],
            }
        },
        'ipulse': {
            'lib_name': 'analogLib',
            'cell_name': 'ipulse',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'i1': [3, ''],
                'i2': [3, ''],
                'idc': [3, ''],
                'per': [3, ''],
                'pw': [3, ''],
                'srcType': [3, 'pulse'],
                'td': [3, ''],
            }
        },
        'isin': {
            'lib_name': 'analogLib',
            'cell_name': 'isin',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'freq': [3, ''],
                'ia': [3, ''],
                'idc': [3, ''],
                'srcType': [3, 'sine'],
            }
        },
        'gnd': {
            'lib_name': 'analogLib',
            'cell_name': 'gnd',
            'in_terms': [],
            'io_terms': ['gnd!'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {}
        },
        'res': {
            'lib_name': 'analogLib',
            'cell_name': 'res',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'l': [3, ''],
                'm': [3, ''],
                'r': [3, ''],
                'w': [3, ''],
            }
        },
        'vdc': {
            'lib_name': 'analogLib',
            'cell_name': 'vdc',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'acm': [3, ''],
                'acp': [3, ''],
                'pacm': [3, ''],
                'pacp': [3, ''],
                'srcType': [3, 'dc'],
                'vdc': [3, ''],
                'xfm': [3, ''],
            }
        },
        'vpulse': {
            'lib_name': 'analogLib',
            'cell_name': 'vpulse',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'per': [3, ''],
                'pw': [3, ''],
                'srcType': [3, 'pulse'],
                'td': [3, ''],
                'v1': [3, ''],
                'v2': [3, ''],
                'vdc': [3, ''],
            }
        },
        'vsin': {
            'lib_name': 'analogLib',
            'cell_name': 'vsin',
            'in_terms': [],
            'io_terms': ['PLUS', 'MINUS'],
            'is_prim': True,
            'nets': [],
            'out_terms': [],
            'props': {
                'freq': [3, ''],
                'srcType': [3, 'sine'],
                'va': [3, ''],
                'vdc': [3, ''],
            }
        },
    },
}

mos_default = {
    'lib_name': 'BAG_prim',
    'cell_name': '',
    'in_terms': [],
    'out_terms': [],
    'io_terms': ['B', 'D', 'G', 'S'],
    'nets': [],
    'is_prim': True,
    'props': {
        'l': [3, ''],
        'w': [3, ''],
        'nf': [3, ''],
    },
}

mos_cdl_fmt = """.SUBCKT {{ cell_name }} B D G S
*.PININFO B:B D:B G:B S:B
MM0 D G S B {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
.ENDS
"""

mos_spectre_fmt = """subckt {{ cell_name }} B D G S
MM0 D G S B {{ model_name }}{% for key, val in param_list %} {{ key }}={{ val }}{% endfor %}
ends {{ cell_name }}
"""

mos_verilog_fmt = """module {{ cell_name }}(
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
        'mos': 'mos_cdl',
    },
    DesignOutput.SPECTRE: {
        'fname': 'bag_prim.scs',
        'mos': 'mos_scs',
    },
    DesignOutput.VERILOG: {
        'fname': 'bag_prim.v',
        'mos': 'mos_verilog',
    },
    DesignOutput.SYSVERILOG: {
        'fname': 'bag_prim.sv',
        'mos': '',
    },
}

jinja_env = Environment(
    loader=DictLoader({'mos_cdl': mos_cdl_fmt, 'mos_scs': mos_spectre_fmt, 'mos_verilog': mos_verilog_fmt}),
    keep_trailing_newline=True,
)


def populate_header(config: Dict[str, Any], inc_lines: Dict[DesignOutput, List[str]],
                    inc_list: Dict[int, List[str]]) -> None:
    for v, lines in inc_lines.items():
        inc_list[v.value] = config[v.name]['includes']


def populate_mos(config: Dict[str, Any], netlist_map: Dict[str, Any],
                 inc_lines: Dict[DesignOutput, List[str]]) -> None:
    for cell_name, model_name in config['types']:
        # populate netlist_map
        cur_info = copy.deepcopy(mos_default)
        cur_info['cell_name'] = cell_name
        netlist_map[cell_name] = cur_info

        # write bag_prim netlist
        for v, lines in inc_lines.items():
            param_list = config[v.name]
            template_name = supported_formats[v]['mos']
            if template_name:
                mos_template = jinja_env.get_template(template_name)
                lines.append('\n')
                lines.append(
                    mos_template.render(
                        cell_name=cell_name,
                        model_name=model_name,
                        param_list=param_list,
                        ))


def get_info(config: Dict[str, Any], output_dir) -> Tuple[Dict[str, Any], Dict[int, List[str]], Dict[int, str]]:
    header_config = config['header']
    mos_config = config['mos']

    netlist_map = {}
    inc_lines = {v: [] for v in supported_formats}

    inc_list = {}  # type: Dict[int, List[str]]
    populate_header(header_config, inc_lines, inc_list)
    populate_mos(mos_config, netlist_map, inc_lines)

    prim_files = {}  # type: Dict[int, str]
    for v, lines in inc_lines.items():
        fname = os.path.join(output_dir, supported_formats[v]['fname'])
        if lines:
            prim_files[v.value] = fname
            with open(fname, 'w') as f:
                f.writelines(lines)
        else:
            prim_files[v.value] = ''

    return {'BAG_prim': netlist_map}, inc_list, prim_files


def parse_options() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description='Generate netlist setup file.')
    parser.add_argument(
        'config_fname', type=str, help='YAML file containing technology information.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    args = parser.parse_args()
    return args.config_fname, args.output_dir


def main() -> None:
    config_fname, output_dir = parse_options()

    os.makedirs(output_dir, exist_ok=True)

    config = read_yaml(config_fname)

    netlist_map, inc_list, prim_files = get_info(config, output_dir)
    netlist_map.update(netlist_map_default)
    result = {
        'prim_files': prim_files,
        'inc_list': inc_list,
        'netlist_map': netlist_map,
    }

    write_yaml(os.path.join(output_dir, 'netlist_setup.yaml'), result)


if __name__ == '__main__':
    main()

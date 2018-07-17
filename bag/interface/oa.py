# -*- coding: utf-8 -*-

"""This module implements all CAD database manipulations using OpenAccess plugins.
"""

from typing import TYPE_CHECKING, Sequence, List, Dict, Optional, Any, Tuple

import os
import shutil

import bag.io
from .database import DbAccess

try:
    import pybagoa
except ImportError:
    raise ImportError('Cannot import pybagoa library.  Do you have the right shared library file?')

if TYPE_CHECKING:
    from .zmqwrapper import ZMQDealer
    from ..design.module import ModuleDB


def _handle_reply(reply):
    """Process the given reply."""
    if isinstance(reply, dict):
        if reply.get('type') == 'error':
            if 'data' not in reply:
                raise Exception('Unknown reply format: %s' % reply)
            raise VirtuosoException(reply['data'])
        else:
            try:
                return reply['data']
            except Exception:
                raise Exception('Unknown reply format: %s' % reply)
    else:
        raise Exception('Unknown reply format: %s' % reply)


class VirtuosoException(Exception):
    """Exception raised when Virtuoso returns an error."""

    def __init__(self, *args, **kwargs):
        # noinspection PyArgumentList
        Exception.__init__(self, *args, **kwargs)


class OAInterface(DbAccess):
    """OpenAccess interface between bag and Virtuoso.

    Parameters
    ----------
    dealer : :class:`bag.interface.ZMQDealer`
        the socket used to communicate with :class:`~bag.interface.SkillOceanServer`.
    tmp_dir : str
        temporary file directory for DbAccess.
    db_config : Dict[str, Any]
        the database configuration dictionary.
    """

    def __init__(self, dealer, tmp_dir, db_config):
        # type: (ZMQDealer, str, Dict[str, Any]) -> None
        DbAccess.__init__(self, tmp_dir, db_config)
        self.handler = dealer
        self._rcx_jobs = {}

        if 'lib_def_path' in db_config:
            cds_lib_path = db_config['lib_def_path']
        elif 'CDSLIBPATH' in os.environ:
            cds_lib_path = os.path.abspath(os.path.join(os.environ['CDSLIBPATH'], 'cds.lib'))
        else:
            cds_lib_path = os.path.abspath('./cds.lib')

        self._oa_db = pybagoa.PyOADatabase(cds_lib_path, bag.io.get_encoding())

    def _eval_skill(self, expr, input_files=None, out_file=None):
        # type: (str, Optional[Dict[str, Any]], Optional[str]) -> str
        """Send a request to evaluate the given skill expression.

        Because Virtuoso has a limit on the input/output data (< 4096 bytes),
        if your input is large, you need to write it to a file and have
        Virtuoso open the file to parse it.  Similarly, if you expect a
        large output, you need to make Virtuoso write the result to the
        file, then read it yourself.  The parameters input_files and
        out_file help you achieve this functionality.

        For example, if you need to evaluate "skill_fun(arg fname)", where
        arg is a file containing the list [1 2 3], and fname is the output
        file name, you will call this function with:

        expr = "skill_fun({arg} {fname})"
        input_files = { "arg": [1 2 3] }
        out_file = "fname"

        the bag server will then a temporary file for arg and fname, write
        the list [1 2 3] into the file for arg, call Virtuoso, then read
        the output file fname and return the result.

        Parameters
        ----------
        expr : str
            the skill expression to evaluate.
        input_files : Optional[Dict[str, Any]]
            A dictionary of input files content.
        out_file : Optional[str]
            the output file name argument in expr.

        Returns
        -------
        result : str
            a string representation of the result.

        Raises
        ------
        :class: `.VirtuosoException` :
            if virtuoso encounters errors while evaluating the expression.
        """
        request = dict(
            type='skill',
            expr=expr,
            input_files=input_files,
            out_file=out_file,
        )

        self.handler.send_obj(request)
        reply = self.handler.recv_obj()
        return _handle_reply(reply)

    def close(self):
        # type: () -> None
        self.handler.send_obj(dict(type='exit'))
        self.handler.close()
        self._oa_db.close()
        self._oa_db = None

    def get_cells_in_library(self, lib_name):
        # type: (str) -> List[str]
        return self._oa_db.get_cells_in_library(lib_name)

    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        lib_path = lib_path or self.default_lib_path
        tech_lib = self.db_config['schematic']['tech_lib']
        self._oa_db.create_lib(self, lib_name, lib_path, tech_lib)

    def create_implementation(self, lib_name, template_list, change_list, lib_path=''):
        # type: (str, Sequence[Any], Sequence[Any], str) -> None
        raise NotImplementedError('Not implemented yet.')

    def configure_testbench(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[str, List[str], Dict[str, str], Dict[str, str]]
        raise NotImplementedError('Not implemented yet.')

    def get_testbench_info(self, tb_lib, tb_cell):
        # type: (str, str) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]
        raise NotImplementedError('Not implemented yet.')

    def update_testbench(self,  # type: DbAccess
                         lib,  # type: str
                         cell,  # type: str
                         parameters,  # type: Dict[str, str]
                         sim_envs,  # type: Sequence[str]
                         config_rules,  # type: Sequence[List[str]]
                         env_parameters,  # type: Sequence[List[Tuple[str, str]]]
                         ):
        # type: (...) -> None
        raise NotImplementedError('Not implemented yet.')

    def instantiate_layout_pcell(self, lib_name, cell_name, view_name,
                                 inst_lib, inst_cell, params, pin_mapping):
        # type: (str, str, str, str, str, Dict[str, Any], Dict[str, str]) -> None
        raise NotImplementedError('Not implemented yet.')

    def instantiate_layout(self, lib_name, view_name, via_tech, layout_list):
        # type: (str, str, str, Sequence[Any]) -> None
        raise NotImplementedError('Not implemented yet.')

    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        cmd = 'release_write_locks( "%s" {cell_view_list} )' % lib_name
        in_files = {'cell_view_list': cell_view_list}
        self._eval_skill(cmd, input_files=in_files)

    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        # type: (str, str, str, Optional[str], **kwargs) -> None
        # get netlists to copy
        netlist_dir = os.path.dirname(netlist)
        netlist_files = self.checker.get_rcx_netlists(lib_name, cell_name)
        if not netlist_files:
            # some error checking.  Shouldn't be needed but just in case
            raise ValueError('RCX did not generate any netlists')

        # copy netlists to a "netlist" subfolder in the CAD database
        cell_dir = self.get_cell_directory(lib_name, cell_name)
        targ_dir = os.path.join(cell_dir, 'netlist')
        os.makedirs(targ_dir, exist_ok=True)
        for fname in netlist_files:
            shutil.copy(os.path.join(netlist_dir, fname), targ_dir)

        # create symbolic link as aliases
        symlink = os.path.join(targ_dir, 'netlist')
        try:
            os.remove(symlink)
        except FileNotFoundError:
            pass
        os.symlink(netlist_files[0], symlink)

    def get_cell_directory(self, lib_name, cell_name):
        # type: (str, str) -> str
        """Returns the directory name of the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.

        Returns
        -------
        cell_dir : str
            path to the cell directory.
        """
        return os.path.join(self._oa_db.get_lib_path(lib_name), cell_name)

    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **kwargs) -> None
        # delete old verilog view
        cmd = 'delete_cellview( "%s" "%s" "verilog" )' % (lib_name, cell_name)
        self._eval_skill(cmd)
        cmd = 'schInstallHDL("%s" "%s" "verilog" "%s" t)' % (lib_name, cell_name, verilog_file)
        self._eval_skill(cmd)

    def import_sch_cellview(self, lib_name, cell_name, dsn_db, new_lib_path):
        # type: (str, str, ModuleDB, str) -> None
        # create root directory if missing
        root_path = dsn_db.get_library_path(lib_name)
        if root_path is None:
            root_path = new_lib_path
            dsn_db.append_library(lib_name, new_lib_path)

        # read schematic information
        cell_list = self._oa_db.read_sch_recursive(lib_name, cell_name, "schematic",
                                                   root_path, self.exc_libs)

        # create python templates
        for lib, cell in cell_list:
            python_file = os.path.join(root_path, lib, '%s.py' % cell)
            if not os.path.exists(python_file):
                content = self.get_python_template(lib, cell,
                                                   self.db_config.get('prim_table', {}))
                bag.io.write_file(python_file, content + '\n', mkdir=False)

    def import_design_library(self, lib_name, dsn_db, new_lib_path):
        # type: (str, ModuleDB, str) -> None
        raise NotImplementedError('Not implemented yet.')

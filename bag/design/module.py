# -*- coding: utf-8 -*-

"""This module defines base design module class and primitive design classes.
"""

import os
import abc
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any, Type, Set, Sequence, Callable, \
    Union

from bag import float_to_si_string
from bag.io import get_encoding
from bag.util.cache import DesignMaster, MasterDB

try:
    import pybag
except ImportError:
    raise ImportError('Cannot import pybag library.  Do you have the right shared library file?')

if TYPE_CHECKING:
    from bag.core import BagProject
    from bag.layout.core import TechInfo


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is responsible for keeping track of module libraries and
    creating new modules.

    Parameters
    ----------
    lib_defs : str
        path to the design library definition file.
    tech_info : TechInfo
        the TechInfo instance.
    sch_exc_libs : List[str]
        list of libraries that are excluded from import.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    lib_path : str
        path to create generated library in.
    """

    def __init__(self, lib_defs, tech_info, sch_exc_libs, prj=None, name_prefix='',
                 name_suffix='', lib_path=''):
        # type: (str, TechInfo, List[str], Optional[BagProject], str, str, str) -> None
        MasterDB.__init__(self, '', lib_defs=lib_defs, name_prefix=name_prefix,
                          name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info
        self._exc_libs = set(sch_exc_libs)
        self.lib_path = lib_path

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[Module], str, Dict[str, Any], Set[str], **kwargs) -> Module
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[Module]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs
            optional arguments for the generator.

        Returns
        -------
        master : Module
            the non-finalized generated instance.
        """
        kwargs = kwargs.copy()
        kwargs['lib_name'] = lib_name
        kwargs['params'] = params
        kwargs['used_names'] = used_cell_names
        # noinspection PyTypeChecker
        return gen_cls(self, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False):
        # type: (str, Sequence[Any], bool) -> None
        """Create the masters in the design database.

        Parameters
        ----------
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        debug : bool
            True to print debug messages
        """
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        self._prj.instantiate_schematic(lib_name, content_list, lib_path=self.lib_path)

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def is_lib_excluded(self, lib_name):
        # type: (str) -> bool
        """Returns true if the given schematic library does not contain generators.

        Parameters
        ----------
        lib_name : str
            library name

        Returns
        -------
        is_excluded : bool
            True if given library is excluded.
        """
        return lib_name in self._exc_libs


class Module(DesignMaster, metaclass=abc.ABCMeta):
    """The base class of all schematic generators.  This represents a schematic master.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_fname : str
        the netlist information file name.
    **kwargs :
        additional arguments

    Attributes
    ----------
    parameters : dict[str, any]
        the design parameters dictionary.
    instances : dict[str, None or :class:`~bag.design.Module` or list[:class:`~bag.design.Module`]]
        the instance dictionary.
    """

    # noinspection PyUnusedLocal
    def __init__(self, database, yaml_fname, **kwargs):
        # type: (ModuleDB, str, **kwargs) -> None

        lib_name = kwargs['lib_name']
        params = kwargs['params']
        used_names = kwargs['used_names']
        design_fun = kwargs['design_fun']
        design_args = kwargs['design_args']

        self.tech_info = database.tech_info

        self._yaml_fname = os.path.abspath(yaml_fname)
        self._cv = pybag.PySchCellView(self._yaml_fname, get_encoding())
        # get library/cell name from YAML file name
        dir_name, f_name = os.path.split(self._yaml_fname)
        self._orig_lib_name = self._cv.lib_name
        self._orig_cell_name = self._cv.cell_name

        self._design_fun = design_fun
        self._design_args = design_args
        self.instances = self._cv.get_instances(database)

        self._inputs = self._outputs = self._inouts = None

        # initialize schematic master
        DesignMaster.__init__(self, database, lib_name, params, used_names)

    @property
    def inputs(self):
        # type: () -> Set[str]
        return self._inputs

    @property
    def outputs(self):
        # type: () -> Set[str]
        return self._outputs

    @property
    def inouts(self):
        # type: () -> Set[str]
        return self._inouts

    @abc.abstractmethod
    def design(self, **kwargs):
        """To be overridden by subclasses to design this module.

        To design instances of this module, you can
        call their :meth:`.design` method or any other ways you coded.

        To modify schematic structure, call:

        :meth:`.rename_pin`

        :meth:`.delete_instance`

        :meth:`.replace_instance_master`

        :meth:`.reconnect_instance_terminal`

        :meth:`.array_instance`
        """
        pass

    def set_param(self, key, val):
        # type: (str, Union[int, float, str, bool]) -> None
        self._cv.set_param(key, val)

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # invoke design function
        fun = getattr(self, self._design_fun)
        if self._design_args:
            args = self.params.pop(self._design_args)
            fun(*args, **self.params)
        else:
            fun(**self.params)

        # get set of children master keys
        self.children = {inst.master_key for inst in self.instances.values()
                         if not inst.is_primitive}

        # get pins
        self._inputs = self._cv.get_inputs()
        self._outputs = self._cv.get_outputs()
        self._inouts = self._cv.get_inouts()

        # call super finalize routine
        super(Module, self).finalize()

    @classmethod
    def get_params_info(cls):
        # type: () -> Optional[Dict[str, str]]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return None

    def get_master_basename(self):
        # type: () -> str
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self._orig_cell_name

    def get_content(self, lib_name, rename_fun):
        # type: (str, Callable[[str], str]) -> Optional[Tuple[Any,...]]
        """Returns the content of this master instance.

        Parameters
        ----------
        lib_name : str
            the library to create the design masters in.
        rename_fun : Callable[[str], str]
            a function that renames design masters.

        Returns
        -------
        content : Optional[Tuple[Any,...]]
            the master content data structure.
        """
        if self.is_primitive():
            return None

        return rename_fun(self.cell_name), self._cv

    @property
    def cell_name(self):
        # type: () -> str
        """The master cell name."""
        if self.is_primitive():
            return self.get_cell_name_from_parameters()
        return super(Module, self).cell_name

    @property
    def orig_cell_name(self):
        # type: () -> str
        """The original schematic template cell name."""
        return self._orig_cell_name

    def is_primitive(self):
        # type: () -> bool
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG and schematic primitives.  This method prevents
        the module from being copied during design implementation.  Custom subclasses should
        not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return False

    def should_delete_instance(self):
        # type: () -> bool
        """Returns True if this instance should be deleted based on its parameters.

        This method is mainly used to delete 0 finger or 0 width transistors.  However,
        You can override this method if there exists parameter settings which corresponds
        to an empty schematic.

        Returns
        -------
        delete : bool
            True if parent should delete this instance.
        """
        return False

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : Dict[str, str]
            the schematic parameter dictionary.
        """
        return {}

    def get_cell_name_from_parameters(self):
        """Returns new cell name based on parameters.

        NOTE: This method is only used by BAG primitives.  This method
        enables a BAG primitive to change the cell master based on
        design parameters (e.g. change transistor instance based on the
        intent parameter).  Custom subclasses should not override this
        method.

        Returns
        -------
        cell : str
            the cell name based on parameters.
        """
        return super(Module, self).cell_name

    def rename_pin(self, old_pin, new_pin):
        # type: (str, str) -> None
        """Renames an input/output pin of this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        old_pin : str
            the old pin name.
        new_pin : str
            the new pin name.
        """
        self._cv.rename_pin(old_pin, new_pin)

    def add_pin(self, new_pin, pin_type):
        # type: (str, str) -> None
        """Adds a new pin to this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        new_pin : str
            the new pin name.
        pin_type : str
            the new pin type.  We current support "input", "output", or "inputOutput"
        """
        self._cv.add_pin(new_pin, pin_type)

    def remove_pin(self, remove_pin):
        # type: (str) -> bool
        """Removes a pin from this schematic.

        Parameters
        ----------
        remove_pin : str
            the pin to remove.

        Returns
        -------
        success : bool
            True if the pin is successfully found and removed.
        """
        return self._cv.remove_pin(remove_pin)

    def rename_instance(self, old_name, new_name):
        # type: (str, str) -> None
        """Renames an instance in this schematic.

        Parameters
        ----------
        old_name : str
            the old instance name.
        new_name : str
            the new instance name.
        """
        self._cv.rename_instance(old_name, new_name)
        self.instances[new_name] = self.instances.pop(old_name)

    def remove_instance(self, inst_name):
        # type: (str) -> bool
        """Remove the instance with the given name.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.

        Returns
        -------
        success : bool
            True if the instance is successfully found and removed.
        """
        return self._cv.remove_instance(inst_name)

    def delete_instance(self, inst_name):
        # type: (str) -> bool
        """Delete the instance with the given name.

        This method is identical to remove_instance().  It's here only for backwards
        compatibility.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.

        Returns
        -------
        success : bool
            True if the instance is successfully found and removed.
        """
        return self._cv.remove_instance(inst_name)

    def replace_instance_master(self, inst_name, lib_name, cell_name, static=False):
        # type: (str, str, str, bool) -> None
        """Replace the master of the given instance.

        NOTE: all terminal connections will be reset.  Call reconnect_instance_terminal() to modify
        terminal connections.

        Parameters
        ----------
        inst_name : str
            the child instance to replace.
        lib_name : str
            the new library name.
        cell_name : str
            the new cell name.
        static : bool
            True if we're replacing instance with a static schematic instead of a design module.
        """
        if inst_name not in self.instances:
            raise ValueError('Cannot find instance with name: %s' % inst_name)

        self.instances[inst_name].change_generator(lib_name, cell_name, static=static)

    def reconnect_instance_terminal(self, inst_name, term_name, net_name):
        """Reconnect the instance terminal to a new net.

        Parameters
        ----------
        inst_name : str
            the child instance to modify.
        term_name : Union[str, List[str]]
            the instance terminal name to reconnect.
            If a list is given, it is applied to each arrayed instance.
        net_name : Union[str, List[str]]
            the net to connect the instance terminal to.
            If a list is given, it is applied to each arrayed instance.
        """
        inst = self.instances.get(inst_name, None)
        if inst is None:
            raise ValueError('Cannot find instance {}'.format(inst_name))

        inst.update_connection(term_name, net_name)

    def array_instance(self, inst_name, inst_name_list, term_list=None):
        # type: (str, List[str], Optional[List[Dict[str, str]]]) -> None
        """Replace the given instance by an array of instances.

        This method will replace self.instances[inst_name] by a list of
        Modules.  The user can then design each of those modules.

        Parameters
        ----------
        inst_name : str
            the instance to array.
        inst_name_list : List[str]
            a list of the names for each array item.
        term_list : Optional[List[Dict[str, str]]]
            a list of modified terminal connections for each array item.  The keys are
            instance terminal names, and the values are the net names to connect
            them to.  Only terminal connections different than the parent instance
            should be listed here.
            If None, assume terminal connections are not changed.
        """
        print('WARNING: not supported yet.')
        pass

    def design_dc_bias_sources(self,  # type: Module
                               vbias_dict,  # type: Optional[Dict[str, List[str]]]
                               ibias_dict,  # type: Optional[Dict[str, List[str]]]
                               vinst_name,  # type: str
                               iinst_name,  # type: str
                               define_vdd=True,  # type: bool
                               ):
        # type: (...) -> None
        """Convenience function for generating DC bias sources.

        Given DC voltage/current bias sources information, array the given voltage/current bias
        sources and configure the voltage/current.

        Each bias dictionary is a dictionary from bias source name to a 3-element list.  The first
        two elements are the PLUS/MINUS net names, respectively, and the third element is the DC
        voltage/current value as a string or float. A variable name can be given to define a
        testbench parameter.

        Parameters
        ----------
        vbias_dict : Optional[Dict[str, List[str]]]
            the voltage bias dictionary.  None or empty to disable.
        ibias_dict : Optional[Dict[str, List[str]]]
            the current bias dictionary.  None or empty to disable.
        vinst_name : str
            the DC voltage source instance name.
        iinst_name : str
            the DC current source instance name.
        define_vdd : bool
            True to include a supply voltage source connected to VDD/VSS, with voltage value 'vdd'.
        """
        if define_vdd and 'SUP' not in vbias_dict:
            vbias_dict = vbias_dict.copy()
            vbias_dict['SUP'] = ['VDD', 'VSS', 'vdd']

        for bias_dict, name_template, param_name, inst_name in \
                ((vbias_dict, 'V%s', 'vdc', vinst_name), (ibias_dict, 'I%s', 'idc', iinst_name)):
            if bias_dict:
                name_list, term_list, val_list, param_dict_list = [], [], [], []
                for name in sorted(bias_dict.keys()):
                    value_tuple = bias_dict[name]
                    pname, nname, bias_val = value_tuple[:3]
                    param_dict = value_tuple[3] if len(value_tuple) > 3 \
                        else None  # type: Optional[Dict]
                    term_list.append(dict(PLUS=pname, MINUS=nname))
                    name_list.append(name_template % name)
                    param_dict_list.append(param_dict)
                    if isinstance(bias_val, str):
                        val_list.append(bias_val)
                    elif isinstance(bias_val, int) or isinstance(bias_val, float):
                        val_list.append(float_to_si_string(bias_val))
                    else:
                        raise ValueError('value %s of type %s '
                                         'not supported' % (bias_val, type(bias_val)))

                self.array_instance(inst_name, name_list, term_list=term_list)
                for name, val, param_dict in zip(name_list, val_list, param_dict_list):
                    inst = self.instances[name]
                    inst.set_param(param_name, val)
                    if param_dict is not None:
                        for k, v in param_dict.items():
                            if isinstance(v, str):
                                pass
                            elif isinstance(v, int) or isinstance(v, float):
                                v = float_to_si_string(v)
                            else:
                                raise ValueError('value %s of type %s not supported' % (v, type(v)))

                            inst.set_param(k, v)
            else:
                self.remove_instance(inst_name)

    def design_dummy_transistors(self, dum_info, inst_name, vdd_name, vss_name, net_map=None):
        # type: (List[Tuple[Any]], str, str, str, Optional[Dict[str, str]]) -> None
        """Convenience function for generating dummy transistor schematic.

        Given dummy information (computed by AnalogBase) and a BAG transistor instance,
        this method generates dummy schematics by arraying and modifying the BAG
        transistor instance.

        Parameters
        ----------
        dum_info : List[Tuple[Any]]
            the dummy information data structure.
        inst_name : str
            the BAG transistor instance name.
        vdd_name : str
            VDD net name.  Used for PMOS dummies.
        vss_name : str
            VSS net name.  Used for NMOS dummies.
        net_map : Optional[Dict[str, str]]
            optional net name transformation mapping.
        """
        if not dum_info:
            self.remove_instance(inst_name)
        else:
            num_arr = len(dum_info)
            arr_name_list = ['XDUMMY%d' % idx for idx in range(num_arr)]
            self.array_instance(inst_name, arr_name_list)

            for name, ((mos_type, w, lch, th, s_net, d_net), fg) in zip(arr_name_list, dum_info):
                if mos_type == 'pch':
                    cell_name = 'pmos4_standard'
                    sup_name = vdd_name
                else:
                    cell_name = 'nmos4_standard'
                    sup_name = vss_name
                if net_map is not None:
                    s_net = net_map.get(s_net, s_net)
                    d_net = net_map.get(d_net, d_net)
                s_name = s_net if s_net else sup_name
                d_name = d_net if d_net else sup_name
                inst = self.instances[name]
                inst.change_generator('BAG_prim', cell_name)
                inst.update_connection('G', sup_name)
                inst.update_connection('B', sup_name)
                inst.update_connection('D', d_name)
                inst.update_connection('S', s_name)
                inst.design(w=w, l=lch, nf=fg, intent=th)


class MosModuleBase(Module):
    """The base design class for the bag primitive transistor.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='transistor width, in meters or number of fins.',
            l='transistor length, in meters.',
            nf='transistor number of fingers.',
            intent='transistor threshold flavor.',
        )

    def design(self, w=1e-6, l=60e-9, nf=1, intent='standard'):
        pass

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        w_res = self.tech_info.tech_params['mos']['width_resolution']
        l_res = self.tech_info.tech_params['mos']['length_resolution']
        w = self.params['w']
        l = self.params['l']
        nf = self.params['nf']
        wstr = w if isinstance(w, str) else float_to_si_string(int(round(w / w_res)) * w_res)
        lstr = l if isinstance(l, str) else float_to_si_string(int(round(l / l_res)) * l_res)
        nstr = nf if isinstance(nf, str) else '%d' % nf

        return dict(w=wstr, l=lstr, nf=nstr)

    def get_cell_name_from_parameters(self):
        # type: () -> str
        mos_type = self.orig_cell_name.split('_')[0]
        return '{}_{}'.format(mos_type, self.params['intent'])

    def is_primitive(self):
        # type: () -> bool
        return True

    def should_delete_instance(self):
        # type: () -> bool
        return self.params['nf'] == 0 or self.params['w'] == 0 or self.params['l'] == 0


class ResPhysicalModuleBase(Module):
    """The base design class for a real resistor parametrized by width and length.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            intent='resistor flavor.',
        )

    def design(self, w=1e-6, l=1e-6, intent='standard'):
        pass

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        w = self.params['w']
        l = self.params['l']
        wstr = w if isinstance(w, str) else float_to_si_string(w)
        lstr = l if isinstance(l, str) else float_to_si_string(l)

        return dict(w=wstr, l=lstr)

    def get_cell_name_from_parameters(self):
        # type: () -> str
        return 'res_{}'.format(self.params['intent'])

    def is_primitive(self):
        # type: () -> bool
        return True

    def should_delete_instance(self):
        # type: () -> bool
        return self.params['w'] == 0 or self.params['l'] == 0


class ResMetalModule(Module):
    """The base design class for a metal resistor.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            layer='the metal layer ID.',
        )

    def design(self, w=1e-6, l=1e-6, layer=1):
        """Create a metal resistor.

        Parameters
        ----------
        w : float
            the resistor width, in meters.
        l: float
            the resistor length, in meters.
        layer : int
            the metal layer ID.
        """
        # get technology parameters
        tech_dict = self.tech_info.tech_params['res_metal']
        lib_name = tech_dict['lib_name']
        l_name = tech_dict['l_name']
        w_name = tech_dict['w_name']
        layer_name = tech_dict.get('layer_name', None)
        precision = tech_dict.get('precision', 6)
        cell_name = tech_dict['cell_table'][layer]

        inst = self.instances['R0']

        if layer_name is None:
            # replace resistor cellview
            inst.change_generator(lib_name, cell_name, static=True)
        else:
            inst.set_param(layer_name, cell_name)

        inst.set_param(l_name, float_to_si_string(l, precision=precision))
        inst.set_param(w_name, float_to_si_string(w, precision=precision))
        for key, val in tech_dict['others'].items():
            if isinstance(val, float):
                val = float_to_si_string(val, precision=6)
            elif isinstance(val, int):
                val = '{:d}'.format(val)
            elif isinstance(val, bool) or isinstance(val, str):
                pass
            else:
                raise ValueError('unsupported type: %s' % type(val))

            inst.set_param(key, val)

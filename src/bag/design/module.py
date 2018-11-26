# -*- coding: utf-8 -*-

"""This module defines base design module class and primitive design classes.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any, Union, Iterable

import os
import abc
from itertools import zip_longest

from ..math import float_to_si_string
from ..util.cache import DesignMaster
from .instance import SchInstance

try:
    from pybag.schematic import PySchCellView
    from pybag.enum import TermType, DesignOutput, is_model_type, get_extension
except ImportError:
    raise ImportError('Cannot import pybag library.  Do you have the right shared library file?')

if TYPE_CHECKING:
    from .database import ModuleDB
    from ..layout.core import TechInfo


class Module(DesignMaster, metaclass=abc.ABCMeta):
    """The base class of all schematic generators.  This represents a schematic master.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    yaml_fname : str
        the netlist information file name.
    database : ModuleDB
        the design database object.
    params : Dict[str, Any]
        the parameters dictionary.
    **kwargs : Any
        additional arguments

    Attributes
    ----------
    params : Dict[str, Any]
        the parameters dictionary.
    instances : Dict[str, SchInstance]
        the instance dictionary.
    """

    def __init__(self, yaml_fname, database, params, **kwargs):
        # type: (str, ModuleDB, Dict[str, Any], **Any) -> None
        copy_state = kwargs.get('copy_state', None)

        if copy_state:
            self._cv = copy_state['cv']  # type: PySchCellView
            self._pins = copy_state['pins']  # type: Dict[str, TermType]
            self._model_params = copy_state['model_params']  # type: Optional[Dict[str, Any]]
            self.instances = copy_state['instances']  # type: Dict[str, SchInstance]
        else:
            self._cv = PySchCellView(os.path.abspath(yaml_fname), 'symbol')
            self._pins = {}  # type: Dict[str, TermType]
            self._model_params = None  # type: Optional[Dict[str, Any]]
            self.instances = {name: SchInstance(database, ref)
                              for (name, ref) in
                              self._cv.inst_refs()}  # type: Dict[str, SchInstance]

        # initialize schematic master
        DesignMaster.__init__(self, database, params, copy_state=copy_state)

    def compute_unique_key(self, model_params=None):
        # type: (Optional[Dict[str, Any]]) -> Any
        if model_params is None:
            model_params = self._model_params
        return self.get_qualified_name(), self.params, model_params

    def get_master_basename(self):
        # type: () -> str
        return self.orig_cell_name

    def get_copy_state(self):
        # type: () -> Dict[str, Any]
        base = DesignMaster.get_copy_state(self)
        new_cv = self._cv.get_copy()
        new_inst = {name: SchInstance(self.master_db, ref, master=self.instances[name].master)
                    for name, ref in new_cv.inst_refs()}
        base['cv'] = new_cv
        base['pins'] = self._pins.copy()
        base['model_params'] = self._model_params
        base['instances'] = new_inst
        return base

    def get_copy(self):
        # type: () -> Module
        """Returns a copy of this master instance."""
        copy_state = self.get_copy_state()
        return self.__class__('', self._master_db, {}, copy_state=copy_state)

    @property
    def tech_info(self):
        # type: () -> TechInfo
        return self.master_db.tech_info

    @property
    def pins(self):
        # type: () -> Dict[str, TermType]
        return self._pins

    @abc.abstractmethod
    def design(self, **kwargs):
        # type: (**Any) -> None
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

    def design_model(self, model_params):
        # type: (Dict[str, Any]) -> None
        self._model_params = model_params

        if 'view_name' not in model_params:
            # this is a hierarchical model
            for name, inst in self.instances.items():
                cur_params = self._model_params.get(name, None)
                if cur_params is None:
                    raise ValueError('Cannot find model parameters for instance {}'.format(name))
                # TODO: finish
                pass

    def set_param(self, key, val):
        # type: (str, Union[int, float, bool, str]) -> None
        """Set schematic parameters for this master.

        This method is only used to set parameters for BAG primitives.

        Parameters
        ----------
        key : str
            parameter name.
        val : Union[int, float, bool, str]
            parameter value.
        """
        self._cv.set_param(key, val)

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # invoke design function
        self.design(**self.params)

        # get set of children master keys
        for inst in self.instances.values():
            if not inst.is_primitive:
                self.add_child_key(inst.master_key)

        # get pins
        self._pins = dict(self._cv.terminals())

        # update cell name
        old_cell_name = self._cv.cell_name
        new_cell_name = self.cell_name
        if old_cell_name != new_cell_name:
            self._cv.cell_name = new_cell_name

        # call super finalize routine
        DesignMaster.finalize(self)

    def get_content(self, rename_dict, name_prefix, name_suffix):
        # type: (Dict[str, str], str, str) -> Tuple[str, Any]
        if self.is_primitive():
            return '', None

        cell_name = self.format_cell_name(self.cell_name, rename_dict, name_prefix, name_suffix)
        return cell_name, self._cv

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
        return self._cv.cell_name

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
        # type: () -> str
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
        # type: (str, Union[TermType, str]) -> None
        """Adds a new pin to this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        new_pin : str
            the new pin name.
        pin_type : Union[TermType, str]
            the new pin type.
        """
        if isinstance(pin_type, TermType):
            self._cv.add_pin(new_pin, pin_type)
        else:
            self._cv.add_pin(new_pin, TermType[pin_type].value)

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

    def rename_instance(self, old_name, new_name, conn_list=None):
        # type: (str, str, Optional[Iterable[Tuple[str, str]]]) -> None
        """Renames an instance in this schematic.

        Parameters
        ----------
        old_name : str
            the old instance name.
        new_name : str
            the new instance name.
        conn_list : Optional[Iterable[Tuple[str, str]]]
            an optional connection list.
        """
        self._cv.rename_instance(old_name, new_name)
        self.instances[new_name] = inst = self.instances.pop(old_name)
        if conn_list:
            for term, net in conn_list:
                inst.update_connection(new_name, term, net)

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
        success = self._cv.remove_instance(inst_name)
        if success:
            del self.instances[inst_name]
        return success

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
        # type: (str, str, str) -> None
        """Reconnect the instance terminal to a new net.

        Parameters
        ----------
        inst_name : str
            the instance to modify.
        term_name : str
            the instance terminal name to reconnect.
        net_name : str
            the net to connect the instance terminal to.
        """
        inst = self.instances.get(inst_name, None)
        if inst is None:
            raise ValueError('Cannot find instance {}'.format(inst_name))

        inst.update_connection(inst_name, term_name, net_name)

    def reconnect_instance(self, inst_name, term_net_iter):
        # type: (str, Iterable[Tuple[str, str]]) -> None
        """Reconnect all give instance terminals

        Parameters
        ----------
        inst_name : str
            the instance to modify.
        term_net_iter : Iterable[Tuple[str, str]]
            an iterable of (term, net) tuples.
        """
        inst = self.instances.get(inst_name, None)
        if inst is None:
            raise ValueError('Cannot find instance {}'.format(inst_name))

        for term, net in term_net_iter:
            inst.update_connection(inst_name, term, net)

    def array_instance(self,
                       inst_name: str,
                       inst_name_list: Optional[List[str]] = None,
                       term_list: Optional[List[Dict[str, str]]] = None,
                       inst_term_list: Optional[List[Tuple[str, Iterable[Tuple[str, str]]]]] = None,
                       dx: int = 0,
                       dy: int = 0
                       ) -> None:
        """Replace the given instance by an array of instances.

        This method will replace self.instances[inst_name] by a list of
        Modules.  The user can then design each of those modules.

        Parameters
        ----------
        inst_name : str
            the instance to array.
        inst_name_list : Optional[List[str]]
            a list of the names for each array item.
        term_list : Optional[List[Dict[str, str]]]
            a list of modified terminal connections for each array item.  The keys are
            instance terminal names, and the values are the net names to connect
            them to.  Only terminal connections different than the parent instance
            should be listed here.
            If None, assume terminal connections are not changed.
        inst_term_list : Optional[List[Tuple[str, List[Tuple[str, str]]]]]
            zipped version of inst_name_list and term_list.  If given, this is used instead.
        dx : int
            the X coordinate shift.  If dx = dy = 0, default to shift right.
        dy : int
            the Y coordinate shift.  If dx = dy = 0, default to shift right.
        """
        if inst_term_list is None:
            if inst_name_list is None:
                raise ValueError('inst_name_list cannot be None if inst_term_iter is None.')
            # get instance/terminal list iterator
            if term_list is None:
                inst_term_list = zip_longest(inst_name_list, [], fillvalue=[])
            elif len(inst_name_list) != len(term_list):
                raise ValueError('inst_name_list and term_list length mismatch.')
            else:
                inst_term_list = zip_longest(inst_name_list, (term.items() for term in term_list))
        else:
            inst_name_list = [arg[0] for arg in inst_term_list]
        # array instance
        self._cv.array_instance(inst_name, dx, dy, inst_term_list)

        # update instance dictionary
        orig_inst = self.instances.pop(inst_name)
        db = orig_inst.database
        for name in inst_name_list:
            inst_ptr = self._cv.get_inst_ref(name)
            self.instances[name] = SchInstance(db, inst_ptr, master=orig_inst.master)

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
                inst.update_connection(name, 'G', sup_name)
                inst.update_connection(name, 'B', sup_name)
                inst.update_connection(name, 'D', d_name)
                inst.update_connection(name, 'S', s_name)
                inst.design(w=w, l=lch, nf=fg, intent=th)

    def design_transistor(self,  # type: Module
                          inst_name,  # type: str
                          w,  # type: Union[float, int]
                          lch,  # type: float
                          seg,  # type: int
                          intent,  # type: str
                          m,  # type: str
                          d='',  # type: str
                          g='',  # type: Union[str, List[str]]
                          s='',  # type: str
                          b='',  # type: str
                          stack=1  # type: int
                          ):
        # type: (...) -> None
        """Design a BAG_prim transistor (with stacking support).

        This is a convenient method to design a stack transistor.  Additional transistors
        will be created on the right.  The intermediate nodes of each parallel segment are not
        shorted together.

        Parameters
        ----------
        inst_name : str
            name of the BAG_prim transistor instance.
        w : Union[float, int]
            the width of the transistor, in meters/number of fins.
        lch : float
            the channel length, in meters.
        seg : int
            number of parallel segments of stacked transistors.
        intent : str
            the threshold flavor.
        m : str
            base name of the intermediate nodes.  the intermediate nodes will be named
            'midX', where X is an non-negative integer.
        d : str
            the drain name.  Empty string to not rename.
        g : Union[str, List[str]]
            the gate name.  Empty string to not rename.
            If a list is given, then a NAND-gate structure will be built where the gate nets
            may be different.  Index 0 corresponds to the gate of the source transistor.
        s : str
            the source name.  Empty string to not rename.
        b : str
            the body name.  Empty string to not rename.
        stack : int
            number of series stack transistors.
        """
        inst = self.instances[inst_name]
        if not issubclass(inst.master_class, MosModuleBase):
            raise ValueError('This method only works on BAG_prim transistors.')
        if stack <= 0 or seg <= 0:
            raise ValueError('stack and seg must be positive')

        g_is_str = isinstance(g, str)
        if stack == 1:
            # design instance
            inst.design(w=w, l=lch, nf=seg, intent=intent)
            # connect terminals
            if not g_is_str:
                g = g[0]
            for term, net in (('D', d), ('G', g), ('S', s), ('B', b)):
                if net:
                    inst.update_connection(inst_name, term, net)
        else:
            if not m:
                raise ValueError('Intermediate node base name cannot be empty.')
            # design instance
            inst.design(w=w, l=lch, nf=1, intent=intent)
            # rename G/B
            if g_is_str and g:
                inst.update_connection(inst_name, 'G', g)
            if b:
                inst.update_connection(inst_name, 'B', b)
            if not d:
                d = inst.get_connection('D')
            if not s:
                s = inst.get_connection('S')

            if seg == 1:
                # only one segment, array instance via naming
                # rename instance
                new_name = inst_name + '<0:{}>'.format(stack - 1)
                self.rename_instance(inst_name, new_name)
                # rename D/S
                if stack > 2:
                    m += '<0:{}>'.format(stack - 2)
                new_s = s + ',' + m
                new_d = m + ',' + d
                inst.update_connection(new_name, 'D', new_d)
                inst.update_connection(new_name, 'S', new_s)
                if not g_is_str:
                    inst.update_connection(new_name, 'G', ','.join(g))
            else:
                # multiple segment and stacks, have to array instance
                # construct instance name/terminal map iterator
                inst_term_list = []
                last_cnt = (stack - 1) * seg
                g_cnt = 0
                for cnt in range(0, last_cnt + 1, seg):
                    d_suf = '<{}:{}>'.format(cnt + seg - 1, cnt)
                    s_suf = '<{}:{}>'.format(cnt - 1, cnt - seg)
                    iname = inst_name + d_suf
                    if cnt == 0:
                        s_name = s
                        d_name = m + d_suf
                    elif cnt == last_cnt:
                        s_name = m + s_suf
                        d_name = d
                    else:
                        s_name = m + s_suf
                        d_name = m + d_suf
                    term_list = [('S', s_name), ('D', d_name)]
                    if not g_is_str:
                        term_list.append(('G', g[g_cnt]))
                        g_cnt += 1
                    inst_term_list.append((iname, term_list))

                self.array_instance(inst_name, inst_term_list=inst_term_list)


class MosModuleBase(Module):
    """The base design class for the bag primitive transistor.
    """

    def __init__(self, yaml_fname, database, params, **kwargs):
        # type: (str, ModuleDB, Dict[str, Any], **Any) -> None
        Module.__init__(self, yaml_fname, database, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='transistor width, in meters or number of fins.',
            l='transistor length, in meters.',
            nf='transistor number of fingers.',
            intent='transistor threshold flavor.',
        )

    def design(self, w, l, nf, intent):
        # type: (Union[float, int], float, int, str) -> None
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
    """

    def __init__(self, yaml_fname, database, params, **kwargs):
        # type: (str, ModuleDB, Dict[str, Any], **Any) -> None
        Module.__init__(self, yaml_fname, database, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            intent='resistor flavor.',
        )

    def design(self, w, l, intent):
        # type: (float, float, str) -> None
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
    """

    def __init__(self, yaml_fname, database, params, **kwargs):
        # type: (str, ModuleDB, Dict[str, Any], **Any) -> None
        Module.__init__(self, yaml_fname, database, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            layer='the metal layer ID.',
        )

    def design(self, w, l, layer):
        # type: (float, float, int) -> None
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

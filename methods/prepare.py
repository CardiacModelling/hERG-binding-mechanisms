#
# Prepare myokit model
#
import os
import myokit

from . import DIR_METHOD

_model_dir = os.path.join(DIR_METHOD, '..', 'models')

_model_files = {
    'dutta': 'dutta-2017.mmt',
}

_model_voltage = {
    'dutta': 'membrane.V',
}

_model_i_ion = {
    'dutta': 'membrane.i_ion',
}

_default_stim_amp = -80  # in [A/F]
_model_stim_amp = {
    'dutta': ('stimulus.amplitude', _default_stim_amp),
}

# stim_dur, stim_offset, cl, stim_amp
_default_stim_setup = (0.5, 50, 1000, 1)
_model_stim_setup = {
    'dutta': _default_stim_setup,
}

_model_i_net = { # The order matters: INaL, ICaL, IKr, IKs, IK1, Ito
    #'dutta': ('inal.INaL', 'ical.ICaL_total', 'ikr.IKr', 'iks.IKs', 'ik1.IK1',
    #          'ito.Ito'),
    'dutta': ('inal.INaL', 'ical.ICaL', 'ikr.IKr', 'iks.IKs', 'ik1.IK1',
              'ito.Ito'),
}


def mmt(model):
    """
    Takes a short name (e.g. "chang") and returns the path to its .mmt file.
    """
    return os.path.join(_model_dir, _model_files[model])


def change_component_name(x):
    # Replace the component name with 'ikr_bind'
    return 'ikr_bind.' + x.split('.')[-1]


def prepare(model, binding_model):
    """
    Prepares a :class:`myokit.Model` by replacing a current model in `model`
    with `binding_model`.

    Inputs
    ======
        model: :class:`myokit.Model` the base model.
        binding_model: :class:`myokit.Model` the model to replace a current
                       model in `model`.
    """

    # Check model
    model.validate()
    binding_model.validate()

    # Check major variables exist and have the right units
    model.timex()
    #model.convert_units(time, _time_units)
    vm = model.labelx('membrane_potential')
    #model.convert_units(vm, _voltage_units)
    ek = model.labelx('EK')
    ko = model.labelx('K_o')

    var_map = {
        'membrane.V': vm,
        'nernst.EK': ek,
        binding_model.labelx('K_o'): ko,
    }
    #try:
    #    binding_ikr = binding_model.get('ikr')
    #except KeyError:
    #    binding_ikr = binding_model.get('IKr')
    binding_ikr = binding_model.get('ikr')
    model.import_component(binding_ikr, new_name='ikr_bind', var_map=var_map)

    old_ikr = model.labelx('ikr')
    new_ikr = model.get('ikr_bind.IKr')
    subst = {myokit.Name(old_ikr): myokit.Name(new_ikr)}
    refs_by = list(old_ikr.refs_by())
    for var in refs_by:
        v = var.rhs().clone(subst=subst)
        var.set_rhs(v)
    model.remove_component(old_ikr.parent(kind=myokit.Component))

    # Validate final model
    model.validate()
    #print(model.code())

    return model

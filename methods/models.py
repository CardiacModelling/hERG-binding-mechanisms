import os
import numpy as np
import pints
import myokit
import myokit.lib.markov as markov

###############################################################################
## Defining Model
###############################################################################

from . import DIR_METHOD
from . import concentrations
from . import t_hold, v_hold
from .parameters import _model_current, _model_dose, _model_conductance
from .parameters import _model_kinetics, _model_binding
from .parameters import _model_non_hERG_conductance
from . import prepare

#_time_units = 'ms'
#_voltage_units = 'mV'
#_current_units = 'A/F'
#_concentration_units = 'mM'
#_capacitance_units = 'pF'
#_compound_concentration_units = 'nM'

# Model files
_model_dir = os.path.join(DIR_METHOD, '..', 'models')
_model_files = {
    'm0a': 'lei-2019-m0a.mmt',
    'm0b': 'lei-2019-m0b.mmt',
    'm1': 'lei-2019-m1.mmt',
    'm2': 'lei-2019-m2.mmt',
    'm2i': 'lei-2019-m2i.mmt',
    'm3': 'lei-2019-m3.mmt',
    'm4': 'lei-2019-m4.mmt',
    'm5': 'lei-2019-m5.mmt',
    'm5i': 'lei-2019-m5i.mmt',
    'm6': 'lei-2019-m6.mmt',
    'm7': 'lei-2019-m7.mmt',
    'm8': 'lei-2019-m8.mmt',
    'm9': 'lei-2019-m9.mmt',
    'm10': 'lei-2019-m10.mmt',
    'm11': 'lei-2019-m11.mmt',
    'm12': 'lei-2019-m12.mmt',  # Li et al. 2017 binding model
    'm13': 'lei-2019-m13.mmt',
    'li': 'li-2017.mmt',
    'li-m0a': 'li-2017-m0a.mmt',
    'li-m0b': 'li-2017-m0b.mmt',
    'li-m1': 'li-2017-m1.mmt',
    'li-m2': 'li-2017-m2.mmt',
    'li-m2i': 'li-2017-m2i.mmt',
    'li-m3': 'li-2017-m3.mmt',
    'li-m4': 'li-2017-m4.mmt',
    'li-m5': 'li-2017-m5.mmt',
    'li-m5i': 'li-2017-m5i.mmt',
    'li-m6': 'li-2017-m6.mmt',
    'li-m7': 'li-2017-m7.mmt',
    'li-m8': 'li-2017-m8.mmt',
    'li-m9': 'li-2017-m9.mmt',
    'li-m10': 'li-2017-m10.mmt',
    'li-m11': 'li-2017-m11.mmt',
    'li-m12': 'li-2017-m12.mmt',  # Should be the same as 'li-2017.mmt'
    'li-m13': 'li-2017-m13.mmt',
}


def mmt(model):
    """
    Takes a short name (e.g. "m1") and returns the path to its .mmt file.
    """
    return os.path.join(_model_dir, _model_files[model])


#
# Create ForwardModel
#
class Model(pints.ForwardModel):

    def __init__(self, model_name, protocol_def, useFilterCap=False,
                 parameters=['conductance', 'binding'], analytical=False):
        # model_name: model name that matches those in _model_files.
        # protocol_def: func take model as arg and return tuple
        #               (modified model, step)
        #               or
        #               str to a file name contains protocol time series.
        # useFilterCap: apply capacitive filtering if True.
        # parameters: a list that sets parameters of the model, choices:
        #             ['conductance', 'kinetics', 'binding'].
        # analytical: if True, use analytical solver, only works for step
        #             functions.

        # Load model
        model = myokit.load_model(mmt(model_name))
        self._model_name = model_name
        self._current = _model_current[model_name]

        # Parameters
        model_parameters = []
        self._conductance = _model_conductance[model_name]
        self._kinetics = _model_kinetics[model_name]
        self._binding = _model_binding[model_name]
        if 'conductance' in parameters:
            model_parameters += self._conductance
        if 'kinetics' in parameters:
            model_parameters += self._kinetics
        if 'binding' in parameters:
            model_parameters += self._binding
        self.set_parameters(model_parameters)

        # Dose
        #self._dose_idx = model.get(_model_dose[model_name]).indice()
        self._model_dose = _model_dose[model_name]

        # 0. Clamp concentrations
        for var, value in concentrations.items():
            var = model.labelx(var)
            #model.convert_units(var, _concentration_units)
            var.clamp(value)

        self.useFilterCap = useFilterCap
        self.analytical = analytical
        if not self.analytical:
            # 1. Create holding protocol
            protocol = myokit.pacing.constant(v_hold)
            self.simulation1 = myokit.Simulation(model, protocol)

            # 2. Create specified protocol
            if type(protocol_def) is str:
                # NOTE: Do not use the mmt version here!
                # mmt version contains the holding step too.
                d = myokit.DataLog.load_csv(protocol_def).npview()
                self.simulation2 = myokit.Simulation(model)
                self.simulation2.set_fixed_form_protocol(
                    d['time'] * 1e3,  # s -> ms
                    d['voltage']  # mV
                )
                if self.useFilterCap:
                    raise ValueError('Cannot use capacitance filtering with'
                                     + ' the given format of protocol_def')
            elif type(protocol_def) is dict:
                self.simulation2 = myokit.Simulation(model)
                self.simulation2.set_fixed_form_protocol(
                    protocol_def['time'],  # ms
                    protocol_def['voltage']  # mV
                )
            else:
                if self.useFilterCap:
                    model, steps, fcap = protocol_def(model, self.useFilterCap)
                    self.fcap = fcap
                else:
                    model, steps = protocol_def(model)
                protocol = myokit.Protocol()
                for f, t in steps:
                    protocol.add_step(f, t)

                self.simulation2 = myokit.Simulation(model, protocol)

            self.simulation2.set_tolerance(1e-8, 1e-10)
            self.simulation2.set_max_step_size(1e-2)
        else:
            if type(protocol_def) is str:
                self.p = myokit.load_protocol(protocol_def)
            else:
                self.p = protocol_def
            if not isinstance(self.p, myokit.Protocol):
                raise ValueError(
                    'Analytical simulation can only be used with'
                    + ' myokit.Protocol')
            model.get('membrane.V').set_label('membrane_potential')
            self.m = markov.LinearModel.from_component(
                #model.get('ikr'),
                model.get(self._current.split('.')[0]),
                parameters=self.PARAM + [self._model_dose],
                current=self._current,
            )

            # 1. Create holding protocol
            self.p0 = myokit.pacing.constant(v_hold)
            self.simulation1 = markov.AnalyticalSimulation(self.m, self.p0)
            # 2. Create specified protocol
            self.simulation2 = markov.AnalyticalSimulation(self.m, self.p)

        self.init_state = self.simulation1.state()
        self.set_fix_parameters({})

    def n_parameters(self):
        # n_parameters() method for PINTS
        return len(self.PARAM)

    def simulate(self, parameters, times, extra_log=[], reset=True):
        # simulate() method for PINTS

        if self.analytical:
            # Parameters can only set once; need to recreate simulation obj
            self.simulation1 = markov.AnalyticalSimulation(self.m, self.p0)
            self.simulation2 = markov.AnalyticalSimulation(self.m, self.p)

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Update model parameters
        for i, name in enumerate(self.PARAM):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        if not self.analytical:
            self.simulation1.set_time(0)
            self.simulation2.set_time(0)
        if reset:
            self.simulation1.reset()
            self.simulation2.reset()
            self.simulation1.set_state(self.init_state)
            self.simulation2.set_state(self.init_state)
            self._update_dose()  # set after state in case D is state
            self.pre_state = None
        else:
            self.simulation1.reset()
            self.simulation2.reset()
            self._update_dose()  # set before state in case pre_state has D
            self.simulation1.set_state(self.pre_state)
            self.simulation2.set_state(self.pre_state)

        # Run!
        try:
            self.simulation1.pre(t_hold)
            self.simulation2.set_state(self.simulation1.state())
            if not self.analytical:
                d = self.simulation2.run(np.max(times) + 0.02,
                    log_times = times,
                    log = [
                           self._current,
                          ] + extra_log,
                    ).npview()
                #self.simulation1.set_state(self.simulation2.state())
            else:
                d = self.simulation2.run(np.max(times) + 0.02,
                    log_times = times,
                    ).npview()
            self.pre_state = self.simulation2.state()
        except (myokit.SimulationError, ValueError):
            self.pre_state = np.copy(self.init_state)
            return np.ones(times.shape) * float('inf')

        # Apply capacitance filter and return
        if self.useFilterCap:
            d[self._current] = d[self._current] * self.fcap(times)

        if len(extra_log) > 0:
            return d
        return d[self._current]

    def voltage(self, times):
        # Return voltage protocol

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Run
        #self.simulation1.reset()
        #self.simulation2.reset()
        #self.simulation1.set_state(self.init_state)
        #self.simulation2.set_state(self.init_state)
        self.simulation1.set_time(0)
        self.simulation2.set_time(0)
        s1 = self.simulation1.state()
        s2 = self.simulation2.state()
        try:
            self.simulation1.pre(t_hold)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times) + 0.02, 
                log_times = times, 
                log = ['membrane.V'],
                ).npview()
        except myokit.SimulationError:
            self.simulation1.set_state(s1)
            self.simulation2.set_state(s2)
            return float('inf')
        # Return
        self.simulation1.set_time(0)
        self.simulation2.set_time(0)
        self.simulation1.set_state(s1)
        self.simulation2.set_state(s2)
        return d['membrane.V']

    def cap_filter(self, times):
        if self.useFilterCap:
            return self.fcap(times)
        else:
            return None

    def set_parameters(self, p):
        # return the name of the parameters
        self.PARAM = p

    def fix_hill(self):
        # If True, exclude the Hill coefficient from the fitting parameters.
        hills = ['.hill']  #, '.n']
        x = []
        for i, p in enumerate(self.PARAM):
            for h in hills:
                if h in p:
                    x.append(i)
        if len(x) == 1:
            print(f'Old parameters: {self.PARAM}')
            self.PARAM.pop(x[0])
            print(f'New parameters: {self.PARAM}')
        else:
            raise ValueError(f'Ambigous parameter name to find the Hill'\
                    ' coefficient {self.PARAM}')

    def fix_kt(self):
        # If True, exclude the parameter kt from the fitting parameters.
        kts = ['.kt']
        x = []
        for i, p in enumerate(self.PARAM):
            for k in kts:
                if k in p:
                    x.append(i)
        if len(x) == 1:
            print(f'Old parameters: {self.PARAM}')
            self.PARAM.pop(x[0])
            print(f'New parameters: {self.PARAM}')
        else:
            raise ValueError(f'Ambigous parameter name to find the parameter'\
                    ' kt {self.PARAM}')

    def parameters(self):
        # return the name of the parameters
        return self.PARAM

    def set_temperature(self, value):
        # Set simulation temperature (K)
        try:
            self._set_fix_parameters({'physical_constants.T': value})
        except KeyError:
            self._set_fix_parameters({'nernst.T': value})

    def set_dose(self, value):
        # Set dose concentration (nM)
        self._dose = float(value)
        self._update_dose()

    def _update_dose(self):
        # Update dose concentration (nM) internally
        try:
            assert(self._dose >= 0)
        except NameError:
            raise ValueError('Please set dose value with method `set_dose()`')
        '''
        s1 = self.simulation1.state()
        s1[self._dose_idx] = self._dose
        self.simulation1.set_state(s1)

        s2 = self.simulation2.state()
        s2[self._dose_idx] = self._dose
        self.simulation2.set_state(s2)
        '''
        self._set_fix_parameters({self._model_dose: self._dose})
        #'''

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.simulation1.set_constant(p, parameters[p])
            self.simulation2.set_constant(p, parameters[p])

    def name(self):
        # name
        return self._model_name


#
# AP model
#
class APModel(pints.ForwardModel):

    def __init__(self, ap_model_name, model_name, cl=None, n_prepace=1000,
                 parameters=['conductance', 'binding']):
        # ap_model_name: model name that matches those in prepare._model_files.
        # model_name: model name that matches those in _model_files.
        # cl: pacing cycle length (in ms).
        # n_prepace: number of pre-pacing (default 1000).
        # parameters: a list that sets parameters of the model, choices:
        #             ['conductance', 'kinetics', 'binding'].
        #
        # NOTE: 'parameters' refer to only IKr/hERG related parameters, for
        #       other ionic currents, the Hill equation is used to model the
        #       compound action, and the parameters can be set using
        #       `set_non_hERG_parameters` methods.

        # Load model
        ap_model = myokit.load_model(prepare.mmt(ap_model_name))
        binding_model = myokit.load_model(mmt(model_name))
        self._ap_model_name = ap_model_name
        self._model_name = model_name
        c = _model_current[model_name]
        self._current = prepare.change_component_name(c)
        self._i_ion = prepare._model_i_ion[ap_model_name]
        self._i_net = list(prepare._model_i_net[ap_model_name])
        self._i_net[2] = prepare.change_component_name(self._i_net[2])
        self._voltage = prepare._model_voltage[ap_model_name]
        self._non_hERG_gs = _model_non_hERG_conductance[ap_model_name]

        # Parameters
        model_parameters = []
        self._conductance = [prepare.change_component_name(i)
                for i in _model_conductance[model_name]]
        self._kinetics = [prepare.change_component_name(i)
                for i in _model_kinetics[model_name]]
        self._binding = [prepare.change_component_name(i)
                for i in _model_binding[model_name]]
        if 'conductance' in parameters:
            model_parameters += self._conductance
        if 'kinetics' in parameters:
            model_parameters += self._kinetics
        if 'binding' in parameters:
            model_parameters += self._binding
        self.set_parameters(model_parameters)

        # Prepare model
        model = prepare.prepare(ap_model, binding_model)

        # Dose
        dose_var = prepare.change_component_name(_model_dose[model_name])
        #self._dose_idx = model.get(dose_var).indice()
        self._model_dose = dose_var
        self._dose = 0.  # Set dose once

        # Get original (non IKr current) conductance values
        self._non_hERG_gs_values = []
        for v in self._non_hERG_gs:
            self._non_hERG_gs_values.append(float(model.get(v).rhs()))
            #print(v, float(model.get(v).rhs()))

        # 0. Prepare simulations
        stim_amp_var, stim_amp_val = prepare._model_stim_amp[ap_model_name]
        stim_dur, stim_offset, _cl, stim_amp = \
                prepare._model_stim_setup[ap_model_name]
        if cl is None:
            cl = _cl
        self._cl = cl
        self._n_prepace = n_prepace

        # 1. Create prepace stimulus
        self.simulation1 = myokit.Simulation(model)
        self.simulation1.set_max_step_size(1e-1)  # ms
        self.simulation1.set_constant(stim_amp_var, stim_amp_val)
        preprotocol = myokit.pacing.blocktrain(period=cl,
                                               duration=stim_dur,
                                               offset=stim_offset,
                                               level=stim_amp)
        self.simulation1.set_protocol(preprotocol)
        del(preprotocol)

        # 2. Create specified protocol
        self.simulation2 = myokit.Simulation(model)
        self.simulation2.set_max_step_size(1e-1)  # ms
        self.simulation2.set_constant(stim_amp_var, stim_amp_val)
        protocol = myokit.pacing.blocktrain(period=cl,
                                            duration=stim_dur,
                                            offset=stim_offset,
                                            level=stim_amp)
        self.simulation2.set_protocol(protocol)
        del(protocol)

        self.init_state = self.simulation1.state()
        self.set_fix_parameters({})

    def update_init_state_as_steady_state(self, n_paces=None):
        # Update self.init_state to the steady state with the current model
        # setting.

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Reset to ensure each simulate has same init condition
        self.simulation1.set_time(0)
        self.simulation1.reset()
        self.simulation1.set_state(self.init_state)
        #self._update_dose()

        if n_paces is None:
            n_paces = self._n_prepace

        # Run!
        self.simulation1.pre(n_paces * self._cl)
        self.init_state = self.simulation1.state()

    def n_parameters(self):
        # n_parameters() method for PINTS
        return len(self.PARAM)

    def set_non_hERG_parameters(self, ic50s, hs):
        # Setting the compound parameters (IC50s and hill coefficients) for
        # currents other than hERG.
        self._non_hERG_ic50s = np.asarray(ic50s)
        self._non_hERG_hs = np.asarray(hs)
        self._update_non_hERG_parameters()

    def _hill(self, dose, ic50, h):
        # Hill equation for fraction of block
        if ic50 is None:
            return 0
        return (dose**h) / (ic50**h + dose**h)

    def _update_non_hERG_parameters(self):
        # Update the current other than hERG with `self._dose`.
        try:
            assert(len(self._non_hERG_ic50s) == len(self._non_hERG_gs))
            assert(len(self._non_hERG_hs) == len(self._non_hERG_gs))
        except (AttributeError, NameError):
            raise ValueError('Please set compound parameters for non-hERG ' +\
                    'currents with method `set_non_hERG_parameters()`')
        for v, g, ic50, h in zip(self._non_hERG_gs, self._non_hERG_gs_values,
                                 self._non_hERG_ic50s, self._non_hERG_hs):
            # Compute scaling using Hill equation
            s = 1. - self._hill(self._dose, ic50, h)
            #print(s)
            # Scale the conductance
            b = g * s
            #print(v, b)
            self.simulation1.set_constant(v, b)
            self.simulation2.set_constant(v, b)

    def simulate(self, parameters, times, extra_log=[], reset=True):
        # simulate() method for PINTS

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Update model parameters
        for i, name in enumerate(self.PARAM):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation1.set_time(0)
        self.simulation2.set_time(0)
        if reset:
            self.simulation1.reset()
            self.simulation2.reset()
            self.simulation1.set_state(self.init_state)
            self.simulation2.set_state(self.init_state)
            self._update_dose()

        # Run!
        try:
            self.simulation1.pre(self._n_prepace * self._cl)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times) + 0.02,
                log_times = times,
                log = [
                       self._voltage,
                      ] + extra_log,
                ).npview()
            self.simulation1.set_state(self.simulation2.state())
        except myokit.SimulationError:
            return np.ones(times.shape) * float('inf')

        if len(extra_log) > 0:
            return d
        return d[self._voltage]

    def inet(self, parameters, times, save=None):
        # Compute the I_net in pA/pF.
        # Following Dutta et al. 2017 https://doi.org/10.3389/fphys.2017.00616
        # I_net = (INaL + ICaL) + (IKr + IKs + IK1 + Ito)
        d = self.simulate(parameters, times, extra_log=list(self._i_net),
                          reset=True)
        if save is not None:
            d.save(save)
        i_net = 0
        for i in self._i_net:
            i_net += d[i]  # pA/pF
        return i_net

    def qnet(self, parameters, times, save=None):
        # Compute the qNet metric ($\int I_net dt$) in pC/pF.
        # times in ms.
        # Save inet if `save` is not None.
        if np.abs(self._cl - 2000) > 1e-8:
            print(f'The pacing cycle length is set to {self._c}ms.')
            print('Warning: qNet should be calculated at 0.5Hz (cl=2000).')
        if self._n_prepace != 1000:
            print(f'The number of prepaces is set to {self._n_prepace}.')
            print('Warning: Dutta et al. 2017 used 1000 prepace.')
        if np.abs(times[1] - times[0] - 0.01) > 1e-8:
            print('Warning: Time step should be 0.01ms instead of ' \
                  f'{times[1] - times[0]}ms.')
        if np.abs(times[-1] - times[0] - 2000) > 1e-8:
            print('Warning: qNet should be calculated with time length ' \
                  f'2000ms instead of {times[-1] - times[0]}ms.')
        i_net = self.inet(parameters, times, save)
        return np.trapz(i_net, x=times) * 1e-3  # pA/pF*ms -> pA/pF*s

    def qnet_total(self, parameters, times):
        # Compute the qNet metric with the total current I_ion instead of I_net
        # that is $\int I_ion dt$ in pC/pF.
        # times in ms.
        d = self.simulate(parameters, times, extra_log=[self._i_ion])
        i_ion = d[self._i_ion]  # pA/pF
        return np.trapz(i_ion, x=times) * 1e-3  # pA/pF*ms -> pA/pF*s

    def set_parameters(self, p):
        # return the name of the parameters
        self.PARAM = p

    def fix_hill(self):
        # If True, exclude the Hill coefficient from the fitting parameters.
        hills = ['.hill']  #, '.n']
        x = []
        for i, p in enumerate(self.PARAM):
            for h in hills:
                if h in p:
                    x.append(i)
        if len(x) == 1:
            print(f'Old parameters: {self.PARAM}')
            self.PARAM.pop(x[0])
            print(f'New parameters: {self.PARAM}')
        else:
            raise ValueError(f'Ambigous parameter name to find the Hill'\
                    ' coefficient {self.PARAM}')

    def parameters(self):
        # return the name of the parameters
        return self.PARAM

    def set_temperature(self, value):
        # Set simulation temperature (K)
        try:
            self._set_fix_parameters({'physical_constants.T': value})
        except KeyError:
            self._set_fix_parameters({'nernst.T': value})

    def set_dose(self, value):
        # Set dose concentration (nM)
        self._dose = float(value)
        self._update_dose()

    def _update_dose(self):
        # Set dose concentration (nM) internally
        try:
            assert(self._dose >= 0)
        except NameError:
            raise ValueError('Please set dose value with method `set_dose()`')
        '''
        s1 = self.simulation1.state()
        s1[self._dose_idx] = self._dose
        self.simulation1.set_state(s1)

        s2 = self.simulation2.state()
        s2[self._dose_idx] = self._dose
        self.simulation2.set_state(s2)
        '''
        self._set_fix_parameters({self._model_dose: self._dose})
        #'''

        # Update non-hERG currents
        self._update_non_hERG_parameters()

    def set_ikr_conductance(self, value):
        # Set IKr conductance
        assert(len(self._conductance) == 1)
        self._set_fix_parameters({self._conductance[0]: value})

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.simulation1.set_constant(p, parameters[p])
            self.simulation2.set_constant(p, parameters[p])

    def name(self):
        # name
        return self._model_name

from multiprocessing.sharedctypes import Value
from sympyplus.param import GeneralForm, Param
from sympyplus.form import lhsForm, rhsForm, TestTrialBase
from sympyplus.variational import variationalDerivative, secondVariationalDerivative

def newtonsMethod(lhs_form,rhs_form,const_func,trial_func,test_func):
    const_func = Param(const_func)
    trial_func = Param(trial_func)
    test_func = Param(test_func)

    new_lhs_form = lhsForm(trial_func,test_func)
    for form in lhs_form.forms:
        new_lhs_form.add_form(secondVariationalDerivative(form,const_func,trial_func,test_func))

    new_rhs_form = rhsForm(test_func)
    for form in lhs_form.forms:
        new_rhs_form.add_form(form.mul(-1).new_params(const_func,test_func))
    for form in rhs_form.forms:
        new_rhs_form.add_form(form.new_params(test_func))

    return (new_lhs_form,new_rhs_form)

class PDE_System:
    def __init__(self,domain_PDE,boundary_PDE=None):
        self.set_domain(domain_PDE)
        self.set_boundary(boundary_PDE)
        if (self.domain.trial_func,self.domain.test_func) != (self.boundary.trial_func,self.boundary.test_func):
            raise ValueError("Test and trial functions on domain and boundary must match")
    def __repr__(self):
        if self.boundary is None:
            return f'<PDE System: {self.domain.eqn_str} >'
        else:
            return f'<PDE System: {self.domain.eqn_str}\n             {self.boundary.eqn_str} >'
    @property
    def domain(self):
        return self.__domain
    @property
    def boundary(self):
        return self.__boundary


    def set_domain(self,domain_PDE):
        if domain_PDE is None:
            raise TypeError
        if not isinstance(domain_PDE,PDE):
            raise TypeError
        if not domain_PDE.over == 'domain':
            raise ValueError('Domain PDE must be over domain')
        self.__domain = domain_PDE
    def set_boundary(self,boundary_PDE):
        if boundary_PDE is None:
            self.__boundary = None
        if not isinstance(boundary_PDE,PDE):
            raise TypeError
        if not boundary_PDE.over == 'boundary':
            raise ValueError('Boundary PDE must be over domain')
        self.__boundary = boundary_PDE

class PDE(TestTrialBase):
    def __init__(self,lhs,rhs,trial_func,test_func,over='domain'):
        # set trial and test funcs first
        self.set_trial_func(trial_func)
        self.set_test_func(test_func)
        # then make sure lhs and rhs forms contain test and trial funcs
        self.lhs = lhs
        self.rhs = rhs
        # Set over, ovr
        self.over = over
        self.ovr = None
    def __repr__(self) -> str:
        return f'<PDE : {self.eqn_str}>'
    @property
    def lhs(self):
        return self.__lhs
    @lhs.setter
    def lhs(self,value):
        if not isinstance(value,list):
            raise TypeError(f'Must be list')
        for form in value:
            if not isinstance (form,GeneralForm):
                raise TypeError('Forms must be type GeneralForm')
            if not self.trial_func in form.params or not self.test_func in form.params:
                raise ValueError(f'The form {form.name} of lhs does not contain both test and trial function')
        self.__lhs = value
    @property
    def rhs(self):
        return self.__rhs
    @rhs.setter
    def rhs(self,value):
        if not isinstance(value,list):
            raise TypeError(f'Must be list')
        for form in value:
            if not isinstance (form,GeneralForm):
                raise TypeError('Forms must be type GeneralForm')
            if not self.test_func in form.params:
                raise ValueError(f'The form {form.name} of lhs does not contain test function')
        self.__rhs = value
    @property
    def over(self):
        return self.__over
    @over.setter
    def over(self,value):
        if value not in ('domain','boundary'):
            raise ValueError('Must choose "domain" or "boundary"')
        self.__over = value
    @property
    def ovr(self): # shorthand for self.over
        return self.__ovr
    @ovr.setter
    def ovr(self,value):
        if value is None:
            self.__ovr = 'Ω' if self.over == 'domain' else '∂Ω'
        else:
            self.__ovr = value
    @property
    def ion(self):
        return 'in' if self.over == 'domain' else 'on'
    @property
    def eqn_str(self):
        lhs = ' + '.join([repr(form) for form in self.lhs])
        rhs = ' + '.join([repr(form) for form in self.rhs])
        return f'[{lhs} = {rhs}] {self.ion} {self.ovr}'
    
    def add_form(self,xhs,*forms): # maybe this should be the func used when lhs, rhs are initially created
        if xhs not in ('lhs','rhs'):
            raise ValueError('Must choose "lhs" or "rhs"')
        hs = self.lhs if xhs == 'lhs'else self.rhs
        for form in forms:
            if not isinstance(form,GeneralForm):
                raise TypeError('Form must be type GeneralForm.')
            hs.append(form)
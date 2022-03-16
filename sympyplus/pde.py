from sympyplus.param import GeneralForm, Param
from sympyplus.form import lhsForm, rhsForm, TestTrialBase
from sympyplus.variational import variational_derivative

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
            raise ValueError('Boundary PDE must be over boundary')
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
        if over not in ('domain','boundary'):
            raise ValueError('Must choose "domain" or "boundary"')
        self.__over = over
        self.ovr = None
    @classmethod
    def empty(cls,trial_func,test_func,over='domain'):
        return cls([],[],trial_func,test_func,over)
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
    @property
    def ovr(self): # shorthand for self.over
        if self.__ovr == None:
            return 'Ω' if self.over == 'domain' else '∂Ω'
        else:
            return self.__ovr
    @ovr.setter
    def ovr(self,value):
        if value is None:
            self.__ovr = None
            return
        if not isinstance(value,str):
            raise TypeError('Must be a string')
        if len(value) != 1:
            raise ValueError('Must be a single character')
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
    
    def newtons_method(self,trial_func_prev,test_func,diff_func):
        """ Performs Newton's method on 'self', returns new pde 'new'.
        
        Newton's method can be thought of as the following:

        Given a binary form A[Q](P) and a linear form F[P], we wish to find Q such that

                A[Q](P) = F[P],                                     for all P.

        We begin Newton's method with an initial guess Q(0), which must be sufficiently close
        to the true solution Q. Then for each integer (n+1) we solve the following: Find
        Q(n+1) such that
        
                ∂A[Q(n)](P,Q(n+1)-Q(n)) = -A[Q(n)](P) + L[P],       for all P.
        
        To simplify, we may define S(n+1) := Q(n+1) - Q(n) and then find S(n+1) such that

                ∂A[Q(n)](P,S(n+1)) = -A[Q(n)](P) + L[P],            for all P.
        
        Then we simply add Q(n) to S(n+1) to achieve the desired Q(n+1). We repeat the process
        until S(n+1) is sufficiently close to zero.

        In the case of this function, I have chosen the following naming convention:

            trial_func_prev     :           Q(n)
            test_func           :           P
            diff_func           :           S(n+1)
        
        Newton's method below uses this convention.
         """

        # define new empty PDE
        new = PDE.empty(diff_func,test_func)

        # add forms to lhs (could maybe use map() here)
        for form in self.lhs:
            new.add_form('lhs',variational_derivative(form,trial_func_prev,test_func,diff_func))

        # add forms to rhs
        for form in self.lhs:
            new.add_form('rhs',form.mul(-1).new_params(trial_func_prev,test_func))
        for form in self.rhs:
            new.add_form('rhs',form.new_params(test_func))

        # return completed PDE
        return new
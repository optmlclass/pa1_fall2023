'''provides  implementations of various operations, subclasses of Operation.'''
import functools
import numpy as np

from operation import Operation
from variable import Variable


class VariableAdd(Operation):
    '''tensor addition operation.
    given a list of Variable objects with a common shape, adds them up.'''

    def __init__(self):
        super(VariableAdd, self).__init__(name="Add")

    def forward_call(self, summands):
        '''summands should be a list of ndarrays, all with the same dimensions.
        args:
            summands: list of Variables all with same shape.
        returns:
            numpy array of same shape as input Variables containing the sum
            of the input tensors.
            
        Example: 
        if  summands = [A, B, C] where
        A.data = np.array([1, 2, 0])
        B.data = np.array([2, 3, 10])
        C.data = np.array([3, -2, 90]),
        
        the return value should be:
        np.array([6, 3, 100]).'''

        assert len(summands) > 0, "Add called with no inputs!"
        shape = summands[0].data.shape
        for summand in summands:
            assert summand.data.shape == shape, "Shape mismatch in Add: shapes: {}".format(
                [summand.data.shape for summand in summands])

        ### YOUR CODE HERE ###
        raise NotImplementedError 

    def backward_call(self, downstream_grad):
        '''
        backward functions for tensor addition.
        args:
            downstream_grad: numpy array containing gradient of final
            computation output with respect to the output of this operation.
        returns: list of  numpy arrays such that ith entry of returned list
            is gradient of final computation output with respect to ith entry
            of the input list to forward_call.'''

        ### YOUR CODE HERE ###
        raise NotImplementedError 

class VariableMultiply(Operation):
    '''coordinate-wise multiply operation.'''

    def __init__(self):
        super(VariableMultiply, self).__init__(name="Multiply")

    def forward_call(self, multiplicands):
        '''inputs should be a list of Variables, all with the same dimensions.
        Like all forward_call implementations, this function should also
        set self.parents appriopriately.
        args:
            multiplicands: list of Variables all of same shape.
        returns:
            a numpy array of the same shape as all the input Variables that is
            equal to all the Variables multiplied together entry-wise.
        
        Example:
        if multiplicands = [A, B, C] where
        A.data = np.array([1, 2, 0])
        B.data = np.array([2, 3, 10])
        C.data = np.array([3, -2, 90]),
        
        the return value should be:
        np.array([6, -12, 0]).
        '''

        assert len(multiplicands) > 0, "Multiply called with no inputs!"
        shape = multiplicands[0].data.shape
        for multiplicand in multiplicands:
            assert multiplicand.data.shape == shape, "Shape mismatch in Multiply!"

        ### YOUR CODE HERE ###
        raise NotImplementedError 

    def backward_call(self, downstream_grad):


        ### YOUR CODE HERE ###

        raise NotImplementedError 


class ScalarMultiply(Operation):
    '''multiplication by a scalar.'''

    def __init__(self):
        super(ScalarMultiply, self).__init__(name="Scalar Multiply")

    def forward_call(self, scalar, tensor):
        '''
        multiplies a tensor by a scalar.
        args:
            scalar: a Variable of shape (1) (i.e. np.shape(scalar.data) is (1).
            tensor: a Variable of arbitrary shape.
            
        returns: a numpy array of same shape as input tensor containing the result
            multiplying each element of tensor by scalar.
        
        Example:
        if scalar.data is np.array([3.0]) and tensor.data is np.array([1.0, 2.0, 3.0])
        then the return value should be np.array([3.0, 6.0, 9.0])'''

        assert scalar.data.size == 1, "ScalarMultiply called with non-scalar input!"

        ### YOUR CODE HERE ###
        raise NotImplementedError 

    def backward_call(self, downstream_grad):

        ### YOUR CODE HERE  ###
        raise NotImplementedError 



class MatrixMultiply(Operation):
    '''matrix multiplication operation.'''
    def __init__(self):
        super(MatrixMultiply, self).__init__(name="MatrixMultiply")

    def forward_call(self, A, B):
        '''
        computes a matrix multiply forward pass.
        args:
            A: a 2-dimensional Variable (i.e. a matrix) of shape (x, y)
            B: a 2-dimensional Variable of shape (y, z).
            
        returns:
            a numpy array of shape (x, z) containing the matrix product of A
            and B.'''
        assert len(A.data.shape) == 2 and len(B.data.shape) == 2, \
            "inputs to matrix multiply are not matrices! A shape: {}, B shape: {}".format(A.data.shape, B.data.shape)
        
        ### YOUR CODE HERE ###
        raise NotImplementedError 

    def backward_call(self, downstream_grad):

        ### YOUR CODE HERE ###

        raise NotImplementedError 


class HingeLoss(Operation):
    '''compute hinge loss.
    assumes input are a scores for a single example (no need to support
    minibatches of scores).
    
    Input "scores" will be [1 x C] tensor representing scores for each of
    C classes.
    "label" is an integer in [0,..., C-1].
    The multi-class hinge loss is given by the (unweighted) formula here:
    https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html
    
    '''

    def __init__(self, label):
        super(HingeLoss, self).__init__(name="Hinge Loss")
        self.label = label

    def forward_call(self, scores):
        '''
        forward pass for Hinge Loss.
        args:
            scores: 1xC Variable object containing scores for different classes.
        returns:
            float or shape (1) numpy array containing multiclass hinge loss.
        '''

        ### YOUR CODE HERE ###

        raise NotImplementedError 

    def backward_call(self, downstream_grad):
        '''
        backward pass for Hinge Loss.
        args:
            downstream_grad: shape (1) numpy array or float containing
                downstream grad in backpropogation (gradient of final
                output with respect to output of the hinge loss).
        returns:
            gradient of final output with respect to input scores of hinge loss.
        '''

        ### YOUR CODE HERE ###

        raise NotImplementedError 

class Power(Operation):
    '''raise to a power'''

    def __init__(self, exponent):
        super(Power, self).__init__(name="{} Power".format(exponent))

        self.exponent = exponent

    def forward_call(self, tensor):
        self.parents = [tensor]

        return np.power(tensor.data, self.exponent)

    def backward_call(self, downstream_grad):
        tensor = self.parents[0]
        return [downstream_grad * self.exponent * np.power(tensor.data, self.exponent - 1.0)]


class Exp(Operation):
    '''exponentiate'''

    def __init__(self):
        super(Exp, self).__init__(name="exp")

    def forward_call(self, tensor):
        self.parents = [tensor]
        self.output = np.exp(tensor.data)
        return self.output

    def backward_call(self, downstream_grad):
        return [downstream_grad * self.output]


class Maximum(Operation):
    '''computes coordinate-wise maximum of a list of tensors'''

    def __init__(self):
        super(Maximum, self).__init__(name="maximum")

    def forward_call(self, terms):
        '''
        args:
            terms: a list of Variable objects to compute maximum.
        returns:
            a numpy array whose ith coordinate is the maximum value of the
            ith coordinate of all the Variables in terms.'''
        self.parents = terms
        self.output = functools.reduce(
            lambda x, y: np.maximum(x, y), [t.data for t in terms])

        return self.output

    def backward_call(self, downstream_grad):
        masks = [t.data == self.output for t in self.parents]

        return [m * downstream_grad for m in masks]

class ReLU(Operation):
    '''computes coordinate-wise maximum with 0'''

    def __init__(self):
        super(ReLU, self).__init__(name="relu")

    def forward_call(self, A):
        self.parents = [A]
        self.output = np.maximum(A.data, 0.0)

        return self.output

    def backward_call(self, downstream_grad):
        mask = (self.output == self.parents[0].data)
        return [mask * downstream_grad]

class ReduceMax(Operation):
    '''computes the maximum element of a tensor'''

    def __init__(self):
        super(ReduceMax, self).__init__(name="ReduceMax")

    def forward_call(self, A):
        self.parents = [A]
        self.output = np.max(A.data)

        return self.output

    def backward_call(self, downstream_grad):
        A = self.parents[0]

        mask = (A.data == self.output)
        return [mask * downstream_grad]


class TensorDot(Operation):
    def __init__(self):
        super(TensorDot, self).__init__(name="TensorDot")

    def forward_call(self, A, B, dims_to_contract):
        '''A and B are are Variables, dims_to_contract is number
        of dimensions to contract. This is a special case of np.tensordot.
        Example:
        A is dim [2, 3, 4]
        B is dim [3, 4, 5]

        if dims_to_contract is 2, output will be [2, 5]
        Otherwise it is an error.
        '''

        self.parents = [A, B]
        self.dims_to_contract = dims_to_contract

        return np.tensordot(A.data, B.data, dims_to_contract)

    def backward_call(self, downstream_grad):
        A = self.parents[0]
        B = self.parents[1]
        A_indices = np.arange(0, len(A.data.shape) - self.dims_to_contract)
        B_indices = np.arange(self.dims_to_contract, len(B.data.shape))
        A_grad = np.tensordot(downstream_grad, B.data, [B_indices, B_indices])
        B_grad = np.tensordot(A.data, downstream_grad, [A_indices, A_indices])
        return [A_grad, B_grad]


###### Helper functions for operator overloading ######

Variable.__add__ = lambda self, other: VariableAdd()([self, other])

def mul(self, other):
    if not isinstance(other, Variable):
        other = Variable(other)
    if other.data.size == 1:
        return ScalarMultiply()(other, self)
    else:
        return VariableMultiply()([self, other])
Variable.__mul__ = Variable.__rmul__ = mul

Variable.__neg__ = lambda self: -1 * self
Variable.__sub__ = lambda self, other: self + (- other)

def div(A, B):
    if not isinstance(B, Variable):
        B = Variable(B)
    B_inverse = Power(-1.0)(B)
    if B.data.size == 1:
        return ScalarMultiply()(B_inverse, A)
    else:
        return VariableMultiply()([A, B_inverse])

Variable.__truediv__ = div
Variable.__rtruediv__ = lambda self, other: div(other, self)


###### Helper functions for applying operations ######

def matmul(a, b):
    mm = MatrixMultiply()
    return mm(a, b)

def pow(tensor, exponent):
    power = Power(exponent)
    return power(tensor)

def tensordot(A, B, dims_to_contract):
    tensordot_op = TensorDot()
    return tensordot_op(A, B, dims_to_contract)

def relu(A):
    relu_op = ReLU()
    return relu_op(A)


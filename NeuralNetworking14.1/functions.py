from numpy import *
import math

#activation functions

class reluactivation():
    def function(self, x):
        return x * (x > 0)
    
    def derivative(self, x):
        return 1 * (x > 0)
        
class leakyreluactivation():
    def __init__(self, a=0.1): 
        self.a = a
        
    def function(self, x):
        return maximum(x * self.a, x)

    def derivative(self, x):
        return 1 if x > 0 else self.a
        
class tanhactivation():
    def function(self, x):
        return math.tanh(x)
    
    def derivative(self, x):
        tah = math.tanh(x)
        return 1 - tah ** 2

def softmax(x):
    exps = exp(x - max(x))
    return exps / sum(exps)

class sigmoidactivation():
    def function(self, x):
        try:
            return 1 / (1 + exp(-x))
        except (OverflowError, RuntimeWarning) as e:
            return 1 * (x > 0)
            
    def derivative(self, x):
        sig = self.function(x)
        return sig * (1 - sig)

class customizedsigmoidactivation():
    def __init__(self, margin=0.05): #margin > 0 | bounds -> y = 0 at 0 and y = 1 at 1 | equation form = 1 / (1 + exp(-a * (x-b)))
        self.marginlower = margin
        self.marginupper = 1 - margin
        
        self.a = log(1/margin - 1) * 2
        self.equation = lambda x : 1 / (1 + exp(-self.a * (x-0.5)))
            
    def function(self, x):
        try:
            y = self.equation(x)
            if self.marginlower < y < self.marginupper: #if y >= (1-self.margin): return 1 elif y <= self.margin: return 0 else: return y
                return y
            else:
                return 1 * (y >= self.marginupper)
        except OverflowError:
            return 1 * (x > 0)
            
    def derivative(self, x):
        sig = self.equation(x)
        if self.marginlower < x < self.marginupper:
            return self.a * sig * (1 - sig)
        else:
            return 0

def softmax(array): #test with e.g. print(softmax(array([1.0, 2.0, 7.0])))
    ex = sum([exp(x) for x in array])
    return [exp(x) / ex for x in array]

#error functions
 
def meansquarederror(output, expected):
    if type(output) == type(int()) or type(output) == type(float()):
        return (output - expected) ** 2
    elif type(output) == type(list()):
        return sum([(output[i] - expected[i]) ** 2 for i in range(len(output))]) / len(output)
    
def meanerror(output, expected):
    if type(output) == type(int()) or type(output) == type(float()):
        return abs(output - expected)
    elif type(output) == type(list()):
        return sum([abs(output[i] - expected[i]) for i in range(len(output))]) / len(output)
    
#normalization functions

def minmaxnormalization(input, minmax=None, bounds=[0, 1]): #input e.g. [[a1, a2], [b1, b2], [c1, c2], [d1, d2]], test with #print(minmaxnormalization([[9, 0], [10, 10], [1, 8]])) | minmax, to define min and max if other values exist, minmax = [min, max], use 765 as max (255 * 3)
    for argument in range(len(input[0])):
        
        if minmax == None:
            minmax = [min([element[argument] for element in input]), max([element[argument] for element in input])]
        
        for element in range(len(input)):
            try:
                input[element][argument] = nan_to_num((input[element][argument] - minmax[0]) / (minmax[1] - minmax[0]) * (bounds[1] - bounds[0]) + bounds[0], nan=(bounds[1] + bounds[0]) / 2, posinf=(bounds[1] + bounds[0]) / 2, neginf=(bounds[1] + bounds[0]) / 2) #schutz vor nan
            except ZeroDivisionError:
                input[element][argument] = (bounds[1] + bounds[0]) / 2
            
    return input

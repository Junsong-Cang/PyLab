from numpy import pi, log, sqrt
import numpy as np

nDims = 3
nDerived = 0
p1_range = {'min':-5,'max':10, 'start':1}
p2_range = {'min':-5,'max':10, 'start':1}
p3_range = {'min':-5,'max':10, 'start':1}

def Model(p1,p2,p3,x):
	return p1+p2*x+p3*x**2

def Squeeze(Range,HyperParam):
  '''Return physical param value for HyperParam in [-1,1]'''
  y1 = Range['min']
  y2 = Range['max']
  return (y2-y1)*(HyperParam+1)/2+y1

a=Squeeze(p1_range, -1.3322583E-01)
print(a)

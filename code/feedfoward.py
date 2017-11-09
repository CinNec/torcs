import pickle
import numpy as np


#a = open("drivelog.pickle",'r', encoding="utf-8")
#log = pickle.load(a)
#print(log)

with open("drivelog.pickle", 'rb') as logfile:
    a = pickle.Unpickler(logfile)
    print(type(a))
    print(a)
    
def forward():
    return
    
    
def backward():
    return
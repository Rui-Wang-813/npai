import sys
sys.path.append("..")
sys.path.append("../npai")

from npai import Variable, variable

var = variable([1,2,3])
print(var.value)
print(var.grad)
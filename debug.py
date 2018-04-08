from inspect import getsource

def code(function):
    print(getsource(function))

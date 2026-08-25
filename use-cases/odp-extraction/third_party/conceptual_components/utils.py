# Vendored verbatim from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# path: conceptual-components-extraction/project_root/utils.py


def getID(u):
    last = u.rfind("#")
    if last < 0:
        last = u.rfind("/")
    return u[last + 1:]

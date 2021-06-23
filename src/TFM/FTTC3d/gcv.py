# This file is a shim to avoid having to replicate the content of the gcv file,
# while avoiding potential issues with pythons import system.

# This is to tell editors, that this function exists

import os


def gcv(U, s, b, lambdarange, plot: bool):
    """ See gcv in FTTC module"""
    # This definition is used to make sure, this function does exist in tfm3d.py
    raise NotImplementedError


_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "FTTC", "gcv.py")
exec(compile(open(_path).read(), 'gcv', 'exec'))

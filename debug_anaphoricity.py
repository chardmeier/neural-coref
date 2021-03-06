# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import anaphoricity
import pydevd
import sys

print('Connection to debug server...', file=sys.stderr)
pydevd.settrace('localhost', port=21000, stdoutToServer=True, stderrToServer=True)

print('Starting main script...', file=sys.stderr)
anaphoricity.main()

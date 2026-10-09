"""Check imports only; does not install or change software."""
import importlib
import sys

if __name__ == '__main__':
    print('Python:', sys.version)
    for name in ['rq1_hr', 'numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'matplotlib', 'wfdb']:
        module = importlib.import_module(name)
        print(name, getattr(module, '__version__', 'import OK'))

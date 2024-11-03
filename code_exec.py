from RestrictedPython import compile_restricted
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Guards import safe_globals, safe_builtins, guarded_iter_unpack_sequence
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Utilities import utility_builtins
from io import StringIO

def eval_restricted_script(script):
    # Set up print collector and output handling
    all_prints = StringIO()
    
    class CustomPrintCollector:
        """Collect printed text, accumulating in shared StringIO"""
        
        def __init__(self, _getattr_=None):
            self.txt = []
            self._getattr_ = _getattr_
        
        def write(self, text):
            all_prints.write(text)
            self.txt.append(text)
            
        def __call__(self):
            result = ''.join(self.txt)
            return result
            
        def _call_print(self, *objects, **kwargs):
            if kwargs.get('file', None) is None:
                kwargs['file'] = self
            else:
                self._getattr_(kwargs['file'], 'write')
            
            print(*objects, **kwargs)

    # Create the restricted builtins dictionary
    restricted_builtins = dict(safe_builtins)
    restricted_builtins.update(utility_builtins)  # Add safe __import__
    restricted_builtins.update({
        # Print handling
        '_print_': CustomPrintCollector,
        '_getattr_': getattr,
        '_getiter_': default_guarded_getiter,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        
        # Define allowed imports
        '__allowed_modules__': ['math'],
        '__import__': __import__,
        
        # Basic functions
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        
        # Math operations
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'pow': pow,
        
        # Type conversions
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'set': set,
        'dict': dict,
        'bytes': bytes,
        'bytearray': bytearray,
        
        # Sequence operations
        'all': all,
        'any': any,
        'sorted': sorted,
        'reversed': reversed,
        
        # String operations
        'chr': chr,
        'ord': ord,
        
        # Other safe operations
        'isinstance': isinstance,
        'issubclass': issubclass,
        'hasattr': hasattr,
        'callable': callable,
        'format': format,
    })

    # Create the restricted globals dictionary
    restricted_globals = dict(safe_globals)
    restricted_globals['__builtins__'] = restricted_builtins

    try:
        byte_code = compile_restricted(script, filename='<inline>', mode='exec')
        exec(byte_code, restricted_globals)
        
        return {
            'prints': all_prints.getvalue(),
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

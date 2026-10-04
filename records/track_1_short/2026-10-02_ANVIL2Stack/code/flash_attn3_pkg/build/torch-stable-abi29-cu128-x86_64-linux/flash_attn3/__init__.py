import ctypes
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _import_from_path(file_path: Path) -> ModuleType:
    # We cannot use the module name as-is, after adding it to `sys.modules`,
    # it would also be used for other imports. So, we make a module name that
    # depends on the path for it to be unique using the hex-encoded hash of
    # the path.
    path_hash = "{:x}".format(ctypes.c_size_t(hash(file_path.absolute())).value)
    module_name = path_hash
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Cannot load module {module_name} from spec")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


globals().update(vars(_import_from_path(Path(__file__).parent.parent / "__init__.py")))

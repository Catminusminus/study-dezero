__version__ = "0.1.0"
is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (
        Variable,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
        setup_variable,
    )
else:
    from dezero.core import (
        Variable,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
        setup_variable
    )
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.dataloaders import Dataloader
    import dezero.functions
    import dezero.datasets


setup_variable()

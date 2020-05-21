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
        setup_variable,
        Config,
        test_mode,
    )
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.dataloaders import DataLoader
    from dezero.dataloaders import SeqDataLoader
    import dezero.functions
    import dezero.datasets
    import dezero.optimizers


setup_variable()

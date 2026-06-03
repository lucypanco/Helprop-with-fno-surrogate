"""Conditional surrogate tools for HelProp.

Import concrete APIs from their modules, for example:

``helprop_surrogate.model``
``helprop_surrogate.kernel``
``helprop_surrogate.matrix_data``
``helprop_surrogate.fno.model``

The package initializer intentionally avoids importing CLI modules so
``python -m helprop_surrogate.<module>`` does not trigger runpy warnings.
"""

__all__: list[str] = []

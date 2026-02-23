"""
PageIndex — Backward compatibility shim.

This module re-exports from idpkit.engine for backward compatibility.
New code should import from idpkit directly.
"""

from idpkit.engine.page_index import *  # noqa: F401,F403
from idpkit.engine.page_index_md import md_to_tree  # noqa: F401
from idpkit.engine.utils import config, ConfigLoader  # noqa: F401

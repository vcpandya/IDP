"""IDP Kit Smart Tools — Swiss Army Knife document processing skills.

All 13 Smart Tools are registered here in TOOL_REGISTRY for auto-discovery
by the API routes and agent system.
"""

from idpkit.tools.smart_summary import SmartSummaryTool
from idpkit.tools.smart_classify import SmartClassifyTool
from idpkit.tools.smart_extract import SmartExtractTool
from idpkit.tools.smart_compare import SmartCompareTool
from idpkit.tools.smart_qa import SmartQATool
from idpkit.tools.smart_split import SmartSplitTool
from idpkit.tools.smart_redaction import SmartRedactionTool
from idpkit.tools.smart_anonymize import SmartAnonymizeTool
from idpkit.tools.smart_fill import SmartFillTool
from idpkit.tools.smart_rewrite import SmartRewriteTool
from idpkit.tools.smart_translate import SmartTranslateTool
from idpkit.tools.smart_merge import SmartMergeTool
from idpkit.tools.smart_audit import SmartAuditTool

# Registry: tool_name -> tool instance
TOOL_REGISTRY: dict[str, "BaseTool"] = {}

_TOOL_CLASSES = [
    SmartSummaryTool,
    SmartClassifyTool,
    SmartExtractTool,
    SmartCompareTool,
    SmartQATool,
    SmartSplitTool,
    SmartRedactionTool,
    SmartAnonymizeTool,
    SmartFillTool,
    SmartRewriteTool,
    SmartTranslateTool,
    SmartMergeTool,
    SmartAuditTool,
]

for _cls in _TOOL_CLASSES:
    _instance = _cls()
    TOOL_REGISTRY[_instance.name] = _instance

__all__ = [
    "TOOL_REGISTRY",
    "SmartSummaryTool",
    "SmartClassifyTool",
    "SmartExtractTool",
    "SmartCompareTool",
    "SmartQATool",
    "SmartSplitTool",
    "SmartRedactionTool",
    "SmartAnonymizeTool",
    "SmartFillTool",
    "SmartRewriteTool",
    "SmartTranslateTool",
    "SmartMergeTool",
    "SmartAuditTool",
]

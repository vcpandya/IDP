"""Smart document metadata layer.

Category-aware extraction of typed key-value metadata ("facets") from indexed
documents. A profiling pass classifies each document into a category (case law,
contract, financial statement, ...) then extracts a standard + contextual set of
key-value pairs for that category. The resulting facets power the Document Map —
a faceted graph for discovering and pre-filtering sets of documents.
"""

"""
Utility functions for constructing OpenSearch queries consistently.
"""
from typing import Optional, Union, List


def build_field_filter_clause(field_name: str, values: List) -> Optional[dict]:
    """Build an OpenSearch filter clause for a field from a list of filter values.

    Knowledge-filter ``data_sources`` values frequently carry a trailing
    wildcard (e.g. ``"<orgId>__*"`` to scope by filename prefix). A plain
    ``term`` clause treats the ``*`` as a literal character and matches
    nothing, which silently drops every document for the scope. This helper:

    - returns an impossible-value ``term`` when ``values`` is empty
      (preserving the "match nothing" semantics of an explicit empty array);
    - skips bare ``"*"`` values (they mean "match everything" for the field,
      so no clause is needed) and returns ``None`` if that leaves nothing;
    - emits a ``wildcard`` clause for any value containing ``*`` or ``?``;
    - emits ``term``/``terms`` for the remaining exact values;
    - ORs multiple clauses for the same field via ``bool.should``.

    Args:
        field_name: Backend OpenSearch field (e.g. ``filename``, ``owner``).
        values: List of filter values from the knowledge filter ``query_data``.

    Returns:
        An OpenSearch query clause dict, or ``None`` when the field should not
        constrain the query at all (e.g. only ``"*"`` was supplied).
    """
    if values is None or not isinstance(values, list):
        return None
    if len(values) == 0:
        # Explicit empty array means "match nothing".
        return {"term": {field_name: "__IMPOSSIBLE_VALUE__"}}

    # A bare "*" means "match everything" for this field — drop it so it does
    # not get treated as a literal term.
    meaningful = [v for v in values if v != "*"]
    if not meaningful:
        return None

    term_values: List = []
    wildcard_values: List[str] = []
    for v in meaningful:
        if isinstance(v, str) and ("*" in v or "?" in v):
            wildcard_values.append(v)
        else:
            term_values.append(v)

    clauses: List[dict] = []
    if term_values:
        if len(term_values) == 1:
            clauses.append({"term": {field_name: term_values[0]}})
        else:
            clauses.append({"terms": {field_name: term_values}})
    for wv in wildcard_values:
        clauses.append({"wildcard": {field_name: {"value": wv}}})

    if len(clauses) == 1:
        return clauses[0]
    return {"bool": {"should": clauses, "minimum_should_match": 1}}


def build_filename_query(filename: str) -> dict:
    """
    Build a standardized query for finding documents by filename.

    Args:
        filename: The exact filename to search for

    Returns:
        A dict containing the OpenSearch query body
    """
    return {
        "term": {
            "filename": filename
        }
    }


def build_filename_search_body(filename: str, size: int = 1, source: Union[bool, List[str]] = False) -> dict:
    """
    Build a complete search body for checking if a filename exists.

    Args:
        filename: The exact filename to search for
        size: Number of results to return (default: 1)
        source: Whether to include source fields, or list of specific fields to include (default: False)

    Returns:
        A dict containing the complete OpenSearch search body
    """
    return {
        "query": build_filename_query(filename),
        "size": size,
        "_source": source
    }


def build_filename_delete_body(filename: str) -> dict:
    """
    Build a delete-by-query body for removing all documents with a filename.

    Args:
        filename: The exact filename to delete

    Returns:
        A dict containing the OpenSearch delete-by-query body
    """
    return {
        "query": build_filename_query(filename)
    }
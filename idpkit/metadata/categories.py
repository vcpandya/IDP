"""Document category registry and per-category metadata schemas.

Each category declares a set of *standard* fields. During extraction the LLM is
asked to fill these standard fields and may additionally surface a few
*contextual* fields it considers important for the specific document — so the
schema is a strong guide, not a hard cap.

Field shape:
    {
        "key": "judge",          # stable machine key (snake_case)
        "label": "Judge",         # human label
        "type": "list",           # one of: text | date | number | list
        "description": "...",     # guidance shown to the extractor LLM
    }

``type == "list"`` fields expand into one facet row per value, so a document with
three parties produces three "party" facets that can each be filtered on.
"""

from __future__ import annotations

# Canonical category key used when a document does not fit a specialised type.
DEFAULT_CATEGORY = "general"


CATEGORIES: dict[str, dict] = {
    "general": {
        "label": "General Document",
        "description": "Any document that does not fit a more specific category.",
        "fields": [
            {"key": "document_type", "label": "Document Type", "type": "text",
             "description": "Short description of what kind of document this is."},
            {"key": "title", "label": "Title", "type": "text",
             "description": "The document's title or main heading."},
            {"key": "author", "label": "Author", "type": "text",
             "description": "Primary author or issuing person."},
            {"key": "organization", "label": "Organization", "type": "text",
             "description": "Primary organization the document belongs to or is about."},
            {"key": "date", "label": "Date", "type": "date",
             "description": "Most relevant date (publication / creation)."},
            {"key": "topic", "label": "Topics", "type": "list",
             "description": "Key topics or subjects covered."},
            {"key": "language", "label": "Language", "type": "text",
             "description": "Primary language of the document."},
        ],
    },
    "case_law": {
        "label": "Case Law / Judgment",
        "description": "A court judgment, ruling, order, or reported case.",
        "fields": [
            {"key": "court", "label": "Court", "type": "text",
             "description": "Name of the court that decided the matter."},
            {"key": "judge", "label": "Judge", "type": "list",
             "description": "Judge(s) / justice(s) who presided over or authored the decision."},
            {"key": "case_number", "label": "Case Number", "type": "text",
             "description": "Docket / case / appeal number."},
            {"key": "citation", "label": "Citation", "type": "text",
             "description": "Neutral or reporter citation, if present."},
            {"key": "party", "label": "Parties", "type": "list",
             "description": "Named parties (petitioner, respondent, appellant, etc.)."},
            {"key": "decision_date", "label": "Decision Date", "type": "date",
             "description": "Date the judgment/order was delivered."},
            {"key": "jurisdiction", "label": "Jurisdiction", "type": "text",
             "description": "Country / state / region of jurisdiction."},
            {"key": "case_type", "label": "Case Type", "type": "text",
             "description": "Nature of the case (civil, criminal, constitutional, tax, etc.)."},
            {"key": "outcome", "label": "Outcome", "type": "text",
             "description": "Disposition / result (allowed, dismissed, convicted, etc.)."},
            {"key": "legal_topic", "label": "Legal Topics", "type": "list",
             "description": "Areas of law or key legal issues involved."},
        ],
    },
    "contract": {
        "label": "Contract / Agreement",
        "description": "A contract, agreement, MOU, NDA, lease, or similar binding document.",
        "fields": [
            {"key": "contract_type", "label": "Contract Type", "type": "text",
             "description": "Kind of agreement (NDA, service agreement, lease, SOW, etc.)."},
            {"key": "party", "label": "Parties", "type": "list",
             "description": "Contracting parties / signatories."},
            {"key": "effective_date", "label": "Effective Date", "type": "date",
             "description": "Date the contract takes effect."},
            {"key": "expiration_date", "label": "Expiration / Term End", "type": "date",
             "description": "End date or termination date, if stated."},
            {"key": "governing_law", "label": "Governing Law", "type": "text",
             "description": "Governing law / jurisdiction clause."},
            {"key": "total_value", "label": "Contract Value", "type": "text",
             "description": "Monetary value / consideration, with currency."},
            {"key": "subject_matter", "label": "Subject Matter", "type": "text",
             "description": "What the contract is about."},
        ],
    },
    "act_legislation": {
        "label": "Act / Legislation",
        "description": "A statute, act, regulation, bill, rule, or other piece of legislation.",
        "fields": [
            {"key": "title", "label": "Title", "type": "text",
             "description": "Short / official title of the act or regulation."},
            {"key": "jurisdiction", "label": "Jurisdiction", "type": "text",
             "description": "Country / state / body that enacted it."},
            {"key": "act_number", "label": "Act / Bill Number", "type": "text",
             "description": "Act number, chapter, or bill identifier."},
            {"key": "enactment_date", "label": "Enactment Date", "type": "date",
             "description": "Date of enactment / commencement."},
            {"key": "subject_area", "label": "Subject Area", "type": "list",
             "description": "Subject areas the legislation regulates."},
            {"key": "authority", "label": "Issuing Authority", "type": "text",
             "description": "Legislature / ministry / regulator that issued it."},
        ],
    },
    "financial_statement": {
        "label": "Financial Statement",
        "description": "Balance sheet, income statement, cash flow, annual report, or filing.",
        "fields": [
            {"key": "company", "label": "Company", "type": "text",
             "description": "Entity the statement belongs to."},
            {"key": "statement_type", "label": "Statement Type", "type": "text",
             "description": "Balance sheet, income statement, cash flow, annual report, 10-K, etc."},
            {"key": "fiscal_year", "label": "Fiscal Year", "type": "text",
             "description": "Fiscal year covered."},
            {"key": "fiscal_period", "label": "Period", "type": "text",
             "description": "Reporting period (Q1, FY, half-year, etc.)."},
            {"key": "currency", "label": "Currency", "type": "text",
             "description": "Reporting currency."},
            {"key": "auditor", "label": "Auditor", "type": "text",
             "description": "Auditing firm, if stated."},
            {"key": "reporting_standard", "label": "Reporting Standard", "type": "text",
             "description": "Accounting standard (IFRS, US GAAP, etc.)."},
        ],
    },
    "invoice": {
        "label": "Invoice / Bill",
        "description": "An invoice, bill, receipt, or purchase order.",
        "fields": [
            {"key": "invoice_number", "label": "Invoice Number", "type": "text",
             "description": "Invoice / bill / PO number."},
            {"key": "vendor", "label": "Vendor / Seller", "type": "text",
             "description": "Issuing vendor / seller."},
            {"key": "customer", "label": "Customer / Buyer", "type": "text",
             "description": "Billed customer / buyer."},
            {"key": "invoice_date", "label": "Invoice Date", "type": "date",
             "description": "Date the invoice was issued."},
            {"key": "due_date", "label": "Due Date", "type": "date",
             "description": "Payment due date."},
            {"key": "total_amount", "label": "Total Amount", "type": "text",
             "description": "Total amount due, with currency."},
        ],
    },
    "research_paper": {
        "label": "Research Paper / Article",
        "description": "An academic paper, journal article, whitepaper, or preprint.",
        "fields": [
            {"key": "title", "label": "Title", "type": "text",
             "description": "Title of the paper."},
            {"key": "author", "label": "Authors", "type": "list",
             "description": "Authors of the paper."},
            {"key": "publication_date", "label": "Publication Date", "type": "date",
             "description": "Date of publication."},
            {"key": "venue", "label": "Venue / Journal", "type": "text",
             "description": "Journal, conference, or publisher."},
            {"key": "institution", "label": "Institution", "type": "list",
             "description": "Affiliated institutions."},
            {"key": "keyword", "label": "Keywords", "type": "list",
             "description": "Keywords / index terms."},
            {"key": "doi", "label": "DOI", "type": "text",
             "description": "DOI or other identifier."},
        ],
    },
    "resume": {
        "label": "Resume / CV",
        "description": "A resume, CV, or candidate profile.",
        "fields": [
            {"key": "candidate_name", "label": "Candidate", "type": "text",
             "description": "Name of the candidate."},
            {"key": "current_role", "label": "Current Role", "type": "text",
             "description": "Most recent job title."},
            {"key": "employer", "label": "Employers", "type": "list",
             "description": "Companies the candidate has worked at."},
            {"key": "skill", "label": "Skills", "type": "list",
             "description": "Key skills / technologies."},
            {"key": "education", "label": "Education", "type": "list",
             "description": "Degrees / institutions."},
            {"key": "location", "label": "Location", "type": "text",
             "description": "Candidate location."},
        ],
    },
}


def list_categories() -> list[dict]:
    """Return all categories as a serialisable list, including their schemas."""
    return [
        {
            "key": key,
            "label": spec["label"],
            "description": spec["description"],
            "fields": spec["fields"],
        }
        for key, spec in CATEGORIES.items()
    ]


def get_category(key: str | None) -> dict:
    """Return the category spec for *key*, falling back to the default category."""
    if key and key in CATEGORIES:
        return CATEGORIES[key]
    return CATEGORIES[DEFAULT_CATEGORY]


def get_category_keys() -> list[str]:
    """Return the list of valid category keys."""
    return list(CATEGORIES.keys())


def field_label(category_key: str | None, field_key: str) -> str:
    """Best-effort human label for a (category, field) pair.

    Falls back to a title-cased version of the key so unexpected/contextual
    fields surfaced by the LLM still display reasonably.
    """
    spec = get_category(category_key)
    for field in spec["fields"]:
        if field["key"] == field_key:
            return field["label"]
    # Also check the general schema (shared keys like title/author/date).
    for field in CATEGORIES[DEFAULT_CATEGORY]["fields"]:
        if field["key"] == field_key:
            return field["label"]
    return field_key.replace("_", " ").title()

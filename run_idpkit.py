#!/usr/bin/env python3
"""
IDP Kit — Intelligent Document Processing Toolkit & AI Agent

Enhanced CLI entry point with support for indexing, querying, Smart Tools, and server mode.

Usage:
    # Index a document
    python run_idpkit.py index --file document.pdf
    python run_idpkit.py index --file document.md --model claude-sonnet-4-20250514

    # Start the web server
    python run_idpkit.py serve --port 8000

    # Show version
    python run_idpkit.py version
"""

import argparse
import asyncio
import json
import os
import sys


def cmd_index(args):
    """Index a document and generate tree structure."""
    from idpkit.engine import page_index_main, config, ConfigLoader
    from idpkit.engine.page_index_md import md_to_tree

    file_path = args.file
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        opt = config(
            model=args.model,
            toc_check_page_num=args.toc_check_pages,
            max_page_num_each_node=args.max_pages_per_node,
            max_token_num_each_node=args.max_tokens_per_node,
            if_add_node_id=args.node_id,
            if_add_node_summary=args.node_summary,
            if_add_doc_description=args.doc_description,
            if_add_node_text=args.node_text,
        )
        result = page_index_main(file_path, opt)

    elif ext in (".md", ".markdown"):
        config_loader = ConfigLoader()
        user_opt = {
            "model": args.model,
            "if_add_node_summary": args.node_summary,
            "if_add_doc_description": args.doc_description,
            "if_add_node_text": args.node_text,
            "if_add_node_id": args.node_id,
        }
        opt = config_loader.load(user_opt)

        result = asyncio.run(
            md_to_tree(
                md_path=file_path,
                if_thinning=args.thinning,
                min_token_threshold=args.thinning_threshold,
                if_add_node_summary=opt.if_add_node_summary,
                summary_token_threshold=args.summary_token_threshold,
                model=opt.model,
                if_add_doc_description=opt.if_add_doc_description,
                if_add_node_text=opt.if_add_node_text,
                if_add_node_id=opt.if_add_node_id,
            )
        )
    else:
        print(f"Error: Unsupported file format: {ext}")
        print("Supported formats: .pdf, .md, .markdown")
        sys.exit(1)

    # Save results
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = args.output or "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_structure.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Tree structure saved to: {output_file}")


def cmd_serve(args):
    """Start the IDP Kit web server."""
    try:
        import uvicorn
        from idpkit.api.app import create_app
    except ImportError as e:
        print(f"Error: Missing dependency for web server: {e}")
        print("Install with: pip install idpkit[all]")
        sys.exit(1)

    app = create_app()
    uvicorn.run(
        app,
        host=args.host or os.getenv("IDP_HOST", "0.0.0.0"),
        port=args.port or int(os.getenv("IDP_PORT", "8000")),
        reload=args.reload,
    )


def cmd_version(args):
    """Show IDP Kit version."""
    from idpkit.version import __version__

    print(f"IDP Kit v{__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="idpkit",
        description="IDP Kit — Intelligent Document Processing Toolkit & AI Agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- index command ---
    index_parser = subparsers.add_parser("index", help="Index a document into a tree structure")
    index_parser.add_argument("--file", "-f", required=True, help="Path to the document file")
    index_parser.add_argument("--model", default="gpt-4o-2024-11-20", help="LLM model to use")
    index_parser.add_argument("--output", "-o", help="Output directory (default: ./results)")
    # PDF options
    index_parser.add_argument("--toc-check-pages", type=int, default=20, help="Pages to check for TOC (PDF only)")
    index_parser.add_argument("--max-pages-per-node", type=int, default=10, help="Max pages per node (PDF only)")
    index_parser.add_argument("--max-tokens-per-node", type=int, default=20000, help="Max tokens per node (PDF only)")
    # Common options
    index_parser.add_argument("--node-id", default="yes", choices=["yes", "no"], help="Add node IDs")
    index_parser.add_argument("--node-summary", default="yes", choices=["yes", "no"], help="Add node summaries")
    index_parser.add_argument("--doc-description", default="no", choices=["yes", "no"], help="Add document description")
    index_parser.add_argument("--node-text", default="no", choices=["yes", "no"], help="Include full text in nodes")
    # Markdown options
    index_parser.add_argument("--thinning", action="store_true", help="Apply tree thinning (Markdown only)")
    index_parser.add_argument("--thinning-threshold", type=int, default=5000, help="Thinning token threshold (Markdown only)")
    index_parser.add_argument("--summary-token-threshold", type=int, default=200, help="Summary token threshold (Markdown only)")
    index_parser.set_defaults(func=cmd_index)

    # --- serve command ---
    serve_parser = subparsers.add_parser("serve", help="Start the IDP Kit web server")
    serve_parser.add_argument("--host", default=None, help="Host to bind to (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    serve_parser.set_defaults(func=cmd_serve)

    # --- version command ---
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

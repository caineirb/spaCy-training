#!/usr/bin/env python3
"""
spaCy NER Evaluation Results Scanner (CLI & Server)
===================================================
A modern, human-readable inspection suite and browser viewer for:
1. Held-Out Test Set evaluations (199 documents).
2. Unseen Benchmark evaluations (Hybrid, Transformer-Only, Entity-Ruler-Only).
3. Runtime Dictionary Overrides audit log.
4. 3-Way Comparative Generalization Matrix (TRTR vs TRSTR-Paraphrase vs TRSTR-LLM).

Usage:
    # Launch interactive browser scanner
    python tools/eval_viewer.py

    # Launch on custom port without opening browser
    python tools/eval_viewer.py --port 8080 --no-browser

    # Build/update standalone offline HTML viewer
    python tools/eval_viewer.py --build-static
"""

import os
import sys
import json
import re
import socket
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results")
DEFAULT_REPORT_FILE = os.path.join(PROJECT_ROOT, "data", "evaluation_report_3way_comparison.json")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web_eval_viewer")
STATIC_HTML_OUTPUT = os.path.join(DEFAULT_EVAL_DIR, "viewer.html")


def find_term_offsets(text: str, term: str, near_offset: int = None):
    """Find start and end character offsets of a term in text."""
    if not term or not text:
        return None, None
    matches = [m.span() for m in re.finditer(re.escape(term), text)]
    if not matches:
        matches = [m.span() for m in re.finditer(re.escape(term), text, re.IGNORECASE)]
    if not matches:
        return None, None
    if near_offset is not None and len(matches) > 1:
        matches.sort(key=lambda s: abs(s[0] - near_offset))
    return matches[0]


def load_eval_data(eval_dir: str = DEFAULT_EVAL_DIR, report_file: str = DEFAULT_REPORT_FILE) -> dict:
    """Scan and parse all evaluation result files and format them for the web viewer."""
    data = {
        "models": ["trtr", "trstr_paraphrase", "trstr_llm"],
        "file_types": [
            {
                "id": "held_out_test",
                "label": "Held-Out Test Set",
                "description": "199 authentic held-out test documents annotated with IT_TERM and CLERICAL_TERM",
                "icon": "📄"
            },
            {
                "id": "unseen_benchmark_hybrid",
                "label": "Unseen Benchmark (Hybrid)",
                "description": "85 modern tech stack sentences evaluated via Hybrid Pipeline (EntityRuler + Transformer)",
                "icon": "⚡"
            },
            {
                "id": "unseen_benchmark_transformer_only",
                "label": "Unseen Benchmark (Transformer Only)",
                "description": "85 modern tech stack sentences evaluated purely with ML Transformer NER",
                "icon": "🤖"
            },
            {
                "id": "unseen_benchmark_entity_ruler_only",
                "label": "Unseen Benchmark (Entity Ruler Only)",
                "description": "85 modern tech stack sentences evaluated purely with Dictionary EntityRuler",
                "icon": "📖"
            }
        ],
        "datasets": {},
        "dictionary_overrides": [],
        "report_summary": None
    }

    # 1. Load 3-way evaluation report if present
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as rf:
                data["report_summary"] = json.load(rf)
        except Exception as e:
            print(f"[WARN] Failed to load {report_file}: {e}")

    # 2. Load dictionary overrides
    dict_file = os.path.join(eval_dir, "dictionary_overrides.jsonl")
    if os.path.exists(dict_file):
        try:
            with open(dict_file, "r", encoding="utf-8") as df:
                for line in df:
                    line = line.strip()
                    if line:
                        data["dictionary_overrides"].append(json.loads(line))
        except Exception as e:
            print(f"[WARN] Failed to load dictionary overrides: {e}")

    # 3. Load comparison datasets
    for ft in data["file_types"]:
        ft_id = ft["id"]
        data["datasets"][ft_id] = {}

        for m in data["models"]:
            full_path = os.path.join(eval_dir, "comparison", m, f"{ft_id}_{m}.jsonl")
            err_path = os.path.join(eval_dir, "comparison", m, f"{ft_id}_{m}_errors_only.jsonl")

            all_records = []
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        # Enrich offsets for predictions
                        _enrich_record_offsets(row)
                        all_records.append(row)

            err_records = []
            if os.path.exists(err_path):
                with open(err_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        _enrich_record_offsets(row)
                        err_records.append(row)

            data["datasets"][ft_id][m] = {
                "all": all_records,
                "errors_only": err_records,
                "counts": {
                    "total": len(all_records),
                    "errors": len(err_records)
                }
            }

    return data


def _enrich_record_offsets(row: dict):
    """Enrich prediction spans in entity_results with precise text start/end offsets."""
    text = row.get("text", "")
    for er in row.get("entity_results", []):
        if er.get("pred_term") and ("pred_start" not in er or er["pred_start"] is None):
            near = er.get("gold_start")
            s, e = find_term_offsets(text, er["pred_term"], near_offset=near)
            er["pred_start"] = s
            er["pred_end"] = e


def build_standalone_html(eval_dir: str = DEFAULT_EVAL_DIR, output_path: str = STATIC_HTML_OUTPUT):
    """Compile a self-contained single-file HTML bundle with inlined data, CSS, and JS."""
    print(f"[*] Compiling standalone offline viewer: {output_path}")

    # Load data
    data = load_eval_data(eval_dir)
    data_json = json.dumps(data, ensure_ascii=False)

    # Read web assets
    with open(os.path.join(WEB_DIR, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()

    with open(os.path.join(WEB_DIR, "style.css"), "r", encoding="utf-8") as f:
        css = f.read()

    with open(os.path.join(WEB_DIR, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    # Inline CSS
    html = re.sub(
        r'<link\s+rel="stylesheet"\s+href="style\.css">',
        f"<style>\n{css}\n</style>",
        html
    )

    # Inline Data & JS
    embedded_script = f"""
<script>
window.EVAL_DATA = {data_json};
</script>
<script>
{js}
</script>
"""
    html = re.sub(
        r'<script\s+src="app\.js"></script>',
        embedded_script,
        html
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[✓] Standalone HTML bundle compiled successfully ({size_mb:.2f} MB)")
    print(f"    Direct open: file://{output_path}")
    return output_path


class EvalServerHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving evaluation web assets and API endpoints."""

    def __init__(self, *args, eval_dir=DEFAULT_EVAL_DIR, **kwargs):
        self.eval_dir = eval_dir
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = load_eval_data(self.eval_dir)
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if parsed.path == "/":
            self.path = "/index.html"

        super().do_GET()

    def log_message(self, format, *args):
        # Mute noisy static request logs, only show API or main hits
        if "/api/" in (args[0] if args else "") or "index.html" in (args[0] if args else ""):
            super().log_message(format, *args)


def find_free_port(start_port: int = 8765) -> int:
    """Find next available free port starting at start_port."""
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, eval_dir: str = DEFAULT_EVAL_DIR):
    """Run local HTTP server and launch browser."""
    actual_port = find_free_port(port)

    # Pre-build standalone HTML as well so both are in sync
    try:
        build_standalone_html(eval_dir)
    except Exception as e:
        print(f"[WARN] Failed to pre-build standalone bundle: {e}")

    handler_factory = lambda *args, **kwargs: EvalServerHandler(*args, eval_dir=eval_dir, **kwargs)
    server = HTTPServer((host, actual_port), handler_factory)

    url = f"http://{host}:{actual_port}/"
    print("\n" + "=" * 68)
    print("  spaCy NER Evaluation Results Scanner")
    print("=" * 68)
    print(f"  Server URL      : {url}")
    print(f"  Offline Viewer  : file://{STATIC_HTML_OUTPUT}")
    print(f"  Data Source     : {eval_dir}")
    print("=" * 68)
    print("  Press Ctrl+C to stop the server.\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down evaluation server...")
        server.server_close()
        print("[✓] Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="spaCy NER Evaluation Scanner & Browser Tool")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--build-static", action="store_true", help="Compile standalone viewer.html and exit")
    parser.add_argument("--eval-dir", type=str, default=DEFAULT_EVAL_DIR, help="Path to evaluation results dir")

    args = parser.parse_args()

    if args.build_static:
        build_standalone_html(args.eval_dir)
        sys.exit(0)

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser, eval_dir=args.eval_dir)


if __name__ == "__main__":
    main()

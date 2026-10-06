"""Command-line entry point for live research, saved sessions, and the offline demo."""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv


def run_graph(app, inputs, config):
    """Show progress on stderr and return the fully checkpointed state."""
    for event in app.stream(inputs, config=config):
        for node, update in event.items():
            if node == "planner":
                print(f"Plan: {len(update['plan'])} research steps", file=sys.stderr)
            elif node == "researcher":
                print(f"Research step {update['current_step']} complete", file=sys.stderr)
    return app.get_state(config).values


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research Assistant Agent (LangGraph)")
    parser.add_argument("--topic", help="Research a topic once and exit")
    parser.add_argument("--interactive", action="store_true", help="Research multiple topics")
    parser.add_argument("--thread-id", "--thread_id", dest="thread_id", help="Saved session ID")
    parser.add_argument("--resume", action="store_true", help="Resume work or display a saved report")
    parser.add_argument("--db", type=Path, default=Path("checkpoints.sqlite"), help="SQLite checkpoint file")
    parser.add_argument("--output", type=Path, help="Save the latest report as Markdown")
    parser.add_argument("--max-steps", type=int, default=3, help="Research queries per topic (1–10)")
    parser.add_argument("--demo", action="store_true", help="Run the fixed offline demo without API keys")
    parser.add_argument("--verbose", action="store_true", help="Include diagnostic logs on stderr")
    parser.add_argument("--version", action="version", version="0.2.0")
    args = parser.parse_args(argv)
    if args.topic is not None and not args.topic.strip():
        parser.error("The research topic cannot be empty.")
    if not 1 <= args.max_steps <= 10:
        parser.error("--max-steps must be between 1 and 10.")
    if args.resume and (not args.thread_id or args.topic):
        parser.error("--resume requires --thread-id and cannot be combined with --topic.")
    interactive = args.interactive or (not args.topic and not args.demo and not args.resume)
    if interactive and not sys.stdin.isatty() and not args.interactive:
        parser.error("Provide --topic, --demo, or --interactive.")

    load_dotenv(Path.cwd() / ".env")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )
    credentials = bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    if not args.demo and not args.resume and not credentials:
        print("Set OPENAI_API_KEY or GOOGLE_API_KEY in .env, or use --demo.", file=sys.stderr)
        return 1
    thread_id = args.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    print(f"Session: {thread_id}", file=sys.stderr)
    if args.demo:
        print("Offline demo: fixed illustrative evidence; no network requests.", file=sys.stderr)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        from graph import create_graph

        args.db = args.db.expanduser()
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(args.db)) as saver:
            app = create_graph(checkpointer=saver, demo=args.demo)
            state = None
            if args.resume:
                snapshot = app.get_state(config)
                if not snapshot.values:
                    raise ValueError(f"No saved session with ID '{thread_id}' in {args.db}.")
                if bool(snapshot.values.get("demo")) != args.demo:
                    raise ValueError("Saved session mode differs; add or remove --demo to match it.")
                if snapshot.next:
                    if not args.demo and not credentials:
                        raise ValueError("Set OPENAI_API_KEY or GOOGLE_API_KEY to resume live research.")
                    state = run_graph(app, None, config)
                else:
                    print("Saved report:", file=sys.stderr)
                    state = snapshot.values
            topic = args.topic
            if args.demo and not args.resume and not topic:
                from demo import DEMO_TOPIC

                topic = DEMO_TOPIC
            while True:
                if state is None:
                    if topic is None:
                        topic = input("\nResearch topic (quit to exit): ").strip()
                    if topic.lower() in ("quit", "exit"):
                        return 0
                    if not topic:
                        topic = None
                        continue
                    if not args.demo and not credentials:
                        raise ValueError("Set OPENAI_API_KEY or GOOGLE_API_KEY before live research.")
                    state = run_graph(
                        app, {"topic": topic, "max_steps": args.max_steps, "demo": args.demo}, config
                    )
                report = state.get("answer", "")
                if not report:
                    raise RuntimeError("This session has no completed report.")
                print(report)
                if args.output:
                    output = args.output.expanduser()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(report.rstrip() + "\n", encoding="utf-8")
                    print(f"Report saved to {output}", file=sys.stderr)
                if not interactive:
                    return 0 if state.get("findings") else 1
                state, topic = None, None
    except (KeyboardInterrupt, EOFError):
        print("\nSession saved. Goodbye!", file=sys.stderr)
        return 130 if sys.exc_info()[0] is KeyboardInterrupt else 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            logging.exception("Research run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

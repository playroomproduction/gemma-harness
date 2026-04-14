#!/usr/bin/env python3
"""
Gemma Harness CLI — interactive terminal client.

Usage:
  python cli.py                  # Interactive chat mode
  python cli.py "your question"  # Single-shot mode
  python cli.py --health         # Check server health
"""

from __future__ import annotations

import json
import sys
import time

import httpx

import config

HARNESS_URL = f"http://{config.HARNESS_HOST}:{config.HARNESS_PORT}"


def health_check() -> None:
    """Check harness and MLX server health."""
    try:
        resp = httpx.get(f"{HARNESS_URL}/health", timeout=5.0)
        data = resp.json()
        print(f"Harness:   {'✅' if data.get('ok') else '❌'}")
        print(f"MLX:       {'✅' if data.get('mlx_healthy') else '❌'}")
        print(f"Model:     {data.get('model', 'unknown')}")
        print(f"Tools:     {data.get('tools_registered', 0)}")
    except httpx.ConnectError:
        print(f"❌ Cannot connect to harness at {HARNESS_URL}")
        print("   Run: python server.py")
        sys.exit(1)


def chat_once(message: str, conversation: list | None = None) -> dict:
    """Send a chat request to the harness."""
    payload = {"message": message}
    if conversation:
        payload["conversation"] = conversation

    try:
        resp = httpx.post(
            f"{HARNESS_URL}/chat",
            json=payload,
            timeout=180.0,
        )
        return resp.json()
    except httpx.ConnectError:
        return {"ok": False, "error": f"Cannot connect to harness at {HARNESS_URL}"}
    except httpx.TimeoutException:
        return {"ok": False, "error": "Request timed out"}


def interactive_mode() -> None:
    """Run in interactive chat mode."""
    print("╔══════════════════════════════════════════════╗")
    print("║  Gemma Harness — Claude-style Agent CLI      ║")
    print("║  Type 'exit' or Ctrl+C to quit               ║")
    print("║  Type '/tools' to list tools                  ║")
    print("║  Type '/memory' to list memories              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    conversation: list[dict] = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Bye!")
            break

        if user_input == "/tools":
            try:
                resp = httpx.get(f"{HARNESS_URL}/tools", timeout=5.0)
                data = resp.json()
                for t in data.get("tools", []):
                    params = ", ".join(
                        f"{p['name']}: {p['type']}" for p in t.get("parameters", [])
                    )
                    print(f"  🔧 {t['name']}({params})")
                    print(f"     {t['description'][:80]}")
            except Exception as exc:
                print(f"  ❌ {exc}")
            continue

        if user_input == "/memory":
            try:
                resp = httpx.get(f"{HARNESS_URL}/memory", timeout=5.0)
                data = resp.json()
                if data.get("memories"):
                    for m in data["memories"]:
                        print(f"  📝 {m['key']}: {m['content'][:60]}...")
                else:
                    print("  (no memories stored)")
            except Exception as exc:
                print(f"  ❌ {exc}")
            continue

        # Send to harness
        print("Gemma > ", end="", flush=True)
        start = time.time()

        result = chat_once(user_input, conversation)

        if not result.get("ok"):
            print(f"❌ {result.get('error', 'Unknown error')}")
            continue

        content = result.get("content", "")
        print(content)

        elapsed = result.get("latency_seconds", time.time() - start)
        tools_used = result.get("tool_calls_made", 0)
        rounds = result.get("rounds_used", 0)
        meta_parts = [f"{elapsed:.1f}s"]
        if tools_used:
            meta_parts.append(f"{tools_used} tools")
        if rounds > 1:
            meta_parts.append(f"{rounds} rounds")
        print(f"\n  ⏱️  {' | '.join(meta_parts)}")
        print()

        # Update conversation history
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": content})


def main():
    if "--health" in sys.argv:
        health_check()
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        # Single-shot mode
        message = " ".join(args)
        result = chat_once(message)
        if result.get("ok"):
            print(result.get("content", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown')}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()

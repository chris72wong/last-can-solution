"""Read-only rosapi topic discovery, adapted from the instructor utility.

python tools/list_topics.py [filter] [--echo /topic] [--config path]
Never advertises or publishes a movement topic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lastcan.config import ConfigStore


def call_service(ws, service, args=None, timeout=5.0):
    import websocket
    call_id = f"call:{service}:{time.monotonic()}"
    ws.send(json.dumps({"op": "call_service", "id": call_id, "service": service, "args": args or {}}))
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        try:
            reply = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        if reply.get("op") == "service_response" and reply.get("id") == call_id:
            return reply.get("values") if reply.get("result", True) else None
    return None


def main():
    import websocket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="")
    parser.add_argument("--echo")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    robot = ConfigStore(args.config).data["robot"]
    try:
        ws = websocket.create_connection(f"ws://{robot['ip']}:{robot['rosbridge_port']}", timeout=3)
    except Exception as exc:
        raise SystemExit(f"Cannot connect to robot: {exc}") from exc
    ws.settimeout(.5)
    try:
        if args.echo:
            found = call_service(ws, "/rosapi/topic_type", {"topic": args.echo}) or {}
            if not found.get("type"):
                raise SystemExit("Topic not found, or rosapi unavailable")
            ws.send(json.dumps({"op": "subscribe", "topic": args.echo, "type": found["type"]}))
            while True:
                try:
                    reply = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if reply.get("op") == "publish" and reply.get("topic") == args.echo:
                    print(json.dumps(reply.get("msg"))[:400])
        else:
            found = call_service(ws, "/rosapi/topics")
            if not found:
                raise SystemExit("No rosapi response; ask TA or use ros2 topic list -t over SSH")
            for topic, message_type in sorted(zip(found.get("topics", []), found.get("types", []))):
                if args.filter.lower() in topic.lower():
                    print(f"{topic:<58} {message_type}")
    except KeyboardInterrupt:
        pass
    finally:
        ws.close()


if __name__ == "__main__":
    main()

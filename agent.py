#!/usr/bin/env python3
"""
Bike Safety Analysis Agent

A tool-calling agent powered by NVIDIA Nemotron-3-Nano (via Ollama)
that answers natural language questions about bike tracking videos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import ollama

from src.video_processor import VideoProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "llama3.1:8b"
REPORTS_DIR = Path("output/reports")
OUTPUT_DIR = Path("output")
CONFIG_PATH = "config.yaml"

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a bike safety analysis agent. You answer questions about "
        "cycling videos that have been analyzed for safety compliance.\n\n"
        "RULES:\n"
        "1. ALWAYS call list_processed_videos() first to see which videos exist.\n"
        "2. The video_name parameter must be an EXACT name returned by "
        "list_processed_videos (like 'nightbike' or 'nolight'). NEVER guess names.\n"
        "3. Use the right tool to answer the question, then respond in plain English.\n"
        "4. If asked about 'all' or 'worst', query each video and compare."
    ),
}


def _build_seed_messages() -> list:
    """Seed the conversation with an initial list_processed_videos call.

    This teaches the model the correct workflow by example and gives it
    the actual video names upfront.
    """
    video_list = list_processed_videos()
    return [
        SYSTEM_PROMPT,
        {"role": "user", "content": "What videos are available?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "list_processed_videos", "arguments": {}}}
        ]},
        {"role": "tool", "content": video_list},
        {"role": "assistant", "content": f"Here are the processed videos:\n{video_list}\n\nWhat would you like to know about them?"},
    ]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_report(video_name: str) -> Optional[dict]:
    """Load a JSON report by video name (without extension)."""
    video_name = video_name.replace(".mp4", "").replace("_report", "")
    report_path = REPORTS_DIR / f"{video_name}_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Tools — each is a plain function with type hints + docstring.
# Ollama auto-generates the JSON schema from these.
# ---------------------------------------------------------------------------

def process_video(video_path: str) -> str:
    """Process a video file through the bike detection pipeline including YOLO detection, light detection, speed estimation, depth estimation, and heatmap generation. Returns a summary of results."""
    if not os.path.exists(video_path):
        return f"Error: Video file not found at '{video_path}'"
    try:
        processor = VideoProcessor(config_path=CONFIG_PATH)
        results = processor.process_video(video_path, output_dir=str(OUTPUT_DIR))
        summary = results["report"]["summary"]
        video_name = Path(video_path).stem
        return (
            f"Video processed successfully: {video_path}\n"
            f"Summary:\n{json.dumps(summary, indent=2)}\n"
            f"Outputs saved to: {OUTPUT_DIR}/\n"
            f"Use get_report_summary('{video_name}') for details."
        )
    except Exception as e:
        return f"Error processing video: {e}"


def list_processed_videos() -> str:
    """List all videos that have been processed and have reports available."""
    if not REPORTS_DIR.exists():
        return "No reports directory found. No videos have been processed yet."
    reports = sorted(REPORTS_DIR.glob("*_report.json"))
    if not reports:
        return "No processed videos found. Use process_video() to analyze a video."
    lines = []
    for r in reports:
        video_name = r.stem.replace("_report", "")
        with open(r) as f:
            data = json.load(f)
        info = data.get("video_info", {})
        summary = data.get("summary", {})
        description = data.get("description", "")
        desc_part = f" — {description}" if description else ""
        lines.append(
            f"- {video_name}: {info.get('total_frames', '?')} frames, "
            f"{info.get('duration_seconds', '?'):.1f}s, "
            f"{summary.get('total_bikes_detected', '?')} bikes detected, "
            f"{summary.get('compliance_rate', '?')}% compliance{desc_part}"
        )
    return "Processed videos:\n" + "\n".join(lines)


def get_report_summary(video_name: str) -> str:
    """Get the summary statistics for a processed video including bike counts, light compliance rate, and speed data."""
    report = _load_report(video_name)
    if not report:
        return f"No report found for '{video_name}'. Use list_processed_videos() to see available videos."
    summary = report["summary"]
    info = report["video_info"]
    description = report.get("description", "")
    result = (
        f"Video: {info['filename']}\n"
    )
    if description:
        result += f"Description: {description}\n"
    result += (
        f"Duration: {info['duration_seconds']:.1f}s | FPS: {info['fps']} | "
        f"Resolution: {info['resolution'][0]}x{info['resolution'][1]}\n"
        f"Total bikes: {summary['total_bikes_detected']}\n"
        f"With front lights: {summary['bikes_with_front_lights']}\n"
        f"With rear lights: {summary['bikes_with_rear_lights']}\n"
        f"With both lights: {summary['bikes_with_both_lights']}\n"
        f"No lights: {summary['bikes_with_no_lights']}\n"
        f"Compliance rate: {summary['compliance_rate']}%\n"
    )
    if 'avg_speed_all_bikes_kmh' in summary:
        result += (
            f"Avg speed: {summary['avg_speed_all_bikes_kmh']} km/h\n"
            f"Max speed: {summary['max_speed_any_bike_kmh']} km/h"
        )
    return result


def get_bike_details(video_name: str, track_id: int) -> str:
    """Get full details for a specific bike by its track ID, including speed, depth, lights, and color information."""
    report = _load_report(video_name)
    if not report:
        return f"No report found for '{video_name}'."
    bikes = report.get("bikes", [])
    for bike in bikes:
        if bike["track_id"] == track_id:
            return json.dumps(bike, indent=2)
    available = [b["track_id"] for b in bikes]
    return f"Bike #{track_id} not found. Available track IDs: {available}"


def get_violations(video_name: str) -> str:
    """List all light compliance violations for a processed video. Shows bikes missing front or rear lights."""
    report = _load_report(video_name)
    if not report:
        return f"No report found for '{video_name}'."
    violations = report.get("violations", [])
    if not violations:
        return f"No violations found for '{video_name}'. All bikes are compliant!"
    lines = []
    for v in violations:
        lines.append(
            f"- Bike #{v['track_id']}: {v['violation_type']} (severity: {v['severity']})\n"
            f"  Missing: {', '.join(v['missing_lights'])}\n"
            f"  Visible: {v['time_in_video']['first_seen_seconds']}s - {v['time_in_video']['last_seen_seconds']}s\n"
            f"  Color: {v['bike_details'].get('color', 'unknown')}"
        )
    return f"Violations ({len(violations)} found):\n" + "\n".join(lines)


def list_bikes(video_name: str) -> str:
    """List all tracked bikes in a processed video with their basic information."""
    report = _load_report(video_name)
    if not report:
        return f"No report found for '{video_name}'."
    bikes = report.get("bikes", [])
    if not bikes:
        return f"No bikes detected in '{video_name}'."
    lines = []
    for b in bikes:
        color = b.get("color", {}).get("primary_color", "unknown")
        lights = b.get("lights", {})
        compliance = lights.get("compliance_status", "UNKNOWN")
        speed = b.get("speed", {})
        avg_speed = speed.get("avg_speed_kmh", "N/A")
        lines.append(
            f"- Bike #{b['track_id']}: color={color}, "
            f"lights={compliance}, "
            f"avg_speed={avg_speed}km/h, "
            f"visible={b.get('duration_seconds', '?')}s"
        )
    return f"Bikes ({len(bikes)} total):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    process_video,
    list_processed_videos,
    get_report_summary,
    get_bike_details,
    get_violations,
    list_bikes,
]

TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_chat(user_message: str, messages: list) -> str:
    """
    Run one turn of the agent loop. Handles multi-step tool calls.

    Args:
        user_message: The user's input text
        messages: Conversation history (mutated in place)

    Returns:
        The agent's final text response
    """
    messages.append({"role": "user", "content": user_message})

    while True:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        # Add assistant message to history
        messages.append(response.message)

        # If no tool calls, return the text response
        if not response.message.tool_calls:
            return response.message.content or ""

        # Execute each tool call
        for tool_call in response.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            fn = TOOL_MAP.get(fn_name)
            if fn:
                print(f"  [Tool: {fn_name}({fn_args})]")
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = f"Error calling {fn_name}: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            messages.append({"role": "tool", "content": str(result)})


def main():
    """Run the CLI chat loop."""
    print("=" * 60)
    print("Bike Safety Analysis Agent")
    print(f"Model: {MODEL} (via Ollama)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    messages = _build_seed_messages()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = agent_chat(user_input, messages)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()

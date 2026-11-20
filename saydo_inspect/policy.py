"""The model under test as an Inspect Robots policy: SayDo's ablation-7/4 conversation.

One conversation per trial, as sim/episode.py holds it: the system prompt (the deployment's own,
which the embodiment publishes as `EmbodimentInfo.docs`), the skills as native tool declarations
(contributed by the embodiment, `bind_tools`), the first operator command with the camera feed,
then after each model turn one tool message per call and a camera turn with the new frames
(ablation 4: the sensor log in their place). Images are kept for the last `image_steps` camera
turns and replaced by a note before that. The model's reasoning is not passed back.

Each `act()` is one model turn. Its tool calls come back as one chunk, one action per call; a
reply with no tool call is the robot's report to the operator, which ends the job (an action
with `reply`). Later commands arrive as operator messages (`ShiftOperator`).

Model calls go through ladder/run.py's `chat` (OpenRouter, or Anthropic's API for the models it
lists), with episode.py's cut-off retries: a turn cut off or failed without a tool call is asked
again, up to CUTOFF_RETRIES times, a cut-off one with twice the tokens. Still cut off, the policy
raises PolicyError: Inspect Robots records the trial and does not score it, as aggregation leaves
episode.py's `model_cutoff` episodes out.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots.errors import PolicyError
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ObservationSpace
from inspect_robots.types import Action, ActionChunk, Observation

from .embodiment import ACTION_SPACE, NO_MOTION

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ladder"))

# episode.py's wording, checked against it at start-up (run.py: check_wording).
CAMERAS = ("wrist", "overview")
CAMERA_FEED = ("Camera feed: " + ", then ".join(f"{c} camera" for c in CAMERAS) + ". "
               "Each object is marked with its number.")
EARLIER_IMAGE = "[earlier camera image]"
CUTOFF_RETRIES = 3
MAX_TOKENS_CAP = 64000


class SayDoPolicy(PolicyBase):
    """A model behind ladder/run.py's chat(), or `replay` of a logged episode's turns."""

    def __init__(self, model: str = "deepseek/deepseek-v4.1-flash", *, temperature: float = 1.0,
                 max_tokens: int | None = None, reasoning_effort: str | None = None,
                 image_steps: int = 3, replay: list | None = None) -> None:
        self.model, self.temperature, self.image_steps = model, temperature, image_steps
        self.max_tokens = max_tokens or (32000 if "deepseek" in model.lower() else 16000)
        self.replay = replay
        self.info = PolicyInfo(name=f"saydo-agent/{model}" if replay is None else "saydo-replay",
                               action_space=ACTION_SPACE, observation_space=ObservationSpace())
        self.config = PolicyConfig(action_horizon=1, temperature=temperature)
        self.system: str | None = None
        self.tools: list[dict] = []
        self.client = None
        if replay is None:
            from run import REASONING_EFFORT, make_client  # ladder/
            self.client = make_client()
            self.reasoning_effort = reasoning_effort or REASONING_EFFORT

    # -- binding -----------------------------------------------------------------------------

    def bind(self, embodiment_info) -> None:
        self.system = embodiment_info.docs

    def bind_tools(self, library) -> None:
        """The skills the embodiment declares, in its order: episode.py's tool declarations."""
        self.tools = [json.loads(json.dumps(s)) for s in library.schemas()]

    # -- the trial ---------------------------------------------------------------------------

    def reset(self, scene: Scene) -> None:
        self.conversation: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        self.images: dict[str, str] = {}  # image key -> data URL, for the turns that keep theirs
        self.turn = 0
        self.pending: list[str] = []  # call ids of the last turn, awaiting their results
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        self.logged: list[dict[str, Any]] = []
        self._delta_at = 0

    def act(self, observation: Observation) -> ActionChunk:
        extra = observation.extra
        # The last turn's tool results, then this camera turn (with the operator's command, if any).
        results = {r["tool_call_id"]: r["content"] for r in extra.get("tool_results", [])}
        for cid in self.pending:
            self.conversation.append({"role": "tool", "tool_call_id": cid, "content": results.get(cid, "skipped")})
        self.pending = []
        messages = [m for m in extra.get("operator_messages", []) if isinstance(m, dict)]
        if self.turn == 0:
            command = observation.instruction
        else:
            command = "\n\n".join(m["text"] for m in messages) or None
        self.conversation.append(self._camera_turn(observation, command))
        text, meta = self._ask()
        calls = meta.get("tool_calls") or []
        self.conversation.append(_assistant_turn(text, calls))
        said = self._said(text, meta)
        turn = self.turn
        self.turn += 1
        if not calls:  # the reply that ends a job
            return self._chunk([{"turn": turn, "reply": text, "said": said}], meta)
        self.pending = [c["id"] for c in calls]
        actions = []
        for i, c in enumerate(calls):
            actions.append({"turn": turn, "call": c, "first": i == 0, "last": i == len(calls) - 1,
                            **({"said": said} if i == 0 else {})})
        return self._chunk(actions, meta)

    # -- helpers -----------------------------------------------------------------------------

    def _chunk(self, metas, meta) -> ActionChunk:
        return ActionChunk(actions=[Action(data=NO_MOTION.copy(), meta=m) for m in metas],
                           inference_latency_s=meta.get("latency_s"))

    def _said(self, text, meta):
        return {"response": text, "reasoning": meta.get("reasoning"), "usage": meta.get("usage"),
                "finish_reason": meta.get("finish_reason"), **({"retries": meta["retries"]} if meta.get("retries") else {}),
                **({"provider_block": meta["provider_block"]} if meta.get("provider_block") else {})}

    def _camera_turn(self, obs: Observation, command: str | None) -> dict:
        if "sensor_log" in obs.extra:  # ablation 4
            parts = [{"type": "text", "text": obs.extra["sensor_log"]}]
        else:
            parts = [{"type": "text", "text": CAMERA_FEED}]
            for cam in CAMERAS:
                key = f"t{self.turn}_{cam}"
                self.images[key] = _data_url(obs.images[cam])
                parts.append({"type": "image", "key": key})
        if command:
            parts.append({"type": "text", "text": command})
        return {"role": "user", "content": parts}

    def _image_turns(self):
        return [i for i, m in enumerate(self.conversation)
                if isinstance(m["content"], list) and any(p["type"] == "image" for p in m["content"])]

    def _as_sent(self):
        """episode.py's as_sent: images kept for the last image_steps camera turns."""
        with_images = self._image_turns()
        keep = set(with_images[-self.image_steps:] if self.image_steps else with_images)
        sent = []
        for i, m in enumerate(self.conversation):
            if not isinstance(m["content"], list):
                sent.append(m)
                continue
            parts = []
            for p in m["content"]:
                if p["type"] != "image":
                    parts.append(p)
                elif i in keep:
                    parts.append({"type": "image_url", "image_url": {"url": self.images[p["key"]]}})
                else:
                    parts.append({"type": "text", "text": EARLIER_IMAGE})
            sent.append({**m, "content": parts})
        return sent

    def _next_change(self):
        """episode.py's next_change: where the next call first differs, for prompt caching."""
        with_images = self._image_turns()
        if not self.image_steps or len(with_images) < self.image_steps:
            return None
        return with_images[-self.image_steps]

    def _ask(self, tool_choice=None):
        if self.replay is not None:
            return self._replayed(tool_choice)
        from run import chat  # ladder/
        sent = self._as_sent()
        tokens = self.max_tokens
        cache = self._next_change() if tool_choice is None else (self._next_change() or len(self.conversation) - 1)
        for retry in range(CUTOFF_RETRIES + 1):
            t0 = time.monotonic()
            text, meta = chat(self.client, self.model, sent, self.temperature, tokens, tools=self.tools,
                              tool_choice=tool_choice, reasoning_effort=self.reasoning_effort, cache_before=cache)
            meta["latency_s"] = time.monotonic() - t0
            meta["retries"], meta["max_tokens"] = retry, tokens
            self._account(meta)
            if tool_choice or meta.get("tool_calls") or meta.get("finish_reason") not in ("length", "error"):
                return text, meta
            print(f"  model turn {meta.get('finish_reason')} at {tokens} tokens; asking again", flush=True)
            if meta.get("finish_reason") == "length":
                tokens = min(2 * tokens, MAX_TOKENS_CAP)
        raise PolicyError(f"model_cutoff: turn {self.turn} ended {meta.get('finish_reason')} with no tool call "
                          f"after {CUTOFF_RETRIES} retries")

    def _replayed(self, tool_choice):
        if self.turn >= len(self.replay):
            return "", {"tool_calls": []}  # the replay ran out: an empty reply ends the job
        text, meta = self.replay[self.turn]
        return text, dict(meta)

    def _account(self, meta):
        u = meta.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens"):
            self.usage_total[k] += u.get(k) or 0
        self.usage_total["cost"] += u.get("cost") or 0.0

    # -- the record --------------------------------------------------------------------------

    def _logged(self):
        out = []
        for m in self.conversation:
            if isinstance(m.get("content"), list):
                m = {**m, "content": [p if p["type"] != "image" else {"type": "image", "frame": p["key"]}
                                      for p in m["content"]]}
            out.append(m)
        return out

    def transcript(self):
        return {"model": self.model, "usage": dict(self.usage_total), "messages": self._logged()}

    def transcript_delta(self):
        logged = self._logged()
        new, self._delta_at = logged[self._delta_at:], len(logged)
        return new or None

    def on_trial_end(self, record, log_dir, run_id) -> None:
        record.metadata["llm_usage"] = dict(self.usage_total)


def _assistant_turn(text, tool_calls):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}} for c in tool_calls]
    return msg


def _data_url(image) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def replay_turns(path):
    """A logged episode's model turns, as episode.py's --replay reads them."""
    episode = sys.modules.get("saydo_episode")
    if episode is None:
        raise RuntimeError("load the embodiment (it imports sim/episode.py) before reading a replay")
    return episode.replay_turns(path)

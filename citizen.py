# -*- coding: utf-8 -*-
import json
import logging

import mesa

from prompt import system_persona, evaluate_prompt
from utils import get_completion_from_router_json

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Citizen(mesa.Agent):

    def __init__(self, model, unique_id, persona, model_string):
        super().__init__(model)
        self.unique_id = unique_id

        # Person (replicated in all five models: same persona_id)
        self.persona_id = persona["persona_id"]
        self.name = persona["name"]
        self.age = persona["age"]
        self.qualification = persona["qualification"]
        self.qualification_ord = persona["qualification_ord"]
        self.big5 = persona["big5"]
        self.big5_prompt = persona["big5_prompt"]

        self.model_string = model_string
        self.study_history = []

        # System message built once
        self.system_msg = system_persona.format(
            agent_name=self.name,
            agent_age=self.age,
            agent_qualification=self.qualification,
            big5_block=self.big5_prompt,
        )

    def headline_evaluation(self, headline):
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": evaluate_prompt.format(headline=headline)},
        ]

        raw, usage = get_completion_from_router_json(messages, self.model_string)
        if raw is None:
            raw = ""
        if raw.strip().startswith("```"):
            raw = raw.strip().strip("`").replace("json", "", 1).strip()

        #unparsable answers are dropped and set with none
        belief, reasoning, confidence = None, "unparsable", None
        try:
            out = json.loads(raw)
            ev = str(out.get("evaluation", "")).strip().upper()
            if ev.startswith("TRUE"):
                belief, reasoning = "TRUE", out.get("reasoning", "")
            elif ev.startswith("FALSE"):
                belief, reasoning = "FALSE", out.get("reasoning", "")
            c = out.get("confidence_true", None)
            if c is not None:
                try:
                    confidence = max(0, min(100, int(round(float(c)))))
                except (TypeError, ValueError):
                    confidence = None
        except Exception:
            up = raw.upper()
            if "TRUE" in up and "FALSE" not in up:
                belief, reasoning = "TRUE", raw[:200]
            elif "FALSE" in up and "TRUE" not in up:
                belief, reasoning = "FALSE", raw[:200]

        self.belief = belief
        self.reasoning = reasoning
        self.confidence = confidence

        record = {
            "agent_id": self.unique_id,
            "persona_id": self.persona_id,
            "name": self.name,
            "age": self.age,
            "qualification": self.qualification,
            "qualification_ord": self.qualification_ord,
            "model": self.model_string,
            "step": self.model.current_step,
            "headline": headline,
            "belief": belief,
            "confidence_true": confidence,
            "reasoning": reasoning,
            "provider": usage["provider"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
        }
        record.update(self.big5) 
        self.study_history.append(record)

    def step(self):
        if self.model.current_headline:
            self.headline_evaluation(self.model.current_headline)

"""Pydantic models for LLM-as-judge structured outputs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

ValueYesNo = Literal["Yes", "No"]
ValueYesNoNA = Literal["Yes", "No", "N/A"]


class YesNoCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)
    value: ValueYesNo


class YesNoNACriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)
    value: ValueYesNoNA


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape_outlines_are_closed: YesNoCriterion
    angle_labels_matches_arcs: YesNoNACriterion
    labeled_lengths_areas_match_proportions: YesNoNACriterion
    core_mathematical_properties_of_shapes_correct: YesNoCriterion
    diagram_fully_in_canvas: YesNoCriterion
    diagram_elements_are_readable_size: YesNoCriterion
    labels_associated_with_elements: YesNoNACriterion
    diagram_elements_dont_problematically_overlap: YesNoCriterion


STRICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "JudgeRubric",
        "schema": JudgeOutput.model_json_schema(),
        "strict": True,
    },
}

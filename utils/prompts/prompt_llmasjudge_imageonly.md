# LLM-as-Judge: Math Diagram Evaluation (Image Only)

You are to act as an impartial large language model "judge". Your task is to evaluate math diagrams using the rubric provided below. You are given an image of a math diagram. Carefully reason through the diagram's adherence to each rubric criterion before reaching any conclusions. For each diagram you review:
- Analyze and internalize the full provided diagram and rubric.
- Systematically assess each rubric item, explaining your reasoning and specific evidence from the diagram for each, and then output your evaluation. Strictly output from the options for that rubric criterion that are provided below.

## Rubric

### Mathematical Correctness

- **Shape is closed (no gaps in outline):** Yes | No - whether the diagram's shape is closed. This is independent from whether it is fully in frame (below) - is the diagram formed that it would likely be closed regardless of its framing?
- **Labeled angles (if any) match the drawn angle:** Yes | No | N/A - whether the labeled angles in the diagram match their labeled value. Right angle markers without a number also count. N/A if there are no labeled angles.
- **Labeled lengths (if any) match visual proportions:** Yes | No | N/A - whether the labeled lengths or areas shown in the diagram are reasonable lengths or areas in relationship to each other. N/A if there are no labeled lengths or areas.
- **Core mathematical properties of the shape are correct:** Yes | No - whether the core mathematical properties of the shape are correct, independent of the criteria above.

### Spatial Correctness

- **Diagram is fully in frame:** Yes | No - whether all diagram elements are in the frame, and nothing is cut off.
- **Diagram elements are scaled to be readable:** Yes | No - whether elements such as shapes, labels, etc. are sized to be readable, especially in relationship to each other.
- **Labels (if any) are associated with correct elements:** Yes | No | N/A - whether the labels are associated with the correct elements (e.g. sides, line segments, angles, etc) in the diagram. N/A if there are no labels.
- **Diagram elements don't problematically overlap:** Yes | No - whether no elements problematically overlap. Problematically overlapping could include labels overlapping with something so they cannot be read easily, shapes or elements of the diagram overlapping in a way that makes it challenging to interpret. A label directly intersected by a line segment would be considered problematically overlapping.

## Output Format

After reasoning and determining each criteria's evaluation, output a JSON object in the following format:
```json
{
  "shape_outlines_are_closed": {
    "rationale": "[Placeholder: rationale for Shape is closed (no gaps in outline)]",
    "value": "[Placeholder: Yes or No]"
  },
  "angle_labels_matches_arcs": {
    "rationale": "[Placeholder: rationale for Labeled angles (if any) match the drawn angle]",
    "value": "[Placeholder: Yes, No, or N/A]"
  },
  "labeled_lengths_areas_match_proportions": {
    "rationale": "[Placeholder: rationale for Labeled lengths (if any) match visual proportions]",
    "value": "[Placeholder: Yes, No, or N/A]"
  },
  "core_mathematical_properties_of_shapes_correct": {
    "rationale": "[Placeholder: rationale for Core mathematical properties of the shape are correct]",
    "value": "[Placeholder: Yes or No]"
  },
  "diagram_fully_in_canvas": {
    "rationale": "[Placeholder: rationale for Diagram is fully in frame]",
    "value": "[Placeholder: Yes or No]"
  },
  "diagram_elements_are_readable_size": {
    "rationale": "[Placeholder: rationale for Diagram elements are scaled to be readable]",
    "value": "[Placeholder: Yes or No]"
  },
  "labels_associated_with_elements": {
    "rationale": "[Placeholder: rationale for Labels (if any) are associated with correct elements]",
    "value": "[Placeholder: Yes, No, or N/A]"
  },
  "diagram_elements_dont_problematically_overlap": {
    "rationale": "[Placeholder: rationale for Diagram elements don't problematically overlap]",
    "value": "[Placeholder: Yes or No]"
  }
}
```

Output ONLY the JSON code. Your role is to act as a thorough, unbiased judge; always complete detailed reasoning for every rubric criterion before scoring or conclusion. Be meticulous and transparent in your evaluations. Ensure the rationale clearly explains your evaluation from the criteria based on the provided diagram, and that the value is selected from the available options.

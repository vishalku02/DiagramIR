You are a deterministic parser that extracts geometric entities from TikZ code into JSON that matches the provided schema.

### Rules
1. Extract ONLY entities explicitly present in the TikZ code.
2. Omit empty fields (do not include keys with empty lists).
3. Preserve exact numerical coordinates from the code. Resolve and compute if necessary (e.g. \\def or \\newcommand), but DO NOT infer extra ones that are not present in the code or complete partial shapes.
4. For shapes array, list vertices in the order they are drawn, set "cycle": true if the draw command ends with `-- cycle`.
5. For rectangle_primitives array, set "is_right_angle_symbol": true when the rectangle is drawn as a right-angle marker (e.g., tiny square sharing corners with two incident edges or comments mentioning a right angle). Otherwise set it to false. If a `\\draw` explicitly passes the `right angle symbol` option, do not add it as a rectangle_primitive.
6. For 3D parts (shapes w/ 3D coords) and Ucubes, use one integer id per physical solid (e.g., 1, 1, 1). Faces or unit cube entries that belong to the same block, especially those emitted inside the same scope/loop, must reuse that id; only assign a new id when you are describing a genuinely different solid.
7. When you encounter the helper macro `\\Ucube` (or any change of coordinates that draws the front/right/top faces of a unit cube), output a single shape object with `"type": "Ucube"` instead of three separate `3D-part` entries. The cube should include its `id`, `size` (usually `[1,1,1]` unless the macro scales it), the scope transform (`scale`, `shift`, `xshift`, `yshift`), and `fill`. Do not emit the individual faces separately.
8. If a scope applies transformations (`shift`, `scale`, `xshift`, `yshift`, `rotate`), include them in the optional transform object. Do not numerically apply the transform.
9. Transform separation:
   - `transform.shift` <- only TikZ's `shift={...}` argument.
   - `transform.xshift` <- only TikZ's `xshift=...`.
   - `transform.yshift` <- only TikZ's `yshift=...`.
   These must NEVER be combined. If `xshift`/`yshift` values are omitted or folded into `shift`, the JSON is invalid.
10. For node options such as `rotate=...`, set the node's `node_rotate` field. Keep scope-level rotations in `transform` and do not duplicate them in `node_rotate`.
11. For `tikzpicture_options`, map x, y, and z to the corresponding options, or fill out `scale` if the options include `scale`.
12. Loop expansion: Expand every `\\foreach` loop literally. Substitute each variable with its values and emit the corresponding repeated scopes and draw commands. Never summarize or replace with a generic cube; output must reflect exactly the iterations.
13. For custom commands, resolve the command using this table:

```python
IM_MACROS = [
    "\\TFP": "4.875in",
    "\\TTP": "4.2in",
    "\\TwoThirdsPage": "4.2in",
    "\\HP": "3.25in",
    "\\HalfPage": "3.25in",
    "\\THP": "2.1in",
    "\\ThirdPage": "2.1in",
    "\\QP": "1.625in",
    "\\QuarterPage": "1.625in",
]
```

14. Resolve relative coordinate syntax precisely: when you encounter forms like `+(...)`, `++(...)`, or `($(P)!t!(Q)$)`, evaluate them to absolute coordinates using previously defined points. For `++`, remember it updates the current point before the next coordinate is processed.

### Example

TikZ:
```latex
\\foreach \\x in {0,1} {
  \\begin{scope}[scale=0.5, shift={(\\x,0,0)}, xshift=-2in, yshift=-3in]
    \\draw (0,0,0) -- (1,0,0) -- (1,1,0) -- (0,1,0) -- cycle;
  \\end{scope}
}
```

JSON:

```json
{
  "shapes": [
    {
      "type": "3D-part",
      "vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
      "cycle": true,
      "id": 1,
      "transform": {
        "scale": 0.5,
        "shift": [0,0,0],
        "xshift": "-2in",
        "yshift": "-3in"
      }
    },
    {
      "type": "3D-part",
      "vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
      "cycle": true,
      "id": 2,
      "transform": {
        "scale": 0.5,
        "shift": [1,0,0],
        "xshift": "-2in",
        "yshift": "-3in"
      }
    }
  ]
}
```

### TikZ
{tikz_code}

### Output
JSON only, no explanations.

import json
import math
from math import isclose
import re
import types
import warnings
from pydantic import ValidationError
from shapely.geometry import Polygon, Point, LineString
from PIL import ImageFont
import os
from typing import Optional
import numpy as np

from geometry_engine import (
    _axes_from_coordinate_system,
    _base_z_vector,
    _dimension_to_pt,
    _eval_simple_expression,
    _parse_axis_vector,
    _parse_simple_term,
    _shift_to_units,
    apply_matrix_transform,
    apply_transform_only,
    apply_transforms,
    build_transformation_matrix,
    expand_macros,
    expand_unit_cube_faces,
    generate_arc_geometry,
    project_3d_to_2d,
    rotate_point_around_center,
    unit_cube_faces,
    unit_cube_vertices,
    to_geometry,
    to_pt,
    _to_float_default,
    _resolve_transform_scale,
    _resolve_transform_shift,
    transformed_3d_vertices,
)

from IR_model import TikzIR

class Evaluator:
    """
    Automatic evaluation system for symbolic representations of diagrams
    The evalutor noramlizes all units to TeX points (pt).

     """

    EPS_LENGTH = 1e-3 # 0.1%
    EPS_ANGLE = 1e-2 # 1%
    EPS_LAW_SINES = 1e-1
    CANVAS_TOLERANCE_PT = 5.0  # allow a 1pt slack when checking frame boundaries

    # label matching tolerances
    ANGLE_LABEL_BASE_TOLERANCE_PT = 55.0
    NUMERIC_LABEL_BASE_TOLERANCE_PT = 55.0
    TEXT_LABEL_BASE_TOLERANCE_PT = 75.0
    LABEL_INSIDE_BUFFER_PT = 5.0
    LABEL_AMBIGUITY_MARGIN_PT = 3.0
    MAX_LABEL_TOLERANCE_PT = 100.0
    ARC_ENDPOINT_TOLERANCE_PT = 1.0


    # readability threshold
    MIN_ELEMENT_DIMENSION_RELATIVE = 0.01  # 1% of canvas min dimension
    DEGENERACY_TOLERANCE = 1e-6

    PAGE_WIDTH_IN = 8.5
    PAGE_HEIGHT_IN = 11.0

    # problematic overlap tolerances
    NODE_OVERLAP_AREA_TOLERANCE_PT2 = 2.0
    NODE_EDGE_DISTANCE_TOLERANCE_PT = 1.0
    NODE_FACE_BOUNDARY_OVERLAP_AREA_TOLERANCE_PT2 = 1.0
    NODE_FACE_BOUNDARY_OVERLAP_LENGTH_TOLERANCE_PT = 2.0
    THREE_D_FACE_OVERLAP_TOLERANCE_PT2 = 4.0
    THREE_D_FACE_Z_SEPARATION_TOLERANCE = 0.5
    SEGMENT_FACE_OVERLAP_LENGTH_TOLERANCE_PT = 2.0
    FACE_EDGE_BUFFER_PT = 0.5
    THREE_D_FACE_PARALLEL_ANGLE_TOLERANCE_DEG = 15.0
    RIGHT_ANGLE_TOLERANCE_DEG = 2.0
    RIGHT_ANGLE_VERTEX_TOLERANCE_PT = 2.0

    UNIT_TO_PT = {
        "pt": 1.0,
        "in": 72.27,
        "cm": 28.45274,
        "mm": 2.845274,
        "px": 72.27 / 96.0, # 96px
    }

    IM_MACROS = {
        "\\TFP": "4.875in",
        "\\TTP": "4.2in",
        "\\TwoThirdsPage": "4.2in",
        "\\HP": "3.25in",
        "\\HalfPage": "3.25in",
        "\\THP": "2.1in",
        "\\ThirdPage": "2.1in",
        "\\QP": "1.625in",
        "\\QuarterPage": "1.625in",
    }

    # Text constants to calculate bounding boxes
    DEFAULT_FONT_SIZE_PT = 11.4 # pt
    DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), "styles", "OpenSans-VariableFont_wdth,wght.ttf")

    # Geometry helpers were extracted for clarity; binding them here keeps the
    # calls identical to the prevoius implementation.
    expand_macros = expand_macros
    to_pt = to_pt
    _parse_simple_term = _parse_simple_term
    _eval_simple_expression = _eval_simple_expression
    _shift_to_units = _shift_to_units
    _parse_axis_vector = _parse_axis_vector
    _axes_from_coordinate_system = _axes_from_coordinate_system
    _base_z_vector = _base_z_vector
    _dimension_to_pt = _dimension_to_pt
    build_transformation_matrix = build_transformation_matrix
    apply_matrix_transform = apply_matrix_transform
    apply_transforms = apply_transforms
    apply_transform_only = apply_transform_only
    generate_arc_geometry = generate_arc_geometry
    project_3d_to_2d = project_3d_to_2d
    rotate_point_around_center = rotate_point_around_center
    to_geometry = to_geometry
    unit_cube_vertices = unit_cube_vertices
    unit_cube_faces = unit_cube_faces
    expand_unit_cube_faces = expand_unit_cube_faces
    _to_float_default = _to_float_default
    _resolve_transform_scale = _resolve_transform_scale
    _resolve_transform_shift = _resolve_transform_shift
    transformed_3d_vertices = transformed_3d_vertices


    def __init__(self, debug: bool = False, enforce_page_bounds: bool = False):
        self.debug = debug
        self.enforce_page_bounds = enforce_page_bounds
        self._page_canvas = self._create_page_canvas_polygon() if self.enforce_page_bounds else None
        self.global_evals = [
            self.angle_labels_matches_arcs,
            self.diagram_fully_in_canvas,
            self.labels_associated_with_elements,
            self.diagram_elements_dont_problematically_overlap,
            self.diagram_elements_are_readable_size,
            self.shape_outlines_are_closed,
            self.core_mathematical_properties_of_shapes_correct,
            self.labeled_lengths_areas_match_proportions,
        ]

        self._font_cache = None
        # preload font so repeated node checks reuse it without reinitializing
        self._load_font()

    def _create_page_canvas_polygon(self) -> Polygon:
        half_w = (self.PAGE_WIDTH_IN * self.UNIT_TO_PT["in"]) / 2
        half_h = (self.PAGE_HEIGHT_IN * self.UNIT_TO_PT["in"]) / 2
        return Polygon([
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ])

    def _debug(self, message: str):
        if getattr(self, 'debug', False):
            print(message)
    
    def is_tikz(self, response: dict) -> bool:
        return 'tikz_code' in response

    def _wrap_namespace(self, value):
        if isinstance(value, dict):
            converted = {}
            for key, val in value.items():
                new_key = 'from_' if key == 'from' else key
                converted[new_key] = self._wrap_namespace(val)
            return types.SimpleNamespace(**converted)
        if isinstance(value, list):
            return [self._wrap_namespace(item) for item in value]
        return value

    def _load_font(self):
        if getattr(self, '_font_cache', None) is not None:
            return self._font_cache
        font_size = int(self.DEFAULT_FONT_SIZE_PT)
        font = ImageFont.truetype(self.DEFAULT_FONT_PATH, size=font_size)
        font.set_variation_by_axes([100.0, 400.0]) # width=100, weight=400
        self._font_cache = font
        return font
    
    def evaluate(self, response):
        """
        Run global evaluation checks.
        Returns a dict with results and a score in [0,1]
        """

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                return {
                    'global': [{
                        'check': 'json_parse',
                        'passed': False,
                        'message': str(exc),
                    }],
                    'score': 0.0,
                    'overall_pass': False,
                }

        results = {'global': [], 'score': 0.0, 'overall_pass': False}

        if isinstance(response, TikzIR):
            ir_model = response
        else:
            try:
                ir_model = TikzIR.model_validate(response, strict=True)
            except ValidationError as exc:
                results["global"].append({
                    "check": "schema_validation",
                    "passed": False,
                    "message": f"Schema validation failed: {str(exc)}",
                })
                ir_model = self._wrap_namespace(response)

        for evaluation_fn in self.global_evals:
            passed, message = evaluation_fn(ir_model)
            results["global"].append({"check": evaluation_fn.__name__, "passed": passed, "message": message})

        applicable_checks = [check for check in results['global'] if check['passed'] != "N/A"]
        total_checks = len(applicable_checks)
        passed_checks = sum(1 for check_result in applicable_checks if check_result['passed'] is True)
        score = (passed_checks / total_checks) if total_checks > 0 else 0.0
        results['score'] = score
        results['overall_pass'] = (score == 1.0)

        return results


    # ----------------------------------------------------------------------------- global

    PROPORTION_REL_TOL = 0.05
    def labeled_lengths_areas_match_proportions(self, ir):
        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        def _parse_numeric(text: str) -> Optional[float]:
            cleaned = self.clean_latex_text(text)
            frac = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*', cleaned)
            if frac:
                try:
                    return float(frac.group(1)) / float(frac.group(2))
                except ZeroDivisionError:
                    return None
            m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
            if not m:
                return None
            try:
                return float(m.group(1))
            except ValueError:
                return None

        def _collect_segments():
            out = []
            for seg in getattr(ir, 'line_segments', []) or []:
                geom = self.to_geometry(seg, coordinate_system, getattr(seg, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if not isinstance(geom, LineString):
                    try:
                        geom = LineString(list(geom.coords))
                    except Exception:
                        continue
                out.append({'entity': seg, 'geometry': geom})
            return out

        def _collect_shapes():
            polys = []
            for shp in getattr(ir, 'shapes', []) or []:
                geom = self.to_geometry(shp, coordinate_system, getattr(shp, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == 'Polygon':
                    polys.append({'entity': shp, 'geometry': geom})
            for rect in getattr(ir, 'rectangle_primitives', []) or []:
                geom = self.to_geometry(rect, coordinate_system, getattr(rect, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == 'Polygon':
                    polys.append({'entity': rect, 'geometry': geom})
            return polys

        def _collect_numeric_labels():
            labels = []

            for node in getattr(ir, 'nodes', []) or []:
                raw = getattr(node, 'text', '') or ''
                cleaned = self.clean_latex_text(raw)
                if not cleaned:
                    continue
                if self._classify_label(raw, cleaned) != 'numeric':
                    continue
                geom = self.to_geometry(node, coordinate_system, getattr(node, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                val = _parse_numeric(cleaned)
                if val is None:
                    continue
                labels.append({'entity': node, 'geometry': geom, 'value': val, 'text': cleaned})

            for seg in getattr(ir, 'line_segments', []) or []:
                seg_geom = self.to_geometry(seg, coordinate_system, getattr(seg, 'transform', None))
                if seg_geom is None or seg_geom.is_empty:
                    continue

                for sub in getattr(seg, 'nodes', []) or []:
                    raw = getattr(sub, 'text', '') or ''
                    cleaned = self.clean_latex_text(raw)
                    if not cleaned or self._classify_label(raw, cleaned) != 'numeric':
                        continue
                    val = _parse_numeric(cleaned)
                    if val is None:
                        continue
                    geom = self.to_geometry(sub, coordinate_system, getattr(sub, 'transform', None))
                    if geom is None or geom.is_empty:
                        mid = seg_geom.interpolate(0.5, normalized=True)
                        geom = Point(float(mid.x), float(mid.y))
                    labels.append({'entity': seg, 'geometry': geom, 'value': val, 'text': cleaned})

                for key in ('label', 'text', 'node_text', 'mid_text', 'annotation'):
                    raw = getattr(seg, key, None)
                    if not raw:
                        continue
                    cleaned = self.clean_latex_text(raw)
                    if self._classify_label(raw, cleaned) != 'numeric':
                        continue
                    val = _parse_numeric(cleaned)
                    if val is None:
                        continue
                    mid = seg_geom.interpolate(0.5, normalized=True)
                    geom = Point(float(mid.x), float(mid.y))
                    labels.append({'entity': seg, 'geometry': geom, 'value': val, 'text': cleaned})

            return labels

        def _associate_labels(numeric_labels, segments, polygons):
            length_pairs, area_pairs = [], []

            SNAP_TO_EDGE_TOL_PT = 12.0
            AREA_INTERIOR_SHRINK_PT = 2.0

            poly_interiors = []
            for poly in polygons:
                interior = poly['geometry'].buffer(-AREA_INTERIOR_SHRINK_PT)
                if interior.is_empty:
                    interior = poly['geometry']
                poly_interiors.append({'geometry': interior, 'entity': poly['entity'], 'orig': poly['geometry']})

            for lab in numeric_labels:
                pt = lab['geometry'].centroid

                nearest_seg = None
                nearest_seg_dist = float('inf')
                for s in segments:
                    d = pt.distance(s['geometry'])
                    if d < nearest_seg_dist:
                        nearest_seg_dist = d
                        nearest_seg = s

                containing = None
                for interior in poly_interiors:
                    if interior['geometry'].covers(pt): 
                        containing = interior
                        break
                
                # feel free to change if you feel like a diff decision boundary is ebtter,
                # but i assume: if very close to edge (inside or outside polygon, then length)
                # if considerably inside polygon, then area
                # else, length to the nearest segment
                if nearest_seg is not None and nearest_seg_dist <= SNAP_TO_EDGE_TOL_PT:
                    length_pairs.append({
                        'label_value': lab['value'],
                        'seg_geom': nearest_seg['geometry'],
                        'seg_id': getattr(nearest_seg['entity'], 'id', None),
                        'label_text': lab['text']
                    })
                elif containing is not None:
                    area_pairs.append({
                        'label_value': lab['value'],
                        'poly_geom': containing['orig'],
                        'poly_id': getattr(containing['entity'], 'id', None),
                        'label_text': lab['text']
                    })
                elif nearest_seg is not None:
                    length_pairs.append({
                        'label_value': lab['value'],
                        'seg_geom': nearest_seg['geometry'],
                        'seg_id': getattr(nearest_seg['entity'], 'id', None),
                        'label_text': lab['text']
                    })

            return length_pairs, area_pairs


        def _check_pairwise_proportions(items, meas_key, label_key, what: str):
            mismatches = []
            n = len(items)
            if n < 2:
                return mismatches
            measured = []
            for it in items:
                if meas_key == 'length':
                    mv = it['seg_geom'].length
                else:
                    mv = it['poly_geom'].area
                measured.append(mv)

            for i in range(n):
                for j in range(i + 1, n):
                    mi, mj = measured[i], measured[j]
                    li, lj = items[i][label_key], items[j][label_key]
                    lhs = mi * lj
                    rhs = mj * li
                    denom = max(abs(rhs), 1e-9)
                    rel_err = abs(lhs - rhs) / denom
                    if rel_err > self.PROPORTION_REL_TOL:
                        a_desc = items[i].get('seg_id' if meas_key == 'length' else 'poly_id')
                        b_desc = items[j].get('seg_id' if meas_key == 'length' else 'poly_id')
                        a_desc = f"id={a_desc}" if a_desc is not None else "(no id)"
                        b_desc = f"id={b_desc}" if b_desc is not None else "(no id)"
                        mismatches.append(
                            f"{what}: ({a_desc}, label={li:g}) vs ({b_desc}, label={lj:g}) "
                            f"→ rel. error {rel_err:.3f}"
                        )
            return mismatches

        segments = _collect_segments()
        polygons = _collect_shapes()
        numeric_labels = _collect_numeric_labels()

        if not numeric_labels:
            return "N/A", "No numeric labels to check"

        length_pairs, area_pairs = _associate_labels(numeric_labels, segments, polygons)

        issues = []
        if len(length_pairs) >= 2:
            issues += _check_pairwise_proportions(
                [{**p, 'length': p['seg_geom'].length, 'label': p['label_value']} for p in length_pairs],
                meas_key='length', label_key='label', what="Length proportion mismatch"
            )

        if len(area_pairs) >= 2:
            issues += _check_pairwise_proportions(
                [{**p, 'area': p['poly_geom'].area, 'label': p['label_value']} for p in area_pairs],
                meas_key='area', label_key='label', what="Area proportion mismatch"
            )

        if issues:
            sample = "; ".join(issues[:3])
            if len(issues) > 3:
                sample += f" (+{len(issues) - 3} more)"
            return False, sample

        checked_groups = (len(length_pairs) >= 2) + (len(area_pairs) >= 2)
        if checked_groups:
            return True, "Labeled lengths/areas match measured proportions"

        return "N/A", "Not enough labeled pairs to compare proportions"

    def core_mathematical_properties_of_shapes_correct(self, ir) -> tuple[bool, str]:
        issues: list[str] = []
        tol = self.DEGENERACY_TOLERANCE

        triangles = [shape for shape in getattr(ir, 'shapes', []) or [] if getattr(shape, 'type', None) == 'triangle']
        for triangle in triangles:
            vertices = (getattr(triangle, 'vertices', []) or [])[:3]
            if len(vertices) < 3:
                issues.append("Triangle missing vertices")
                continue
            duplicate = False
            for i in range(3):
                for j in range(i + 1, 3):
                    vi = vertices[i]
                    vj = vertices[j]
                    if len(vi) < 2 or len(vj) < 2:
                        continue
                    distance = ((vi[0] - vj[0]) ** 2 + (vi[1] - vj[1]) ** 2) ** 0.5
                    if distance <= tol:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                issues.append("Triangle has coincident vertices")
                continue
            ax, ay = vertices[0][:2]
            bx, by = vertices[1][:2]
            cx, cy = vertices[2][:2]
            area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5
            if area <= tol:
                issues.append("Triangle vertices are colinear")

        for rectangle in getattr(ir, 'rectangle_primitives', []) or []:
            corner1 = getattr(rectangle, 'corner1', None) or []
            corner2 = getattr(rectangle, 'corner2', None) or []
            if len(corner1) < 2 or len(corner2) < 2:
                issues.append("Rectangle missing corners")
                continue
            width = abs(corner2[0] - corner1[0])
            height = abs(corner2[1] - corner1[1])
            if width <= tol or height <= tol:
                issues.append("Rectangle has zero area")

        for circle in getattr(ir, 'circles', []) or []:
            radius = getattr(circle, 'radius', 0.0) or 0.0
            if radius <= tol:
                issues.append("Circle radius must be positive")

        part_vertices: dict = {}
        for shape in getattr(ir, 'shapes', []) or []:
            if getattr(shape, 'type', None) != '3D-part':
                continue
            part_id = getattr(shape, 'id', None)
            transform = getattr(shape, 'transform', None)
            raw_vertices = getattr(shape, 'vertices', []) or []
            transformed = self.transformed_3d_vertices(raw_vertices, transform)
            if not transformed:
                transformed = raw_vertices
            key = part_id if part_id is not None else id(shape)
            entry = part_vertices.setdefault(key, [])
            entry.extend(vertex for vertex in transformed if isinstance(vertex, (list, tuple)) and len(vertex) >= 3)

        for part_id, vertices in part_vertices.items():
            if not vertices:
                issues.append(f"3D-part id={part_id} has no 3D vertices")
                continue
            rounded = {(
                round(pt[0], 6),
                round(pt[1], 6),
                round(pt[2], 6)
            ) for pt in vertices}
            unique_count = len(rounded)
            if unique_count < 7:
                issues.append(f"3D-part id={part_id} has too few distinct corners ({unique_count})")
                continue
            if unique_count > 8:
                issues.append(f"3D-part id={part_id} has extra distinct corners ({unique_count})")
                continue
            xs = [pt[0] for pt in vertices]
            ys = [pt[1] for pt in vertices]
            zs = [pt[2] for pt in vertices]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            span_z = max(zs) - min(zs)
            if span_x <= tol or span_y <= tol or span_z <= tol:
                issues.append(f"3D-part id={part_id} collapses along an axis")

        if issues:
            summary = "; ".join(issues[:3])
            if len(issues) > 3:
                summary += f" (+{len(issues) - 3} more)"
            return False, summary

        return True, "Shape parameters are non-degenerate"

    def diagram_fully_in_canvas(self, ir) -> bool:
        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        working_canvas = self._page_canvas if self.enforce_page_bounds else None

        clips = getattr(ir, 'clips', []) or []
        for clip in clips:
            clip_transform = getattr(clip, 'transform', None)
            clip_geom = self.to_geometry(clip, coordinate_system, clip_transform)
            if clip_geom is None or clip_geom.is_empty:
                continue
            working_canvas = clip_geom if working_canvas is None else working_canvas.intersection(clip_geom)

        if working_canvas is None:
            return True, "No canvas constraints applied"

        if working_canvas.is_empty:
            return False, "Canvas constraints result in empty region"

        if self.CANVAS_TOLERANCE_PT:
            working_canvas = working_canvas.buffer(self.CANVAS_TOLERANCE_PT)

        entity_groups = (
            ('shapes', getattr(ir, 'shapes', []) or []),
            ('rectangle_primitives', getattr(ir, 'rectangle_primitives', []) or []),
            ('circles', getattr(ir, 'circles', []) or []),
            ('line_segments', getattr(ir, 'line_segments', []) or []),
            ('arcs', getattr(ir, 'arcs', []) or []),
        )

        for entity_type, entities in entity_groups:
            for entity in entities:
                entity_transform = getattr(entity, 'transform', None)
                geom = self.to_geometry(entity, coordinate_system, entity_transform)
                if geom is None or geom.is_empty:
                    continue
                if not geom.within(working_canvas):
                    entity_id = getattr(entity, 'id', None)
                    id_part = f" id={entity_id}" if entity_id is not None else ""
                    minx, miny, maxx, maxy = geom.bounds
                    bbox = f"(({minx:.1f}, {miny:.1f}) to ({maxx:.1f}, {maxy:.1f}))"
                    return False, (
                        f"{entity_type}{id_part} bbox {bbox} exceeds canvas allowance "
                        f"(+/- {self.CANVAS_TOLERANCE_PT:.1f}pt)"
                    )

        for node in getattr(ir, 'nodes', []) or []:
            raw_text = getattr(node, 'text', '') or ''
            if not self.clean_latex_text(raw_text):
                continue
            node_transform = getattr(node, 'transform', None)
            geom = self.to_geometry(node, coordinate_system, node_transform)
            if geom is None or geom.is_empty:
                continue
            if not geom.within(working_canvas):
                minx, miny, maxx, maxy = geom.bounds
                bbox = f"(({minx:.1f}, {miny:.1f}) to ({maxx:.1f}, {maxy:.1f}))"
                return False, (
                    f"node bbox {bbox} exceeds canvas allowance "
                    f"(+/- {self.CANVAS_TOLERANCE_PT:.1f}pt)"
                )

        return True, "Diagram is fully in canvas"

    def labels_associated_with_elements(self, ir):
        coordinate_system = getattr(ir, 'tikzpicture_options', None)
        issues = []

        # ---------- Helpers & Setup ----------
        def collect_entities(attribute_name, entity_kind):
            collected_entities = []
            for entity in getattr(ir, attribute_name, []) or []:
                if attribute_name == 'arcs':
                    if (
                        getattr(entity, 'center', None) is None
                        or getattr(entity, 'radius', None) is None
                        or getattr(entity, 'start_angle', None) is None
                        or getattr(entity, 'end_angle', None) is None
                    ):
                        continue
                transform = getattr(entity, 'transform', None)
                geometry = self.to_geometry(entity, coordinate_system, transform)
                if geometry is None or geometry.is_empty:
                    continue
                collected_entities.append({'geometry': geometry, 'entity': entity, 'kind': entity_kind})
            return collected_entities

        shape_entities = collect_entities('shapes', 'shape') + collect_entities('rectangle_primitives', 'rectangle')
        segment_entities = collect_entities('line_segments', 'segment')
        arc_entities = collect_entities('arcs', 'arc')

        shape_boundaries = []
        for entity_record in shape_entities:
            geometry = entity_record['geometry']
            boundary_geometry = geometry.boundary if geometry.geom_type == 'Polygon' else geometry
            shape_boundaries.append({'geometry': boundary_geometry, 'entity': entity_record['entity'], 'kind': f"{entity_record['kind']}_edge"})

        label_nodes = []
        for node in getattr(ir, 'nodes', []) or []:
            raw_text = getattr(node, 'text', '') or ''
            clean_text = self.clean_latex_text(raw_text)
            if not clean_text:
                continue
            label_type = self._classify_label(raw_text, clean_text)
            transform = getattr(node, 'transform', None)
            geometry = self.to_geometry(node, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            label_nodes.append({'geometry': geometry, 'entity': node, 'text': clean_text, 'raw': raw_text, 'label_type': label_type})

        if issues:
            return False, issues[0]

        if not label_nodes:
            return "N/A", "No labels to check"

        def compute_tolerance(base_pt: float, candidate_record) -> float:
            geometry = candidate_record['geometry']
            tolerance = base_pt
            minx, miny, maxx, maxy = geometry.bounds
            width = maxx - minx
            height = maxy - miny
            dimensions = [dimension for dimension in (width, height) if dimension > 0]
            if dimensions:
                tolerance = max(tolerance, min(dimensions))
            return min(tolerance, self.MAX_LABEL_TOLERANCE_PT)

        def candidate_priority(kind: str, label_type: str) -> int:
            if label_type == 'angle':
                return 0 if 'arc' in kind else 1
            if label_type == 'numeric':
                if 'segment' in kind or kind.endswith('_edge'):
                    return 0
                if 'shape' in kind or 'rectangle' in kind:
                    return 1
                return 2
            if 'segment' in kind or kind.endswith('_edge'):
                return 0
            if 'shape' in kind or 'rectangle' in kind:
                return 1
            if 'arc' in kind:
                return 2
            return 3

        def nearest_candidates(label_geometry, candidates, label_type):
            ranked_candidates = []
            for candidate_record in candidates:
                distance_to_candidate = label_geometry.distance(candidate_record['geometry'])
                priority = candidate_priority(candidate_record['kind'], label_type)
                ranked_candidates.append((distance_to_candidate, priority, candidate_record))
            ranked_candidates.sort(key=lambda item: (item[0], item[1]))
            return ranked_candidates

        labels_associated = 0
        tie_notes_messages = []
        assignment_summaries = []

        def format_assignment(label_text: str, candidate_record, distance_pt=None, note=None) -> str:
            kind = candidate_record['kind'] or "element"
            entity = candidate_record['entity']
            element_desc = kind
            entity_id = getattr(entity, 'id', None)
            if entity_id is not None:
                element_desc += f" id={entity_id}"
            entity_type = getattr(entity, 'type', None)
            if entity_type and entity_type != kind:
                element_desc += f" ({entity_type})"
            assignment_description = f"Label '{label_text}' -> {element_desc}"
            extra_details = []
            if distance_pt is not None:
                extra_details.append(f"distance={distance_pt:.1f}pt")
            if note:
                extra_details.append(note)
            if extra_details:
                assignment_description += f" ({', '.join(extra_details)})"
            return assignment_description

        # ---------- main loop: match each label to the best diagram element. ----------
        for node in label_nodes:
            label_geometry = node['geometry']
            label_text = node['text']
            label_type = node['label_type']

            if label_type == 'angle':
                if not arc_entities:
                    issues.append(f"Angle label '{label_text}' with no arc present")
                    continue
                candidate_matches = nearest_candidates(label_geometry, arc_entities, label_type)
                closest_distance, closest_priority, closest_candidate = candidate_matches[0]
                tolerance = compute_tolerance(self.ANGLE_LABEL_BASE_TOLERANCE_PT, closest_candidate)
                if closest_distance > tolerance:
                    issues.append(
                        f"Angle label '{label_text}' is {closest_distance:.1f}pt from nearest arc (limit {tolerance:.1f}pt)"
                    )
                    continue
                if len(candidate_matches) > 1:
                    second_distance, second_priority, _ = candidate_matches[1]
                    if (
                        (second_distance - closest_distance) < self.LABEL_AMBIGUITY_MARGIN_PT
                        and second_priority == closest_priority
                    ):
                        tie_notes_messages.append(
                            f"Angle label '{label_text}' ties arcs (difference={second_distance - closest_distance:.1f}pt)"
                        )
                assignment_summaries.append(format_assignment(label_text, closest_candidate, closest_distance))
                labels_associated += 1
                continue

            if label_type == 'numeric':
                numeric_candidates = segment_entities + shape_boundaries
                if not numeric_candidates:
                    issues.append(f"Numeric label '{label_text}' with no segments or edges available")
                    continue
                candidate_matches = nearest_candidates(label_geometry, numeric_candidates, label_type)
                closest_distance, closest_priority, closest_candidate = candidate_matches[0]
                tolerance = compute_tolerance(self.NUMERIC_LABEL_BASE_TOLERANCE_PT, closest_candidate)
                if closest_distance > tolerance:
                    issues.append(
                        f"Numeric label '{label_text}' is {closest_distance:.1f}pt from nearest edge (limit {tolerance:.1f}pt)"
                    )
                    continue
                if len(candidate_matches) > 1:
                    second_distance, second_priority, _ = candidate_matches[1]
                    if (
                        (second_distance - closest_distance) < self.LABEL_AMBIGUITY_MARGIN_PT
                        and second_priority == closest_priority
                    ):
                        tie_notes_messages.append(
                            f"Numeric label '{label_text}' ties edges (difference={second_distance - closest_distance:.1f}pt)"
                        )
                assignment_summaries.append(format_assignment(label_text, closest_candidate, closest_distance))
                labels_associated += 1
                continue

            containing_shape = None
            for candidate in shape_entities:
                if candidate['geometry'].buffer(self.LABEL_INSIDE_BUFFER_PT).contains(label_geometry.centroid):
                    containing_shape = candidate
                    break

            if containing_shape is not None:
                assignment_summaries.append(
                    format_assignment(label_text, containing_shape, None, "inside shape")
                )
                labels_associated += 1
                continue

            all_candidate_entities = shape_entities + segment_entities + arc_entities
            if not all_candidate_entities:
                issues.append(f"Text label '{label_text}' has no nearby diagram elements")
                continue

            candidate_matches = nearest_candidates(label_geometry, all_candidate_entities, label_type)
            closest_distance, closest_priority, closest_candidate = candidate_matches[0]
            tolerance = compute_tolerance(self.TEXT_LABEL_BASE_TOLERANCE_PT, closest_candidate)
            if closest_distance > tolerance:
                issues.append(
                    f"Text label '{label_text}' is {closest_distance:.1f}pt from nearest element (limit {tolerance:.1f}pt)"
                )
                continue
            if len(candidate_matches) > 1:
                second_distance, second_priority, _ = candidate_matches[1]
                if (
                    (second_distance - closest_distance) < self.LABEL_AMBIGUITY_MARGIN_PT
                    and second_priority == closest_priority
                ):
                    tie_notes_messages.append(
                        f"Text label '{label_text}' ties elements (difference={second_distance - closest_distance:.1f}pt)"
                    )

            assignment_summaries.append(format_assignment(label_text, closest_candidate, closest_distance))
            labels_associated += 1

        if issues:
            summary = "; ".join(issues[:3])
            if len(issues) > 3:
                summary += f" (+{len(issues) - 3} more)"
            return False, summary

        success_message = f"Labels assigned ({labels_associated} total)"
        if assignment_summaries:
            success_message += ": " + "; ".join(assignment_summaries)
        if tie_notes_messages:
            tie_summary = "; ".join(tie_notes_messages[:2])
            if len(tie_notes_messages) > 2:
                tie_summary += f" (+{len(tie_notes_messages) - 2} more ties)"
            success_message += f" (ties noted: {tie_summary})"

        return True, success_message

    PROPORTION_REL_TOL = 0.05

    def labeled_lengths_areas_match_proportions(self, ir) -> tuple[bool, str]:
        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        def _parse_numeric(text: str) -> Optional[float]:
            cleaned = self.clean_latex_text(text)
            frac = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*', cleaned)
            if frac:
                try:
                    return float(frac.group(1)) / float(frac.group(2))
                except ZeroDivisionError:
                    return None
            m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
            if not m:
                return None
            try:
                return float(m.group(1))
            except ValueError:
                return None

        def _collect_segments():
            out = []
            for seg in getattr(ir, 'line_segments', []) or []:
                geom = self.to_geometry(seg, coordinate_system, getattr(seg, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if not isinstance(geom, LineString):
                    try:
                        geom = LineString(list(geom.coords))
                    except Exception:
                        continue
                out.append({'entity': seg, 'geometry': geom})
            return out

        def _collect_shapes():
            polys = []
            for shp in getattr(ir, 'shapes', []) or []:
                geom = self.to_geometry(shp, coordinate_system, getattr(shp, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == 'Polygon':
                    polys.append({'entity': shp, 'geometry': geom})
            for rect in getattr(ir, 'rectangle_primitives', []) or []:
                geom = self.to_geometry(rect, coordinate_system, getattr(rect, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == 'Polygon':
                    polys.append({'entity': rect, 'geometry': geom})
            return polys

        def _collect_numeric_labels():
            labels = []

            for node in getattr(ir, 'nodes', []) or []:
                raw = getattr(node, 'text', '') or ''
                cleaned = self.clean_latex_text(raw)
                if not cleaned:
                    continue
                if self._classify_label(raw, cleaned) != 'numeric':
                    continue
                geom = self.to_geometry(node, coordinate_system, getattr(node, 'transform', None))
                if geom is None or geom.is_empty:
                    continue
                val = _parse_numeric(cleaned)
                if val is None:
                    continue
                labels.append({'entity': node, 'geometry': geom, 'value': val, 'text': cleaned})

            for seg in getattr(ir, 'line_segments', []) or []:
                seg_geom = self.to_geometry(seg, coordinate_system, getattr(seg, 'transform', None))
                if seg_geom is None or seg_geom.is_empty:
                    continue

                for sub in getattr(seg, 'nodes', []) or []:
                    raw = getattr(sub, 'text', '') or ''
                    cleaned = self.clean_latex_text(raw)
                    if not cleaned or self._classify_label(raw, cleaned) != 'numeric':
                        continue
                    val = _parse_numeric(cleaned)
                    if val is None:
                        continue
                    geom = self.to_geometry(sub, coordinate_system, getattr(sub, 'transform', None))
                    if geom is None or geom.is_empty:
                        mid = seg_geom.interpolate(0.5, normalized=True)
                        geom = Point(float(mid.x), float(mid.y))
                    labels.append({'entity': seg, 'geometry': geom, 'value': val, 'text': cleaned})

                for key in ('label', 'text', 'node_text', 'mid_text', 'annotation'):
                    raw = getattr(seg, key, None)
                    if not raw:
                        continue
                    cleaned = self.clean_latex_text(raw)
                    if self._classify_label(raw, cleaned) != 'numeric':
                        continue
                    val = _parse_numeric(cleaned)
                    if val is None:
                        continue
                    mid = seg_geom.interpolate(0.5, normalized=True)
                    geom = Point(float(mid.x), float(mid.y))
                    labels.append({'entity': seg, 'geometry': geom, 'value': val, 'text': cleaned})

            return labels

        def _associate_labels(numeric_labels, segments, polygons):
            length_pairs, area_pairs = [], []

            SNAP_TO_EDGE_TOL_PT = 12.0
            AREA_INTERIOR_SHRINK_PT = 2.0

            poly_interiors = []
            for poly in polygons:
                interior = poly['geometry'].buffer(-AREA_INTERIOR_SHRINK_PT)
                if interior.is_empty:
                    interior = poly['geometry']
                poly_interiors.append({'geometry': interior, 'entity': poly['entity'], 'orig': poly['geometry']})

            for lab in numeric_labels:
                pt = lab['geometry'].centroid

                nearest_seg = None
                nearest_seg_dist = float('inf')
                for s in segments:
                    d = pt.distance(s['geometry'])
                    if d < nearest_seg_dist:
                        nearest_seg_dist = d
                        nearest_seg = s

                containing = None
                for interior in poly_interiors:
                    if interior['geometry'].covers(pt): 
                        containing = interior
                        break
                
                # feel free to change if you feel like a diff decision boundary is ebtter,
                # but i assume: if very close to edge (inside or outside polygon, then length)
                # if considerably inside polygon, then area
                # else, length to the nearest segment
                if nearest_seg is not None and nearest_seg_dist <= SNAP_TO_EDGE_TOL_PT:
                    length_pairs.append({
                        'label_value': lab['value'],
                        'seg_geom': nearest_seg['geometry'],
                        'seg_id': getattr(nearest_seg['entity'], 'id', None),
                        'label_text': lab['text']
                    })
                elif containing is not None:
                    area_pairs.append({
                        'label_value': lab['value'],
                        'poly_geom': containing['orig'],
                        'poly_id': getattr(containing['entity'], 'id', None),
                        'label_text': lab['text']
                    })
                elif nearest_seg is not None:
                    length_pairs.append({
                        'label_value': lab['value'],
                        'seg_geom': nearest_seg['geometry'],
                        'seg_id': getattr(nearest_seg['entity'], 'id', None),
                        'label_text': lab['text']
                    })

            return length_pairs, area_pairs


        def _check_pairwise_proportions(items, meas_key, label_key, what: str):
            mismatches = []
            n = len(items)
            if n < 2:
                return mismatches
            measured = []
            for it in items:
                if meas_key == 'length':
                    mv = it['seg_geom'].length
                else:
                    mv = it['poly_geom'].area
                measured.append(mv)

            for i in range(n):
                for j in range(i + 1, n):
                    mi, mj = measured[i], measured[j]
                    li, lj = items[i][label_key], items[j][label_key]
                    lhs = mi * lj
                    rhs = mj * li
                    denom = max(abs(rhs), 1e-9)
                    rel_err = abs(lhs - rhs) / denom
                    if rel_err > self.PROPORTION_REL_TOL:
                        a_desc = items[i].get('seg_id' if meas_key == 'length' else 'poly_id')
                        b_desc = items[j].get('seg_id' if meas_key == 'length' else 'poly_id')
                        a_desc = f"id={a_desc}" if a_desc is not None else "(no id)"
                        b_desc = f"id={b_desc}" if b_desc is not None else "(no id)"
                        mismatches.append(
                            f"{what}: ({a_desc}, label={li:g}) vs ({b_desc}, label={lj:g}) "
                            f"→ rel. error {rel_err:.3f}"
                        )
            return mismatches

        segments = _collect_segments()
        polygons = _collect_shapes()
        numeric_labels = _collect_numeric_labels()

        if not numeric_labels:
            return True, "No numeric labels to check"

        length_pairs, area_pairs = _associate_labels(numeric_labels, segments, polygons)

        issues = []
        if len(length_pairs) >= 2:
            issues += _check_pairwise_proportions(
                [{**p, 'length': p['seg_geom'].length, 'label': p['label_value']} for p in length_pairs],
                meas_key='length', label_key='label', what="Length proportion mismatch"
            )

        if len(area_pairs) >= 2:
            issues += _check_pairwise_proportions(
                [{**p, 'area': p['poly_geom'].area, 'label': p['label_value']} for p in area_pairs],
                meas_key='area', label_key='label', what="Area proportion mismatch"
            )

        if issues:
            sample = "; ".join(issues[:3])
            if len(issues) > 3:
                sample += f" (+{len(issues) - 3} more)"
            return False, sample

        checked_groups = (len(length_pairs) >= 2) + (len(area_pairs) >= 2)
        if checked_groups:
            return True, "Labeled lengths/areas match measured proportions"

        return True, "Not enough labeled pairs to compare proportions"
    

    def diagram_elements_dont_problematically_overlap(self, ir) -> tuple[bool, str]:
        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        issues: list[str] = []

        # for 3d overlap checks
        def compute_mean_z(vertices) -> Optional[float]:
            coords_with_z = [coord for coord in vertices if isinstance(coord, (list, tuple)) and len(coord) >= 3]
            if not coords_with_z:
                return None
            return sum(coord[2] for coord in coords_with_z) / len(coords_with_z)

        def compute_normal(vertices) -> Optional[tuple]:
            coords_with_z = [coord for coord in vertices if isinstance(coord, (list, tuple)) and len(coord) >= 3]
            if len(coords_with_z) < 3:
                return None
            v1 = np.array(coords_with_z[1]) - np.array(coords_with_z[0])
            v2 = np.array(coords_with_z[2]) - np.array(coords_with_z[0])
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm == 0:
                return None
            return tuple(normal / norm)

        nodes: list[dict] = []
        for node in getattr(ir, 'nodes', []) or []:
            raw_text = getattr(node, 'text', '') or ''
            cleaned_text = self.clean_latex_text(raw_text)
            if not cleaned_text:
                continue
            transform = getattr(node, 'transform', None)
            geometry = self.to_geometry(node, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            nodes.append({
                'geometry': geometry,
                'entity': node,
                'text': cleaned_text,
                'fill': getattr(node, 'fill', None),
            })

        three_d_faces: list[dict] = []
        unit_cube_faces: list[dict] = []
        unit_cube_records: list[dict] = []
        shape_boundaries: list[dict] = []
        three_d_part_boxes = {}
        three_d_part_overlap_pairs = set()
        for shape in getattr(ir, 'shapes', []) or []:
            transform = getattr(shape, 'transform', None)
            if getattr(shape, 'type', None) == 'Ucube':
                shift_components: list[Optional[float]] = []
                if transform and getattr(transform, 'shift', None):
                    raw_shift = list(transform.shift)
                else:
                    raw_shift = []
                for idx in range(3):
                    value = raw_shift[idx] if idx < len(raw_shift) else 0.0
                    try:
                        shift_components.append(float(value))
                    except (TypeError, ValueError):
                        shift_components.append(None)
                unit_cube_records.append({
                    'id': getattr(shape, 'id', None),
                    'shift': shift_components,
                    'fill': getattr(shape, 'fill', None),
                })
                expanded_faces = self.expand_unit_cube_faces(shape, coordinate_system)
                unit_cube_faces.extend(expanded_faces)
                continue

            geometry = self.to_geometry(shape, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue

            if getattr(shape, 'type', None) == '3D-part':
                vertices = getattr(shape, 'vertices', []) or []
                part_id = getattr(shape, 'id', None)
                transformed_vertices = self.transformed_3d_vertices(vertices, transform)
                if part_id is not None and transformed_vertices:
                    entry = three_d_part_boxes.setdefault(part_id, {
                        'min': [float('inf'), float('inf'), float('inf')],
                        'max': [float('-inf'), float('-inf'), float('-inf')],
                    })
                    for vx, vy, vz in transformed_vertices:
                        entry['min'][0] = min(entry['min'][0], vx)
                        entry['min'][1] = min(entry['min'][1], vy)
                        entry['min'][2] = min(entry['min'][2], vz)
                        entry['max'][0] = max(entry['max'][0], vx)
                        entry['max'][1] = max(entry['max'][1], vy)
                        entry['max'][2] = max(entry['max'][2], vz)
                three_d_faces.append({
                    'geometry': geometry,
                    'entity': shape,
                    'id': getattr(shape, 'id', None),
                    'vertices': vertices,
                    'mean_z': compute_mean_z(vertices),
                    'normal': compute_normal(vertices),
                    'is_cube_face': False,
                    'entity_type': '3D-part',
                })
            else:
                boundary_geometry = geometry.boundary if geometry.geom_type == 'Polygon' else geometry
                if boundary_geometry is not None and not boundary_geometry.is_empty:
                    shape_boundaries.append({'geometry': boundary_geometry, 'entity': shape})

        for cube_face in unit_cube_faces:
            face_geometry = cube_face['geometry']
            parent_cube = cube_face['entity']
            raw_vertices = cube_face.get('raw_vertices') or self.unit_cube_faces(parent_cube)[cube_face['face']]
            three_d_faces.append({
                'geometry': face_geometry,
                'entity': parent_cube,
                'id': cube_face['parent_id'],
                'vertices': raw_vertices,
                'mean_z': compute_mean_z(raw_vertices),
                'normal': compute_normal(raw_vertices),
                'is_cube_face': True,
                'entity_type': 'Ucube',
            })

        for rectangle in getattr(ir, 'rectangle_primitives', []) or []:
            transform = getattr(rectangle, 'transform', None)
            geometry = self.to_geometry(rectangle, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            boundary_geometry = geometry.boundary if geometry.geom_type == 'Polygon' else geometry
            if boundary_geometry is not None and not boundary_geometry.is_empty:
                shape_boundaries.append({'geometry': boundary_geometry, 'entity': rectangle})

        segment_boundaries: list[dict] = []
        solid_segments: list[dict] = []
        for segment in getattr(ir, 'line_segments', []) or []:
            transform = getattr(segment, 'transform', None)
            geometry = self.to_geometry(segment, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            segment_boundaries.append({'geometry': geometry, 'entity': segment})
            style = getattr(segment, 'style', 'solid') or 'solid'
            if style == 'solid':
                endpoints = [getattr(segment, 'from_', []), getattr(segment, 'to', [])]
                solid_segments.append({
                    'geometry': geometry,
                    'entity': segment,
                    'mean_z': compute_mean_z(endpoints),
                })

        arc_boundaries: list[dict] = []
        for arc in getattr(ir, 'arcs', []) or []:
            transform = getattr(arc, 'transform', None)
            center = getattr(arc, 'center', None)
            radius = getattr(arc, 'radius', None)
            start_angle = getattr(arc, 'start_angle', None)
            end_angle = getattr(arc, 'end_angle', None)
            # skip arcs lacking essential geometry (invalid placeholders)
            if center is None or radius is None or start_angle is None or end_angle is None:
                continue
            geometry = self.to_geometry(arc, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            arc_boundaries.append({'geometry': geometry, 'entity': arc})

        arc_anchor_issues = self._arcs_anchor_to_edges(
            ir,
            coordinate_system,
            shape_boundaries,
            segment_boundaries,
        )
        issues.extend(arc_anchor_issues)

        stacked_hollow_locations = []
        cube_stacks: dict[tuple[float, float], list[dict]] = {}
        for cube_record in unit_cube_records:
            shift = cube_record['shift']
            if shift is None or len(shift) < 3:
                continue
            if any(component is None for component in shift[:2]):
                continue
            key = (round(shift[0], 6), round(shift[1], 6))
            cube_stacks.setdefault(key, []).append(cube_record)

        for location, cubes in cube_stacks.items():
            z_values = {cube['shift'][2] for cube in cubes if cube['shift'][2] is not None}
            if len(z_values) <= 1:
                continue
            has_filled_cube = False
            for cube in cubes:
                fill_value = cube.get('fill')
                if fill_value is None:
                    continue
                if isinstance(fill_value, str):
                    if fill_value.strip() and fill_value.strip().lower() != 'none':
                        has_filled_cube = True
                        break
                else:
                    has_filled_cube = True
                    break
            if has_filled_cube:
                continue
            stacked_hollow_locations.append(location)

        if stacked_hollow_locations:
            sample_locations = stacked_hollow_locations[:3]
            locations_text = ", ".join(f"({loc[0]:.2f}, {loc[1]:.2f})" for loc in sample_locations)
            if len(stacked_hollow_locations) > len(sample_locations):
                locations_text += f" (+{len(stacked_hollow_locations) - len(sample_locations)} more)"
            issues.append(
                "Stacked unit cubes without fill detected at "
                f"{locations_text}; front cubes will expose rear edges"
            )

        # Bounding-box based 3D-part overlap detection
        part_items = sorted(three_d_part_boxes.items())
        volume_tolerance = 1e-6
        for index, (first_id, first_box) in enumerate(part_items):
            for second_id, second_box in part_items[index + 1:]:
                overlap_x = min(first_box['max'][0], second_box['max'][0]) - max(first_box['min'][0], second_box['min'][0])
                if overlap_x <= 0:
                    continue
                overlap_y = min(first_box['max'][1], second_box['max'][1]) - max(first_box['min'][1], second_box['min'][1])
                if overlap_y <= 0:
                    continue
                overlap_z = min(first_box['max'][2], second_box['max'][2]) - max(first_box['min'][2], second_box['min'][2])
                if overlap_z <= 0:
                    continue
                overlap_volume = overlap_x * overlap_y * overlap_z
                if overlap_volume <= volume_tolerance:
                    continue
                pair_key = tuple(sorted((first_id, second_id)))
                if pair_key in three_d_part_overlap_pairs:
                    continue
                three_d_part_overlap_pairs.add(pair_key)
                issues.append(
                    f"3D-part solids id={pair_key[0]} and id={pair_key[1]} overlap (~{overlap_volume:.2f} units³)"
                )

        # Node vs node overlaps
        for index, first_node in enumerate(nodes):
            for second_node in nodes[index + 1:]:
                overlap_area = first_node['geometry'].intersection(second_node['geometry']).area
                if overlap_area > self.NODE_OVERLAP_AREA_TOLERANCE_PT2:
                    issues.append(
                        f"Nodes '{first_node['text']}' and '{second_node['text']}' overlap ({overlap_area:.1f} pt²)"
                    )

        # Node vs boundaries (shape edges, line segments, arcs)
        edge_geometries = shape_boundaries + segment_boundaries + arc_boundaries
        for node_record in nodes:
            if node_record['fill']:
                continue
            node_centroid = node_record['geometry'].centroid
            for boundary_record in edge_geometries:
                distance_to_edge = boundary_record['geometry'].distance(node_centroid)
                if distance_to_edge <= self.NODE_EDGE_DISTANCE_TOLERANCE_PT:
                    issues.append(
                        f"Node '{node_record['text']}' sits directly on top of a diagram edge"
                    )
                    break

        # Node vs 3D face boundary overlaps
        for node_record in nodes:
            if node_record['fill']:
                continue
            node_geometry = node_record['geometry']
            for face_record in three_d_faces:
                face_geometry = face_record['geometry']
                if not face_geometry.intersects(node_geometry):
                    continue
                if node_geometry.within(face_geometry):
                    continue
                boundary_overlap_area = face_geometry.intersection(node_geometry).area
                boundary_overlap_length = face_geometry.boundary.intersection(node_geometry).length
                node_area = node_geometry.area
                dynamic_area_limit = self.NODE_FACE_BOUNDARY_OVERLAP_AREA_TOLERANCE_PT2
                if node_area > 0:
                    dynamic_area_limit = max(dynamic_area_limit, node_area / 3.0)

                if (
                    boundary_overlap_area > dynamic_area_limit
                    and boundary_overlap_length > self.NODE_FACE_BOUNDARY_OVERLAP_LENGTH_TOLERANCE_PT
                ):
                    issues.append(
                        f"Node '{node_record['text']}' crosses the boundary of 3D part id={face_record['id']}"
                    )

        # 3D face vs 3D face overlaps across different ids
        for index, first_face in enumerate(three_d_faces):
            for second_face in three_d_faces[index + 1:]:
                entity_type_first = first_face.get('entity_type')
                entity_type_second = second_face.get('entity_type')
                if entity_type_first == 'Ucube' and entity_type_second == 'Ucube':
                    continue
                if entity_type_first == '3D-part' and entity_type_second == '3D-part':
                    continue
                if first_face['id'] == second_face['id']:
                    continue
                overlap_area = first_face['geometry'].intersection(second_face['geometry']).area
                if overlap_area <= self.THREE_D_FACE_OVERLAP_TOLERANCE_PT2:
                    continue
                mean_z_first = first_face.get('mean_z')
                mean_z_second = second_face.get('mean_z')
                if (
                    mean_z_first is not None
                    and mean_z_second is not None
                    and abs(mean_z_first - mean_z_second) > self.THREE_D_FACE_Z_SEPARATION_TOLERANCE
                ):
                    continue
                normal_first = first_face.get('normal')
                normal_second = second_face.get('normal')
                angle_between_normals = None
                if normal_first and normal_second:
                    dot_product = np.clip(np.dot(normal_first, normal_second), -1.0, 1.0)
                    angle_between_normals = np.degrees(np.arccos(dot_product))
                    if angle_between_normals > self.THREE_D_FACE_PARALLEL_ANGLE_TOLERANCE_DEG:
                        continue
                mean_desc_first = f"{mean_z_first:.2f}" if mean_z_first is not None else "unknown"
                mean_desc_second = f"{mean_z_second:.2f}" if mean_z_second is not None else "unknown"
                descriptor = (
                    f"3D parts id={first_face['id']} (mean z={mean_desc_first}) "
                    f"and id={second_face['id']} (mean z={mean_desc_second}) "
                    f"overlap {overlap_area:.1f} pt²"
                )
                if angle_between_normals is not None:
                    descriptor += f"; normals differ by {angle_between_normals:.1f}°"
                issues.append(descriptor)

        # Solid line segments passing through 3D faces
        for segment_record in solid_segments:
            segment_geometry = segment_record['geometry']
            segment_mean_z = segment_record.get('mean_z')
            for face_record in three_d_faces:
                face_geometry = face_record['geometry']
                mean_z_face = face_record.get('mean_z')
                if (
                    segment_mean_z is not None
                    and mean_z_face is not None
                    and abs(segment_mean_z - mean_z_face) > self.THREE_D_FACE_Z_SEPARATION_TOLERANCE
                ):
                    continue
                intersection_geometry = segment_geometry.intersection(face_geometry)
                if intersection_geometry.is_empty:
                    continue
                if intersection_geometry.length <= self.SEGMENT_FACE_OVERLAP_LENGTH_TOLERANCE_PT:
                    continue
                interior_region = face_geometry.buffer(-self.FACE_EDGE_BUFFER_PT)
                if interior_region.is_empty:
                    interior_region = face_geometry
                midpoint = segment_geometry.interpolate(0.5, normalized=True)
                if interior_region.contains(midpoint):
                    issues.append(
                        f"Solid segment overlaps interior of 3D part id={face_record['id']}"
                    )

        if issues:
            summary = "; ".join(issues[:3])
            if len(issues) > 3:
                summary += f" (+{len(issues) - 3} more)"
            return False, summary

        return True, "No problematic overlaps detected"

    def diagram_elements_are_readable_size(self, ir) -> tuple[bool, str]:
        coordinate_system = getattr(ir, 'tikzpicture_options', None)
        canvas_bounds = self._page_canvas if self.enforce_page_bounds else None

        collected_geometries: list[tuple[str, object, object]] = []

        def collect(kind: str, entity, geometry):
            if geometry is None or geometry.is_empty:
                return
            collected_geometries.append((kind, entity, geometry))

        for shape in getattr(ir, 'shapes', []) or []:
            transform = getattr(shape, 'transform', None)
            geometry = self.to_geometry(shape, coordinate_system, transform)
            if getattr(shape, 'type', None) == 'Ucube':
                collect('Ucube', shape, geometry)
            else:
                collect('shape', shape, geometry)

        for rectangle in getattr(ir, 'rectangle_primitives', []) or []:
            transform = getattr(rectangle, 'transform', None)
            geometry = self.to_geometry(rectangle, coordinate_system, transform)
            collect('rectangle', rectangle, geometry)

        for circle in getattr(ir, 'circles', []) or []:
            transform = getattr(circle, 'transform', None)
            geometry = self.to_geometry(circle, coordinate_system, transform)
            collect('circle', circle, geometry)

        for arc in getattr(ir, 'arcs', []) or []:
            transform = getattr(arc, 'transform', None)
            geometry = self.to_geometry(arc, coordinate_system, transform)
            collect('arc', arc, geometry)

        for segment in getattr(ir, 'line_segments', []) or []:
            transform = getattr(segment, 'transform', None)
            geometry = self.to_geometry(segment, coordinate_system, transform)
            collect('line segment', segment, geometry)

        for node in getattr(ir, 'nodes', []) or []:
            transform = getattr(node, 'transform', None)
            cleaned_text = self.clean_latex_text(getattr(node, 'text', '') or '')
            if not cleaned_text:
                continue
            geometry = self.to_geometry(node, coordinate_system, transform)
            collect('node', node, geometry)

        if collected_geometries:
            combined = collected_geometries[0][2]
            for kind, _, geom in collected_geometries[1:]:
                try:
                    combined = combined.union(geom)
                except Exception as exc:
                    warnings.warn(
                        f"Skipping invalid geometry '{kind}' during readability union: {exc}",
                        RuntimeWarning,
                    )
                    continue
            minx, miny, maxx, maxy = combined.bounds
            span_x = maxx - minx
            span_y = maxy - miny
            canvas_min_dim = max(min(span_x, span_y), 0.0)
        elif canvas_bounds is not None:
            minx, miny, maxx, maxy = canvas_bounds.bounds
            span_x = maxx - minx
            span_y = maxy - miny
            canvas_min_dim = max(min(span_x, span_y), 0.0)
        else:
            canvas_min_dim = 0.0

        if canvas_min_dim > 0.0:
            threshold = canvas_min_dim * self.MIN_ELEMENT_DIMENSION_RELATIVE
        else:
            threshold = 0.0

        undersized: list[tuple[str, float]] = []

        def record_issue(kind: str, entity, geometry):
            if geometry is None or geometry.is_empty:
                return
            descriptor = kind
            entity_id = getattr(entity, 'id', None)
            if entity_id is not None:
                descriptor += f" id={entity_id}"

            minx, miny, maxx, maxy = geometry.bounds
            width = maxx - minx
            height = maxy - miny

            geom_type = geometry.geom_type
            if geom_type in ("LineString", "LinearRing"):
                measure = geometry.length
                if measure < threshold:
                    undersized.append((descriptor, measure))
                return

            if hasattr(geometry, 'geoms') and not geometry.is_empty:
                # For Multi* geometries, inspect each part individually.
                for part in geometry.geoms:
                    record_issue(kind, entity, part)
                return

            min_dimension = min(width, height)
            if min_dimension < threshold:
                undersized.append((descriptor, min_dimension))

        for kind, entity, geometry in collected_geometries:
            record_issue(kind, entity, geometry)

        if undersized:
            undersized.sort(key=lambda item: item[1])
            sample = undersized[:3]
            sample_text = "; ".join(
                f"{name} ({measure:.1f}pt)" for name, measure in sample
            )
            if len(undersized) > len(sample):
                sample_text += f" (+{len(undersized) - len(sample)} more)"
            return False, (
                f"Elements below readability threshold {threshold:.1f}pt: {sample_text}"
            )

        return True, f"All elements >= {threshold:.1f}pt"

    def _arcs_anchor_to_edges(self, ir, coordinate_system, shape_boundaries, segment_boundaries) -> list[str]:
        candidate_geometries: list = []

        def collect_geometries(geometry):
            if geometry is None or geometry.is_empty:
                return
            if hasattr(geometry, 'geoms'):
                for part in geometry.geoms:
                    collect_geometries(part)
                return
            candidate_geometries.append(geometry)

        for boundary_record in shape_boundaries:
            collect_geometries(boundary_record.get('geometry'))

        for segment_record in segment_boundaries:
            collect_geometries(segment_record.get('geometry'))

        if not candidate_geometries:
            return []

        tolerance = self.ARC_ENDPOINT_TOLERANCE_PT
        arc_issues: list[str] = []

        for arc in getattr(ir, 'arcs', []) or []:
            transform = getattr(arc, 'transform', None)
            geometry = self.to_geometry(arc, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            coords = list(geometry.coords)
            if len(coords) < 2:
                continue
            endpoints = (coords[0], coords[-1])
            for idx, endpoint in enumerate(endpoints):
                endpoint_point = Point(endpoint)
                min_distance = min(endpoint_point.distance(candidate) for candidate in candidate_geometries)
                if min_distance > tolerance:
                    arc_id = getattr(arc, 'id', None)
                    id_part = f" id={arc_id}" if arc_id is not None else ""
                    endpoint_label = "start" if idx == 0 else "end"
                    issues_message = (
                        f"Arc{id_part} {endpoint_label} endpoint ({endpoint[0]:.1f}, {endpoint[1]:.1f}) "
                        f"is {min_distance:.1f}pt from nearest edge"
                    )
                    arc_issues.append(issues_message)

        return arc_issues

    def shape_outlines_are_closed(self, ir) -> tuple[bool, str]:
        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        issues: list[str] = []

        def vertices_wrap(vertices) -> bool:
            if not vertices or len(vertices) < 2:
                return False
            first = vertices[0]
            last = vertices[-1]
            if len(first) != len(last):
                return False
            closure_tolerance = 1e-4
            return self.distance(first, last) <= closure_tolerance

        three_d_part_planes = {}

        for shape in getattr(ir, 'shapes', []) or []:
            shape_type = getattr(shape, 'type', None)
            if shape_type == 'Ucube':
                continue  # treat cube solids as closed by design

            transform = getattr(shape, 'transform', None)
            cycle_flag = getattr(shape, 'cycle', None)
            vertices = getattr(shape, 'vertices', None)

            entity_id = getattr(shape, 'id', None)
            id_part = f" id={entity_id}" if entity_id is not None else ""

            if shape_type == '3D-part' and entity_id is not None:
                transformed_vertices = self.transformed_3d_vertices(vertices or [], transform)
                if transformed_vertices:
                    entry = three_d_part_planes.setdefault(entity_id, {
                        'x_planes': set(),
                        'y_planes': set(),
                        'z_planes': set(),
                        'x_max': float('-inf'),
                        'x_min': float('inf'),
                        'y_max': float('-inf'),
                        'y_min': float('inf'),
                        'z_max': float('-inf'),
                        'z_min': float('inf'),
                    })
                    xs = [v[0] for v in transformed_vertices]
                    ys = [v[1] for v in transformed_vertices]
                    zs = [v[2] for v in transformed_vertices]
                    tol = 1e-6
                    if xs:
                        entry['x_max'] = max(entry['x_max'], max(xs))
                        entry['x_min'] = min(entry['x_min'], min(xs))
                        if max(xs) - min(xs) <= tol:
                            entry['x_planes'].add(round(xs[0], 6))
                    if ys:
                        entry['y_max'] = max(entry['y_max'], max(ys))
                        entry['y_min'] = min(entry['y_min'], min(ys))
                        if max(ys) - min(ys) <= tol:
                            entry['y_planes'].add(round(ys[0], 6))
                    if zs:
                        entry['z_max'] = max(entry['z_max'], max(zs))
                        entry['z_min'] = min(entry['z_min'], min(zs))
                        if max(zs) - min(zs) <= tol:
                            entry['z_planes'].add(round(zs[0], 6))

            if cycle_flag is not True:
                if vertices_wrap(vertices):
                    continue
                issues.append(f"shape{id_part} ({shape_type}) vertices do not wrap to close the outline")
                continue

            geometry = self.to_geometry(shape, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                issues.append(f"shape{id_part} ({shape_type}) could not produce polygon geometry")
                continue

            if geometry.geom_type != 'Polygon':
                issues.append(f"shape{id_part} ({shape_type}) geometry unexpected type {geometry.geom_type}")
                continue

        plane_tolerance = 1e-4
        for part_id, data in three_d_part_planes.items():
            missing_faces = []
            if not data['z_planes']:
                missing_faces.append('front face')
            if not data['x_planes']:
                missing_faces.append('right face')
            if not data['y_planes']:
                missing_faces.append('top face')
            if missing_faces:
                issues.append(
                    f"3D-part id={part_id} missing {'; '.join(missing_faces)}"
                )
                continue

            y_max = data['y_max']
            if y_max != float('-inf'):
                if all(abs(y_plane - y_max) > plane_tolerance for y_plane in data['y_planes']):
                    issues.append(f"3D-part id={part_id} top face not drawn at max y")

            z_max = data.get('z_max')
            if z_max is not None and z_max != float('-inf'):
                if all(abs(z_plane - z_max) > plane_tolerance for z_plane in data['z_planes']):
                    issues.append(
                        f"3D-part id={part_id} front face not drawn at max z"
                    )

            x_max = data['x_max']
            if x_max != float('-inf'):
                if all(abs(x_plane - x_max) > plane_tolerance for x_plane in data['x_planes']):
                    issues.append(f"3D-part id={part_id} right face not drawn at max x")

        if issues:
            summary = "; ".join(issues[:3])
            if len(issues) > 3:
                summary += f" (+{len(issues) - 3} more)"
            return False, summary

        return True, "All shapes marked closed"

    def angle_labels_matches_arcs(self, ir):
        coordinate_system = getattr(ir, 'tikzpicture_options', None)
        arcs = getattr(ir, 'arcs', []) or []
        rectangles = getattr(ir, 'rectangle_primitives', []) or []

        if not arcs and not rectangles:
            return "N/A", "No angle indicators to check"

        axis_vec_info = self._axes_from_coordinate_system(coordinate_system)
        axis_scales = (axis_vec_info[2], axis_vec_info[3])
        z_vector_base = self._base_z_vector(coordinate_system)

        arc_records = self._build_arc_records(arcs, coordinate_system, axis_scales, z_vector_base)
        angle_labels = self._build_angle_label_records(getattr(ir, 'nodes', []) or [], coordinate_system)

        issues: list[str] = []
        matched_labels = 0

        if angle_labels and not arc_records:
            issues.append("Angle labels present but no arcs were detected")

        for label in angle_labels:
            if not arc_records:
                break
            nearest_arc = min(
                arc_records,
                key=lambda record: label['geometry'].distance(record['geometry'])
            )
            distance_to_arc = label['geometry'].distance(nearest_arc['geometry'])
            tolerance = self._arc_label_tolerance(nearest_arc['geometry'])
            if distance_to_arc > tolerance:
                issues.append(
                    f"Angle label '{label['text']}' is {distance_to_arc:.1f}pt from nearest arc"
                )
                continue

            parsed_angle = self._parse_angle_value(label['text'])
            if parsed_angle is None:
                issues.append(f"Angle label '{label['text']}' does not contain a numeric value")
                continue

            sweep_options = self._compute_arc_sweep_degrees(nearest_arc)
            if sweep_options is None:
                issues.append("Arc geometry could not produce a sweep angle")
                continue

            oriented_sweep, minor_sweep = sweep_options
            candidate_sweeps = [oriented_sweep]
            if minor_sweep is not None:
                candidate_sweeps.append(minor_sweep)
            chosen_sweep = min(candidate_sweeps, key=lambda value: abs(value - parsed_angle))

            if not isclose(chosen_sweep, parsed_angle, rel_tol=self.EPS_ANGLE, abs_tol=self.EPS_ANGLE):
                issues.append(
                    f"Arc id={getattr(nearest_arc['entity'], 'id', 'unknown')} spans {chosen_sweep:.1f}° but label reads {parsed_angle:.1f}°"
                )
                continue

            matched_labels += 1

        right_angle_issues, right_angle_count = self._right_angle_symbol_issues(
            ir,
            coordinate_system,
        )
        issues.extend(right_angle_issues)

        if issues:
            summary = "; ".join(issues[:3])
            if len(issues) > 3:
                summary += f" (+{len(issues) - 3} more)"
            return False, summary

        has_angle_labels = bool(angle_labels)
        has_right_angle_symbols = right_angle_count > 0

        if not has_angle_labels and not has_right_angle_symbols:
            return "N/A", "No angle labels or right-angle symbols to evaluate"

        parts: list[str] = []
        if has_angle_labels:
            parts.append(f"Angle labels match arcs ({matched_labels}/{len(angle_labels)})")
        else:
            parts.append("No angle labels present")
        if has_right_angle_symbols:
            parts.append("Right-angle symbols consistent")
        else:
            parts.append("No right-angle symbols present")
        return True, "; ".join(parts)


    # ----------------------------------------------------------------------------- helpers

    def distance(self, p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        coords1 = list(p1)
        coords2 = list(p2)
        length = min(len(coords1), len(coords2))
        if length == 0:
            return float('inf')
        squared = 0.0
        for idx in range(length):
            squared += (coords1[idx] - coords2[idx]) ** 2
        return squared ** 0.5

    def clean_latex_text(self, text: str) -> str:
        import re

        latex_symbols = {
            r'\\circ': '°',
            r'\\degree': '°',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\pi': 'π',
            r'\\theta': 'θ',
            r'\\lambda': 'λ',
            r'\\mu': 'μ',
            r'\\sigma': 'σ',
            r'\\infty': '∞',
            r'\\pm': '±',
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\div': '÷',
            r'\\approx': '≈',
            r'\\neq': '≠',
            r'\\leq': '≤',
            r'\\geq': '≥',
        }

        sizing_commands = [r'\\tiny', r'\\scriptsize', r'\\footnotesize', r'\\small',
                          r'\\normalsize', r'\\large', r'\\Large', r'\\LARGE',
                          r'\\huge', r'\\Huge']

        cleaned = text

        # convert fractions like \frac{a}{b} to a/b
        cleaned = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", cleaned)

        for cmd in sizing_commands:
            cleaned = re.sub(cmd + r'\s*', '', cleaned)

        cleaned = re.sub(r'\^\\circ', '°', cleaned)
        cleaned = re.sub(r'\^\{\\circ\}', '°', cleaned)

        for latex_cmd, unicode_char in latex_symbols.items():
            cleaned = re.sub(latex_cmd, unicode_char, cleaned)

        cleaned = re.sub(r'\$([^$]*)\$', r'\1', cleaned)

        cleaned = re.sub(r'\^{([^}]*)}', r'^\1', cleaned)
        cleaned = re.sub(r'_{([^}]*)}', r'_\1', cleaned)

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _classify_label(self, raw_text: str, cleaned_text: str) -> str:
        cleaned = cleaned_text or ""
        if not cleaned:
            return 'empty'
        lowered = cleaned.lower()
        if (
            '°' in cleaned
            or re.search(r"\\angle", raw_text)
            or re.search(r"\bangle\b", lowered)
        ):
            return 'angle'
        if re.search(r'\d', cleaned):
            return 'numeric'
        return 'text'

    def _build_arc_records(self, arcs, coordinate_system, axis_scales, z_vector_base):
        records = []
        for arc in arcs:
            center = getattr(arc, 'center', None)
            radius = getattr(arc, 'radius', None)
            start_angle = getattr(arc, 'start_angle', None)
            end_angle = getattr(arc, 'end_angle', None)
            if center is None or radius is None or start_angle is None or end_angle is None:
                continue
            transform = getattr(arc, 'transform', None)
            geometry = self.to_geometry(arc, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            coords = list(geometry.coords)
            if len(coords) < 2:
                continue
            transformed_center = self.apply_transforms([center], coordinate_system, None)[0]
            transformed_center = (transformed_center[0], transformed_center[1])
            if transform:
                transformed_center = self.apply_transform_only(
                    [transformed_center],
                    transform,
                    axis_scales,
                    z_vector_base,
                )[0]
                transformed_center = (transformed_center[0], transformed_center[1])
            start_point = coords[0]
            end_point = coords[-1]
            if len(start_point) > 2:
                start_point = (start_point[0], start_point[1])
            if len(end_point) > 2:
                end_point = (end_point[0], end_point[1])
            records.append({
                'entity': arc,
                'geometry': geometry,
                'center': transformed_center,
                'start': start_point,
                'end': end_point,
            })
        return records

    def _build_angle_label_records(self, nodes, coordinate_system):
        records = []
        for node in nodes:
            raw_text = getattr(node, 'text', '') or ''
            cleaned = self.clean_latex_text(raw_text)
            if not cleaned:
                continue
            if self._classify_label(raw_text, cleaned) != 'angle':
                continue
            transform = getattr(node, 'transform', None)
            geometry = self.to_geometry(node, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            records.append({
                'entity': node,
                'geometry': geometry,
                'text': cleaned,
            })
        return records

    def _arc_label_tolerance(self, arc_geometry) -> float:
        minx, miny, maxx, maxy = arc_geometry.bounds
        width = maxx - minx
        height = maxy - miny
        candidate = self.ANGLE_LABEL_BASE_TOLERANCE_PT
        dimensions = [dimension for dimension in (width, height) if dimension > 0]
        if dimensions:
            candidate = max(candidate, min(dimensions))
        return min(candidate, self.MAX_LABEL_TOLERANCE_PT)

    def _parse_angle_value(self, cleaned_text: str) -> Optional[float]:
        match = re.search(r'[-+]?\d+(?:\.\d+)?', cleaned_text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _compute_arc_sweep_degrees(self, arc_record) -> Optional[tuple[float, Optional[float]]]:
        center = arc_record.get('center')
        start = arc_record.get('start')
        end = arc_record.get('end')
        if center is None or start is None or end is None:
            return None
        start_vector = (start[0] - center[0], start[1] - center[1])
        end_vector = (end[0] - center[0], end[1] - center[1])
        if math.hypot(*start_vector) <= 1e-6 or math.hypot(*end_vector) <= 1e-6:
            return None
        start_angle = math.atan2(start_vector[1], start_vector[0])
        end_angle = math.atan2(end_vector[1], end_vector[0])
        oriented = (end_angle - start_angle) % (2 * math.pi)
        oriented_deg = math.degrees(oriented)
        dot = start_vector[0] * end_vector[0] + start_vector[1] * end_vector[1]
        denom = math.hypot(*start_vector) * math.hypot(*end_vector)
        minor_deg = None
        if denom > 1e-9:
            cos_val = max(-1.0, min(1.0, dot / denom))
            minor_deg = math.degrees(math.acos(cos_val))
        return oriented_deg, minor_deg

    def _collect_diagram_edges(self, ir, coordinate_system):
        edges = []

        def add_edge(start, end, entity):
            if start is None or end is None:
                return
            if len(start) < 2 or len(end) < 2:
                return
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length <= 1e-6:
                return
            edges.append({
                'geometry': LineString([start, end]),
                'start': (start[0], start[1]),
                'end': (end[0], end[1]),
                'entity': entity,
            })

        for segment in getattr(ir, 'line_segments', []) or []:
            transform = getattr(segment, 'transform', None)
            geometry = self.to_geometry(segment, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            coords = list(geometry.coords)
            for idx in range(len(coords) - 1):
                add_edge(coords[idx], coords[idx + 1], segment)

        for shape in getattr(ir, 'shapes', []) or []:
            transform = getattr(shape, 'transform', None)
            geometry = self.to_geometry(shape, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            if geometry.geom_type == 'Polygon':
                coords = list(geometry.exterior.coords)
                for idx in range(len(coords) - 1):
                    add_edge(coords[idx], coords[idx + 1], shape)
            else:
                coords = list(geometry.coords)
                for idx in range(len(coords) - 1):
                    add_edge(coords[idx], coords[idx + 1], shape)

        for rectangle in getattr(ir, 'rectangle_primitives', []) or []:
            if getattr(rectangle, 'is_right_angle_symbol', False):
                continue
            transform = getattr(rectangle, 'transform', None)
            geometry = self.to_geometry(rectangle, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                continue
            coords = list(geometry.exterior.coords)
            for idx in range(len(coords) - 1):
                add_edge(coords[idx], coords[idx + 1], rectangle)

        return edges

    def _right_angle_symbol_issues(self, ir, coordinate_system):
        rectangles = getattr(ir, 'rectangle_primitives', []) or []
        symbols = [rect for rect in rectangles if getattr(rect, 'is_right_angle_symbol', False)]
        if not symbols:
            return [], 0

        edges = self._collect_diagram_edges(ir, coordinate_system)
        issues: list[str] = []
        for symbol in symbols:
            transform = getattr(symbol, 'transform', None)
            geometry = self.to_geometry(symbol, coordinate_system, transform)
            if geometry is None or geometry.is_empty:
                issues.append("Right-angle symbol could not produce geometry")
                continue

            rect_vertices = list(geometry.exterior.coords)[:-1]
            if not rect_vertices:
                issues.append("Right-angle symbol geometry missing vertices")
                continue

            centroid = geometry.centroid
            descriptor = f"Right-angle symbol at ({centroid.x:.1f}, {centroid.y:.1f})"

            corner_point = None
            corner_distance = float('inf')
            for vertex in rect_vertices:
                vertex_point = Point(vertex)
                nearest = min(
                    (vertex_point.distance(edge['geometry']) for edge in edges),
                    default=float('inf')
                )
                if nearest < corner_distance:
                    corner_distance = nearest
                    corner_point = vertex

            if corner_point is None:
                issues.append(f"{descriptor} is not positioned near a diagram vertex")
                continue

            corner_geom = Point(corner_point)

            def edge_direction(edge_record):
                dx = edge_record['end'][0] - edge_record['start'][0]
                dy = edge_record['end'][1] - edge_record['start'][1]
                length = math.hypot(dx, dy)
                if length <= 1e-6:
                    return None
                return (dx / length, dy / length)

            candidate_vectors: list[dict] = []
            for edge in edges:
                direction = edge_direction(edge)
                if direction is None:
                    continue
                distance_to_edge = corner_geom.distance(edge['geometry'])
                if distance_to_edge > self.RIGHT_ANGLE_VERTEX_TOLERANCE_PT:
                    continue
                candidate_vectors.append({
                    'direction': direction,
                    'distance': distance_to_edge,
                    'edge': edge,
                })

            if len(candidate_vectors) < 2:
                issues.append(f"{descriptor} does not align with two diagram edges")
                continue

            candidate_vectors.sort(key=lambda item: item['distance'])

            angle_match = False
            closest_delta = None
            for idx in range(len(candidate_vectors)):
                for jdx in range(idx + 1, len(candidate_vectors)):
                    v1 = candidate_vectors[idx]['direction']
                    v2 = candidate_vectors[jdx]['direction']
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    dot = max(-1.0, min(1.0, dot))
                    angle = math.degrees(math.acos(min(1.0, abs(dot))))
                    delta = abs(90.0 - angle)
                    if closest_delta is None or delta < closest_delta:
                        closest_delta = delta
                    if delta <= self.RIGHT_ANGLE_TOLERANCE_DEG:
                        angle_match = True
                        break
                if angle_match:
                    break

            if not angle_match:
                approx_text = f"{closest_delta:.1f}°" if closest_delta is not None else "unknown"
                issues.append(f"{descriptor} does not correspond to a 90° corner (closest {approx_text})")

        return issues, len(symbols)

    def get_anchor_offset(self, anchor: str, text_width: float, text_height: float) -> tuple[float, float]:
        if anchor == 'mid' or anchor is None:
            return (0, 0)
        elif anchor == 'above':  # anchor=south
            return (0, -text_height / 2)
        elif anchor == 'below':  # anchor=north
            return (0, text_height / 2)
        elif anchor == 'left':   # anchor=east
            return (text_width / 2, 0)
        elif anchor == 'right':  # anchor=west
            return (-text_width / 2, 0)
        elif anchor == 'above left':  # anchor=south east
            return (text_width / 2, -text_height / 2)
        elif anchor == 'above right':  # anchor=south west
            return (-text_width / 2, -text_height / 2)
        elif anchor == 'below left':  # anchor=north east
            return (text_width / 2, text_height / 2)
        elif anchor == 'below right':  # anchor=north west
            return (-text_width / 2, text_height / 2)
        else:
            return (0, 0)

    def get_verticies(self, ir):
        "get all points where segments meet"
        verticies = set()

        if 'segments' in ir:
            for seg in ir['segments']:
                verticies.add(tuple(seg['from']))
                verticies.add(tuple(seg['to']))

        return list(verticies)

    def visualize_ir(self, ir, output_path="debug_visualization.png"):
        """
        Visualize IR and print debugging information about what entities are captured
        """
        import matplotlib.pyplot as plt

        canvas_width_pt = self.PAGE_WIDTH_IN * self.UNIT_TO_PT["in"]
        canvas_height_pt = self.PAGE_HEIGHT_IN * self.UNIT_TO_PT["in"]
        canvas_width_in = self.PAGE_WIDTH_IN
        canvas_height_in = self.PAGE_HEIGHT_IN
        fig, ax = plt.subplots(1, 1, figsize=(canvas_width_in, canvas_height_in))

        coordinate_system = getattr(ir, 'tikzpicture_options', None)

        self._debug("=== IR DEBUGGING INFO ===")
        self._debug(
            f"Page reference dimensions: {canvas_width_pt:.1f} x {canvas_height_pt:.1f} pt "
            f"({canvas_width_in:.1f} x {canvas_height_in:.1f} in)"
        )
        self._debug(f"Coordinate system: {coordinate_system}")

        # draw canvas bounds
        if self.enforce_page_bounds and self._page_canvas is not None:
            x, y = self._page_canvas.exterior.xy
            ax.plot(x, y, 'red', linewidth=2, label='Canvas')

        # draw clips
        clips = getattr(ir, 'clips', []) or []
        for i, clip in enumerate(clips):
            clip_transform = getattr(clip, 'transform', None)
            clip_geom = self.to_geometry(clip, coordinate_system, clip_transform)
            if clip_geom.geom_type == 'Polygon':
                x, y = clip_geom.exterior.xy
                ax.plot(x, y, 'orange', linewidth=2, label=f'Clip {i}')

        entities = [
            ('shapes', getattr(ir, 'shapes', []) or [], 'blue'),
            ('rectangle_primitives', getattr(ir, 'rectangle_primitives', []) or [], 'green'),
            ('circles', getattr(ir, 'circles', []) or [], 'purple'),
            ('line_segments', getattr(ir, 'line_segments', []) or [], 'brown'),
            ('nodes', getattr(ir, 'nodes', []) or [], 'pink'),
            ('arcs', getattr(ir, 'arcs', []) or [], 'cyan'),
        ]

        # print captured entities 
        for entity_type, entity_list, color in entities:
            self._debug(f"{entity_type}: {len(entity_list)} entities")
            if self.debug:
                for i, entity in enumerate(entity_list):
                    self._debug(f"  [{i}] {entity}")

        self._debug("\n=== SHAPELY GEOMETRY CONVERSION ===")
        for entity_type, entity_list, color in entities:
            for j, entity in enumerate(entity_list):
                entity_transform = getattr(entity, 'transform', None)
                try:
                    if getattr(entity, 'type', None) == 'Ucube':
                        expanded_faces = self.expand_unit_cube_faces(entity, coordinate_system)
                        for face_index, face in enumerate(expanded_faces):
                            geom = face['geometry']
                            self._debug(f"{entity_type}[{j}] face[{face_index}] ({face['face']}): {geom.geom_type}")
                            bounds = geom.bounds
                            width_pt = bounds[2] - bounds[0]
                            height_pt = bounds[3] - bounds[1]
                            area_pt2 = geom.area if hasattr(geom, 'area') else 0
                            self._debug(
                                f"  Bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}) to ({bounds[2]:.1f}, {bounds[3]:.1f}) pt"
                            )
                            self._debug(f"  Size: {width_pt:.1f} x {height_pt:.1f} pt")
                            if area_pt2 > 0:
                                self._debug(f"  Area: {area_pt2:.1f} pt²")
                            if geom.geom_type == 'Polygon':
                                x, y = geom.exterior.xy
                                face_alpha = 0.35 if face.get('face') == 'top' else 0.2
                                ax.plot(x, y, color, linewidth=1.1, alpha=0.8)
                                ax.fill(x, y, color, alpha=face_alpha)
                            elif geom.geom_type == 'LineString':
                                x, y = geom.xy
                                ax.plot(x, y, color, linewidth=1)
                        continue

                    geom = self.to_geometry(entity, coordinate_system, entity_transform)

                    # print geometry info
                    bounds = geom.bounds  # (minx, miny, maxx, maxy)
                    width_pt = bounds[2] - bounds[0]
                    height_pt = bounds[3] - bounds[1]
                    area_pt2 = geom.area if hasattr(geom, 'area') else 0

                    self._debug(f"{entity_type}[{j}]: {geom.geom_type}")
                    self._debug(
                        f"  Bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}) to ({bounds[2]:.1f}, {bounds[3]:.1f}) pt"
                    )
                    self._debug(f"  Size: {width_pt:.1f} x {height_pt:.1f} pt")
                    if area_pt2 > 0:
                        self._debug(f"  Area: {area_pt2:.1f} pt²")
                    if entity_type == 'nodes':
                        self._debug(f"  Text: '{entity.text}'")
                    if self.debug:
                        self._debug("")

                    if geom.geom_type == 'Polygon':
                        x, y = geom.exterior.xy
                        ax.plot(x, y, color, linewidth=1, alpha=0.7)
                        ax.fill(x, y, color, alpha=0.2)
                    elif geom.geom_type == 'LineString':
                        x, y = geom.xy
                        ax.plot(x, y, color, linewidth=1)
                    elif geom.geom_type == 'Point':
                        ax.plot(geom.x, geom.y, 'o', color=color, markersize=4)

                except Exception as e:
                    self._debug(f"Error visualizing {entity_type}[{j}]: {e}")

        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('What Evaluator Sees')
        ax.grid(True, alpha=0.3)

        # plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        # print(f"Visualization saved to {output_path}")
        self._debug("=== END IR DEBUGGING INFO ===")

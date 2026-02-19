"""Geometry helper functions extracted from `evaluator.py`.

Each function expects to be bound to an `Evaluator` instance as the first
argument so the evaluator can continue to provide unit conversions, debug
logging, and font utilities without introducing a new abstraction.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPoint


def expand_macros(self, expr: str) -> str:
    """
    Expand IM document class macros in expressions.
    Handles expressions like '-\\HP/2', '\\QP/2', etc.
    """
    if not isinstance(expr, str):
        return expr

    expanded = expr
    for macro, value in self.IM_MACROS.items():
        expanded = expanded.replace(macro, value)

    return expanded


def to_pt(self, value, unit: str = "pt") -> float:
    "convert a value into TeX pt"
    if isinstance(value, str):
        # expand IM macros first
        expanded_value = self.expand_macros(value.strip())

        try:
            return self._eval_simple_expression(expanded_value)
        except ValueError:
            pass

        # Try to evaluate more general arithmetic expressions involving units (e.g. "1in/2 - 0.4cm")
        # handle expressions with division like "-1.625in/2"
        if '/' in expanded_value:
            parts = expanded_value.split('/')
            if len(parts) == 2:
                numerator = parts[0].strip()
                denominator = float(parts[1].strip())

                # handle negative signs
                negative = False
                if numerator.startswith('-'):
                    negative = True
                    numerator = numerator[1:].strip()

                # parse the numerator (should be like "1.625in")
                match = re.match(r"([0-9.]+)\s*(pt|cm|mm|in)", numerator)
                if match:
                    num_value = float(match.group(1))
                    unit = match.group(2)
                    result = (num_value * self.UNIT_TO_PT[unit]) / denominator
                    return -result if negative else result

        # standard unit parsing
        match = re.match(r"(-?[0-9.]+)\s*(pt|cm|mm|in)", expanded_value)
        if not match:
            if isinstance(expanded_value, str) and expanded_value.startswith('\\'):
                warnings.warn(
                    f"Unrecognized unit '{expanded_value}', falling back to 1cm",
                    RuntimeWarning,
                )
                return self.UNIT_TO_PT["cm"]
            raise ValueError(f"Unrecognized unit: {value}")
        value = float(match.group(1))
        unit = match.group(2)
        return float(value) * self.UNIT_TO_PT[unit]
    else:
        return float(value) * self.UNIT_TO_PT[unit]


def _parse_simple_term(self, term: str) -> float:
    term = term.strip()
    if not term:
        raise ValueError("Empty term")

    match = re.fullmatch(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(pt|cm|mm|in)?", term)
    if not match:
        raise ValueError("Unsupported term")

    value = float(match.group(1))
    unit = match.group(2)
    if unit:
        return value * self.UNIT_TO_PT[unit]
    return value


def unit_cube_vertices(self, cube):
    ox, oy, oz = 0.0, 0.0, 0.0
    sx, sy, sz = (float(component) for component in cube.size)
    return [
        (ox, oy, oz),
        (ox + sx, oy, oz),
        (ox + sx, oy + sy, oz),
        (ox, oy + sy, oz),
        (ox, oy, oz + sz),
        (ox + sx, oy, oz + sz),
        (ox + sx, oy + sy, oz + sz),
        (ox, oy + sy, oz + sz),
    ]


def unit_cube_faces(self, cube):
    ox, oy, oz = 0.0, 0.0, 0.0
    sx, sy, sz = (float(component) for component in cube.size)
    return {
        'back': [
            (ox, oy + sy, oz),
            (ox + sx, oy + sy, oz),
            (ox + sx, oy + sy, oz + sz),
            (ox, oy + sy, oz + sz),
        ],
        'right': [
            (ox + sx, oy, oz),
            (ox + sx, oy + sy, oz),
            (ox + sx, oy + sy, oz + sz),
            (ox + sx, oy, oz + sz),
        ],
        'top': [
            (ox, oy, oz + sz),
            (ox + sx, oy, oz + sz),
            (ox + sx, oy + sy, oz + sz),
            (ox, oy + sy, oz + sz),
        ],
    }


def expand_unit_cube_faces(self, cube, coordinate_system):
    faces = []
    raw_faces = self.unit_cube_faces(cube)
    transform = getattr(cube, 'transform', None)
    for face_name, vertices in raw_faces.items():
        projected = self.apply_transforms(vertices, coordinate_system, transform)
        faces.append({
            'geometry': Polygon(projected),
            'entity': cube,
            'face': face_name,
            'parent_id': getattr(cube, 'id', None),
            'raw_vertices': vertices,
        })
    return faces


def _eval_simple_expression(self, expr: str) -> float:
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    tokens = re.split(r'([*/])', expr)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        raise ValueError("Invalid expression")

    value = self._parse_simple_term(tokens[0])
    i = 1
    while i < len(tokens):
        op = tokens[i]
        if op not in ('*', '/'):
            raise ValueError("Unsupported operator")
        if i + 1 >= len(tokens):
            raise ValueError("Trailing operator")
        rhs = self._parse_simple_term(tokens[i + 1])
        if op == '*':
            value *= rhs
        else:
            value /= rhs
        i += 2

    return value


def _shift_to_units(self, value, axis_base_scale):
    """Convert a shift component into coordinate units (dimensionless)."""
    if value is None:
        return 0.0

    if isinstance(value, str):
        if axis_base_scale == 0:
            return 0.0
        return self.to_pt(value) / axis_base_scale

    return float(value)


def _parse_axis_vector(self, spec, default_angle_deg):
    """Parse a TikZ axis specification into a 2D vector (pt) and its magnitude."""
    base_length = float(self.UNIT_TO_PT["cm"])
    angle_deg = float(default_angle_deg)

    if spec is not None:
        value = spec
        if isinstance(value, str):
            token = value.strip()
            if token.startswith('{') and token.endswith('}'):
                token = token[1:-1].strip()
            if token.startswith('(') and token.endswith(')'):
                token = token[1:-1].strip()

            if ':' in token:
                angle_str, length_str = token.split(':', 1)
                try:
                    angle_deg = float(angle_str)
                except ValueError:
                    angle_deg = float(default_angle_deg)
                length_expr = length_str.strip().strip('{}').strip()
                try:
                    base_length = float(self.to_pt(length_expr))
                except ValueError:
                    try:
                        base_length = float(self._eval_simple_expression(length_expr))
                    except Exception:
                        # if still unresolved (e.g. macro like \unitLength), fall back to 1 cm
                        warnings.warn(
                            f"Unsupported axis length expression '{length_expr}', using 1cm fallback",
                            RuntimeWarning,
                        )
                        base_length = float(self.UNIT_TO_PT["cm"])
            elif '*' in token:
                try:
                    base_length = float(self._eval_simple_expression(token))
                except Exception:
                    try:
                        base_length = float(self.to_pt(token))
                    except Exception:
                        warnings.warn(
                            f"Unsupported axis expression '{token}', using 1cm fallback",
                            RuntimeWarning,
                        )
                        base_length = float(self.UNIT_TO_PT["cm"])
            else:
                token_expr = token
                try:
                    base_length = float(self.to_pt(token_expr))
                except ValueError:
                    try:
                        base_length = float(self._eval_simple_expression(token_expr))
                    except Exception:
                        warnings.warn(
                            f"Unsupported axis token '{token_expr}', using 1cm fallback",
                            RuntimeWarning,
                        )
                        base_length = float(self.UNIT_TO_PT["cm"])
        else:
            base_length = float(self.to_pt(value))

    angle_rad = np.radians(angle_deg)
    vec_x = base_length * np.cos(angle_rad)
    vec_y = base_length * np.sin(angle_rad)
    vector = np.array([vec_x, vec_y], dtype=float)
    magnitude = float(np.hypot(vec_x, vec_y))

    return vector, magnitude


def _axes_from_coordinate_system(self, coordinate_system):
    """Return axis vectors (pt) and their magnitudes for x and y."""
    x_spec = getattr(coordinate_system, 'x', None) if coordinate_system else None
    y_spec = getattr(coordinate_system, 'y', None) if coordinate_system else None

    x_vec, x_mag = self._parse_axis_vector(x_spec, 0.0)
    y_vec, y_mag = self._parse_axis_vector(y_spec, 90.0)

    scale_val = getattr(coordinate_system, 'scale', None) if coordinate_system else None
    if scale_val is not None:
        if isinstance(scale_val, list):
            if len(scale_val) >= 1:
                sx = float(scale_val[0])
                x_vec *= sx
                x_mag *= abs(sx)
            if len(scale_val) >= 2:
                sy = float(scale_val[1])
                y_vec *= sy
                y_mag *= abs(sy)
        else:
            factor = float(scale_val)
            x_vec *= factor
            y_vec *= factor
            x_mag *= abs(factor)
            y_mag *= abs(factor)

    return x_vec, y_vec, x_mag, y_mag


def _base_z_vector(self, coordinate_system):
    """Compute the 2D projection vector corresponding to one unit in TikZ's z direction."""
    default_length_pt = 3.85 * self.UNIT_TO_PT["mm"]
    z_x = -0.707 * default_length_pt
    z_y = -0.707 * default_length_pt

    if coordinate_system and getattr(coordinate_system, 'z', None):
        z_spec = getattr(coordinate_system, 'z', None)

        def _convert_component(val):
            if isinstance(val, str):
                try:
                    return self.to_pt(val.strip())
                except Exception:
                    return float(val)
            return float(val)

        if isinstance(z_spec, (list, tuple)):
            if len(z_spec) == 2:
                z_x = _convert_component(z_spec[0])
                z_y = _convert_component(z_spec[1])
            else:
                raise ValueError(f"Invalid z vector list: {z_spec}")
        elif isinstance(z_spec, str):
            token = z_spec.strip()
            # angle:length syntax must be handled before generic pair parsing
            if ':' in token:
                angle_str, length_str = token.strip('()').split(':')
                angle = float(angle_str)
                length_token = length_str.strip()

                if re.match(r".*[a-zA-Z]", length_token):
                    try:
                        length_pt = self.to_pt(length_token)
                    except ValueError:
                        warnings.warn(
                            f"Unsupported z length expression '{length_token}', using 1cm fallback",
                            RuntimeWarning,
                        )
                        length_pt = self.UNIT_TO_PT["cm"]
                else:
                    length_value = self._eval_simple_expression(length_token)
                    coord_scale = self.UNIT_TO_PT["cm"]
                    if getattr(coordinate_system, 'x', None):
                        _, coord_scale = self._parse_axis_vector(coordinate_system.x, 0.0)
                    elif getattr(coordinate_system, 'y', None):
                        _, coord_scale = self._parse_axis_vector(coordinate_system.y, 90.0)
                    length_pt = length_value * coord_scale

                z_x = length_pt * np.cos(np.radians(angle))
                z_y = length_pt * np.sin(np.radians(angle))
            # Handle coordinate pair format: {(x_component, y_component)} or (x_component, y_component)
            elif (token.startswith('{(') and token.endswith(')}')) or (token.startswith('(') and token.endswith(')')):
                inner = token[2:-2] if token.startswith('{(') else token[1:-1]
                parts = inner.split(',')
                if len(parts) == 2:
                    z_x = _convert_component(parts[0])
                    z_y = _convert_component(parts[1])
                else:
                    raise ValueError(f"Invalid z coordinate pair format: {z_spec}")
            else:
                try:
                    length_pt = self.to_pt(token)
                except ValueError:
                    warnings.warn(
                        f"Unsupported z token '{token}', using 1cm fallback",
                        RuntimeWarning,
                    )
                    length_pt = self.UNIT_TO_PT["cm"]
                z_x = length_pt * np.cos(np.radians(45))
                z_y = length_pt * np.sin(np.radians(45))
        else:
            raise ValueError(f"Unsupported z specification type: {type(z_spec)}")

    if coordinate_system and getattr(coordinate_system, 'scale', None):
        scale_val = coordinate_system.scale
        if isinstance(scale_val, list):
            if len(scale_val) >= 3:
                z_scale = float(scale_val[2])
            elif len(scale_val) >= 1:
                z_scale = float(scale_val[0])
            else:
                z_scale = 1.0
        else:
            z_scale = float(scale_val)
        z_x *= z_scale
        z_y *= z_scale

    return (z_x, z_y)


def _dimension_to_pt(self, value):
    """Convert a dimension-like value (e.g., xshift) into pt."""
    if value is None:
        return 0.0

    if isinstance(value, str):
        return self.to_pt(value)

    return float(value)


def build_transformation_matrix(self, coordinate_system, transform):
    """
    Build a 2D transformation matrix that matches TikZ's composition order.
    Returns the matrix along with metadata about axis bases and the z projection vector.
    """
    matrix = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    x_vec, y_vec, axis_base_x, axis_base_y = self._axes_from_coordinate_system(coordinate_system)
    z_vector_base = self._base_z_vector(coordinate_system)

    axis_matrix = np.array([[x_vec[0], y_vec[0], 0.0],
                            [x_vec[1], y_vec[1], 0.0],
                            [0.0, 0.0, 1.0]], dtype=float)
    matrix = axis_matrix @ matrix

    scale_factor_x = 1.0
    scale_factor_y = 1.0
    scale_factor_z = 1.0

    if transform:
        if getattr(transform, 'scale', None):
            if isinstance(transform.scale, list):
                sx = float(transform.scale[0])
                sy = float(transform.scale[1]) if len(transform.scale) >= 2 else sx
                if len(transform.scale) >= 3:
                    scale_factor_z *= float(transform.scale[2])
                else:
                    scale_factor_z *= sx
            else:
                sx = sy = float(transform.scale)
                scale_factor_z *= sx
            scale_matrix = np.array([[sx, 0.0, 0.0],
                                    [0.0, sy, 0.0],
                                    [0.0, 0.0, 1.0]], dtype=float)
            matrix = scale_matrix @ matrix
            scale_factor_x *= sx
            scale_factor_y *= sy

        if getattr(transform, 'rotate', None):
            angle_rad = np.radians(transform.rotate)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a, 0.0],
                                        [sin_a, cos_a, 0.0],
                                        [0.0, 0.0, 1.0]], dtype=float)
            matrix = rotation_matrix @ matrix

        linear_part = matrix[:2, :2].copy()

        shift_units_x = 0.0
        shift_units_y = 0.0
        shift_units_z = 0.0

        if getattr(transform, 'shift', None):
            shift_vals = transform.shift
            if len(shift_vals) >= 1:
                shift_units_x += self._shift_to_units(shift_vals[0], axis_base_x)
            if len(shift_vals) >= 2:
                shift_units_y += self._shift_to_units(shift_vals[1], axis_base_y)
            if len(shift_vals) >= 3:
                shift_units_z += self._shift_to_units(shift_vals[2], axis_base_x)

        if getattr(transform, 'xshift', None):
            if axis_base_x != 0.0:
                shift_units_x += self._dimension_to_pt(transform.xshift) / axis_base_x

        if getattr(transform, 'yshift', None):
            if axis_base_y != 0.0:
                shift_units_y += self._dimension_to_pt(transform.yshift) / axis_base_y

        delta_xy = linear_part @ np.array([shift_units_x, shift_units_y], dtype=float)

        z_vector_scaled = np.array(z_vector_base, dtype=float) * scale_factor_z
        if getattr(transform, 'rotate', None):
            angle_rad = np.radians(transform.rotate)
            rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]], dtype=float)
            z_rot = rot @ z_vector_scaled
        else:
            z_rot = np.array(z_vector_scaled, dtype=float)

        total_shift = delta_xy + shift_units_z * z_rot

        if not np.allclose(total_shift, 0.0):
            shift_matrix = np.array([[1.0, 0.0, total_shift[0]],
                                    [0.0, 1.0, total_shift[1]],
                                    [0.0, 0.0, 1.0]], dtype=float)
            matrix = shift_matrix @ matrix

    meta = {
        'axis_base': (axis_base_x, axis_base_y),
        'scale_factors': (scale_factor_x, scale_factor_y, scale_factor_z),
        'z_vector_base': z_vector_base,
    }

    return matrix, meta


def apply_matrix_transform(self, coords, matrix):
    """
    Apply a 3x3 transformation matrix to a list of 2D coordinates.
    """
    out = []
    for point in coords:
        if len(point) == 2:
            x, y = point
            # convert to homogeneous coordinates
            homogeneous = np.array([x, y, 1.0])
            # apply transformation
            transformed = matrix @ homogeneous
            # convert back to 2D
            out.append((transformed[0], transformed[1]))
        else:
            # 3D coordinates - apply to x,y and keep z unchanged
            x, y, z = point
            homogeneous = np.array([x, y, 1.0])
            transformed = matrix @ homogeneous
            out.append((transformed[0], transformed[1], z))
    return out


def apply_transforms(self, coords, coordinate_system, transform):
    """
    Apply transformations using TikZ-style matrix composition.
    coords: list of (x, y) or (x, y, z) tuples
    coordinate_system: the tikzpicture params (scale, units, etc.)
    transform: the transform to apply to the coords (shift, xshift, yshift, scale, rotate)
    """
    if transform:
        self._debug(f"apply_transforms: transform={transform}")
    if coordinate_system:
        self._debug(f"apply_transforms: coordinate_system={coordinate_system}")

    # expand macros in coordinates if they are strings
    expanded_coords = []
    for coord in coords:
        if isinstance(coord, (list, tuple)):
            # Some malformed IR entries wrap coordinates inside an extra list layer (or even a
            # list of multiple identical coordinates). Flatten those cases so downstream math stays sane.
            if coord and all(isinstance(item, (list, tuple)) for item in coord):
                candidate = coord[0]
                if 2 <= len(candidate) <= 3 and all(isinstance(v, (int, float, str)) for v in candidate):
                    coord = candidate
            expanded_coord = []
            for val in coord:
                if isinstance(val, str):
                    # try to evaluate macro expressions like "-\\HP/2"
                    try:
                        expanded_val = self.expand_macros(val)
                        # use to_pt to parse and evaluate the expression
                        expanded_coord.append(self.to_pt(expanded_val))
                    except Exception:
                        expanded_coord.append(float(val))
                else:
                    expanded_coord.append(val)
            expanded_coords.append(tuple(expanded_coord))
        else:
            expanded_coords.append(coord)

    # build transformation matrix
    matrix, meta = self.build_transformation_matrix(coordinate_system, transform)
    self._debug(f"transformation matrix:\n{matrix}")

    # apply matrix to coordinates
    out = self.apply_matrix_transform(expanded_coords, matrix)

    # project 3d to 2d if any 3d points are present
    if any(len(point) == 3 for point in coords):
        coords_3d = [(point[0], point[1], point[2]) if len(point) == 3 else (point[0], point[1], 0) for point in out]
        out = self.project_3d_to_2d(coords_3d, coordinate_system, meta)

    return out


def apply_transform_only(self, coords, transform, axis_scales=None, z_vector_base=None):
    """
    Apply only entity transforms (rotation, scaling, shifts) without coordinate system scaling.
    Used for second-stage processing of already-scaled coordinates.
    """
    if transform:
        self._debug(f"apply_transform_only: transform={transform}")

    if axis_scales is None:
        axis_scales = (self.UNIT_TO_PT["cm"], self.UNIT_TO_PT["cm"])

    axis_base_x, axis_base_y = axis_scales

    if z_vector_base is None:
        z_vector_base = self._base_z_vector(None)

    matrix = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    scale_factor_x = 1.0
    scale_factor_y = 1.0
    scale_factor_z = 1.0

    if transform:
        if getattr(transform, 'scale', None):
            if isinstance(transform.scale, list):
                sx = float(transform.scale[0])
                sy = float(transform.scale[1]) if len(transform.scale) >= 2 else sx
                if len(transform.scale) >= 3:
                    scale_factor_z *= float(transform.scale[2])
                else:
                    scale_factor_z *= sx
            else:
                sx = sy = float(transform.scale)
                scale_factor_z *= sx
            scale_matrix = np.array([[sx, 0.0, 0.0],
                                    [0.0, sy, 0.0],
                                    [0.0, 0.0, 1.0]], dtype=float)
            matrix = scale_matrix @ matrix
            scale_factor_x *= sx
            scale_factor_y *= sy

        if getattr(transform, 'rotate', None):
            angle_rad = np.radians(transform.rotate)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a, 0.0],
                                        [sin_a, cos_a, 0.0],
                                        [0.0, 0.0, 1.0]], dtype=float)
            matrix = rotation_matrix @ matrix

        linear_part = matrix[:2, :2].copy()

        shift_units_x = 0.0
        shift_units_y = 0.0
        shift_units_z = 0.0

        if getattr(transform, 'shift', None):
            shift_vals = transform.shift
            if len(shift_vals) >= 1:
                shift_units_x += self._shift_to_units(shift_vals[0], axis_base_x)
            if len(shift_vals) >= 2:
                shift_units_y += self._shift_to_units(shift_vals[1], axis_base_y)
            if len(shift_vals) >= 3:
                shift_units_z += self._shift_to_units(shift_vals[2], axis_base_x)

        if getattr(transform, 'xshift', None):
            if axis_base_x != 0.0:
                shift_units_x += self._dimension_to_pt(transform.xshift) / axis_base_x

        if getattr(transform, 'yshift', None):
            if axis_base_y != 0.0:
                shift_units_y += self._dimension_to_pt(transform.yshift) / axis_base_y

        shift_local_xy = np.array([shift_units_x * axis_base_x, shift_units_y * axis_base_y], dtype=float)
        delta_xy = linear_part @ shift_local_xy

        z_vector_scaled = np.array(z_vector_base, dtype=float) * scale_factor_z
        if getattr(transform, 'rotate', None):
            angle_rad = np.radians(transform.rotate)
            rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]], dtype=float)
            z_rot = rot @ z_vector_scaled
        else:
            z_rot = np.array(z_vector_scaled, dtype=float)

        total_shift = delta_xy + shift_units_z * z_rot

        if not np.allclose(total_shift, 0.0):
            shift_matrix = np.array([[1.0, 0.0, total_shift[0]],
                                    [0.0, 1.0, total_shift[1]],
                                    [0.0, 0.0, 1.0]], dtype=float)
            matrix = shift_matrix @ matrix

    return self.apply_matrix_transform(coords, matrix)


def generate_arc_geometry(self, center, start_angle, end_angle, radius, num_segments=100):
    # generates a shapely geometry for an arc
    center_x, center_y = center
    theta = np.radians(np.linspace(start_angle, end_angle, num_segments))

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)

    return LineString(np.column_stack((x, y)))


def project_3d_to_2d(self, coords_3d, coordinate_system, meta=None):
    if meta is not None:
        z_vector_base = meta.get('z_vector_base', self._base_z_vector(coordinate_system))
        scale_factors = meta.get('scale_factors', (1.0, 1.0, 1.0))
        scale_factor_z = scale_factors[2]
    else:
        z_vector_base = self._base_z_vector(coordinate_system)
        scale_factor_z = 1.0

    z_x = z_vector_base[0] * scale_factor_z
    z_y = z_vector_base[1] * scale_factor_z

    projected = []
    for x, y, z in coords_3d:
        proj_x = x + z * z_x
        proj_y = y + z * z_y
        projected.append((proj_x, proj_y))
    return projected


def rotate_point_around_center(self, point: tuple[float, float], angle_deg: float, center: tuple[float, float]) -> tuple[float, float]:
    """Rotate a point around a center point by angle_deg degrees"""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # translate to origin
    x, y = point[0] - center[0], point[1] - center[1]

    # rotate
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    # translate back
    return (x_rot + center[0], y_rot + center[1])


def to_geometry(self, entity, coordinate_system=None, transform=None):
    "convert an entity to a shapely geometry w/ tikzpicture options and transforms applied"

    axis_vec_info = self._axes_from_coordinate_system(coordinate_system)
    axis_scales = (axis_vec_info[2], axis_vec_info[3])
    z_vector_base = self._base_z_vector(coordinate_system)

    # handle unit cubes
    if getattr(entity, 'type', None) == 'Ucube':
        vertices = self.unit_cube_vertices(entity)
        projected_vertices = self.apply_transforms(vertices, coordinate_system, transform)
        return MultiPoint(projected_vertices).convex_hull

    # handle clips
    if getattr(entity, 'type', None) == "rectangle":
        def _normalize_clip_corner(corner):
            if isinstance(corner, str):
                if corner in self.IM_MACROS:
                    macro_value = self.to_pt(self.IM_MACROS[corner])
                    return [macro_value, macro_value]
                token = corner.strip()
                if token.startswith('(') and token.endswith(')'):
                    parts = token[1:-1].split(',')
                    if len(parts) == 2:
                        return [self.to_pt(parts[0].strip()), self.to_pt(parts[1].strip())]
                raise ValueError(f"Unknown IM macro: {corner}")
            return corner

        # resolve IM macros if present
        corner1 = entity.corner1
        corner2 = entity.corner2
        try:
            corner1 = _normalize_clip_corner(corner1)
            corner2 = _normalize_clip_corner(corner2)
        except ValueError as exc:
            warnings.warn(
                f"Skipping clip due to unparseable corner: {exc}",
                RuntimeWarning,
            )
            return None

        coords = [corner1, corner2]
        coords = self.apply_transforms(coords, coordinate_system, transform)
        (x1, y1), (x2, y2) = coords
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    # rectangle_primitives
    elif hasattr(entity, 'corner1') and hasattr(entity, 'corner2'):
        self._debug(f"rectangle before transforms: corner1={entity.corner1}, corner2={entity.corner2}")
        self._debug(f"rectangle transform object: {transform}")

        # first apply unit conversion
        coords = [entity.corner1, entity.corner2]
        coords_with_units = self.apply_transforms(coords, coordinate_system, None)
        (x1, y1), (x2, y2) = coords_with_units

        self._debug(f"rectangle after unit conversion: ({x1}, {y1}), ({x2}, {y2})")

        # build rectangle
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        all_corners = [
            (min_x, min_y),  # bottom-left
            (max_x, min_y),  # bottom-right
            (max_x, max_y),  # top-right
            (min_x, max_y),  # top-left
        ]

        self._debug(f"rectangle corners before rotation: {all_corners}")
        original_size = (max_x - min_x, max_y - min_y)
        self._debug(f"original rectangle size: {original_size[0]:.1f} x {original_size[1]:.1f} pt")

        # now apply any remaining transforms (rotation, shifts)
        if transform:
            all_corners = self.apply_transform_only(all_corners, transform, axis_scales, z_vector_base)

        self._debug(f"rectangle corners after all transforms: {all_corners}")

        return Polygon(all_corners)

    # shapes
    elif hasattr(entity, 'vertices'):
        # Stage 1: apply coordinate system transforms only
        coords_with_units = self.apply_transforms(entity.vertices, coordinate_system, None)

        # Stage 2: apply entity transforms if present
        if transform:
            coords_with_units = self.apply_transform_only(coords_with_units, transform, axis_scales, z_vector_base)

        if getattr(entity, 'cycle', False):
            return Polygon(coords_with_units)
        else:
            return LineString(coords_with_units)  # non-closed polygons are just lines

    # arcs
    elif hasattr(entity, 'center') and hasattr(entity, 'start_angle'):
        # Stage 1: apply coordinate system transforms to center
        center_with_units = self.apply_transforms([entity.center], coordinate_system, None)[0]

        # Stage 1: scale radius by coordinate system only
        radius = entity.radius
        if isinstance(radius, str):
            radius_expr = self.expand_macros(radius)
            try:
                radius = self.to_pt(radius_expr)
            except Exception:
                radius = float(self._eval_simple_expression(radius_expr))
        axis_mag_x, axis_mag_y = axis_scales
        radius *= (axis_mag_x + axis_mag_y) / 2.0

        # Stage 2: apply entity transforms
        if transform:
            center_transformed = self.apply_transform_only([center_with_units], transform, axis_scales, z_vector_base)[0]

            if getattr(transform, 'scale', None):
                if isinstance(transform.scale, list):
                    radius *= (transform.scale[0] + transform.scale[1]) / 2
                else:
                    radius *= transform.scale
        else:
            center_transformed = center_with_units

        return self.generate_arc_geometry(center_transformed, entity.start_angle, entity.end_angle, radius)

    # circles
    elif hasattr(entity, 'center') and hasattr(entity, 'radius'):
        # Stage 1: apply coordinate system transforms to center
        center_with_units = self.apply_transforms([entity.center], coordinate_system, None)[0]

        # Stage 1: scale radius by coordinate system only
        radius = entity.radius
        if isinstance(radius, str):
            radius_expr = self.expand_macros(radius)
            try:
                radius = self.to_pt(radius_expr)
            except Exception:
                radius = float(self._eval_simple_expression(radius_expr))
        axis_mag_x, axis_mag_y = axis_scales
        radius *= (axis_mag_x + axis_mag_y) / 2.0

        # Stage 2: apply entity transforms
        if transform:
            center_transformed = self.apply_transform_only([center_with_units], transform, axis_scales, z_vector_base)[0]

            if getattr(transform, 'scale', None):
                if isinstance(transform.scale, list):
                    radius *= (transform.scale[0] + transform.scale[1]) / 2
                else:
                    radius *= transform.scale
        else:
            center_transformed = center_with_units

        return Point(center_transformed).buffer(radius)

    # line_segments
    elif hasattr(entity, 'from_') and hasattr(entity, 'to'):
        # Stage 1: apply coordinate system transforms only
        coords_with_units = self.apply_transforms([entity.from_, entity.to], coordinate_system, None)

        # Stage 2: apply entity transforms if present
        if transform:
            coords_with_units = self.apply_transform_only(coords_with_units, transform, axis_scales, z_vector_base)

        return LineString(coords_with_units)

    # nodes
    elif hasattr(entity, 'at') and hasattr(entity, 'text'):
        anchor_label = getattr(entity, 'anchor', 'center')
        self._debug(f"node before transforms: at={entity.at}, text='{entity.text}', anchor={anchor_label}")

        node_scope_transform = transform
        node_local_rotation = getattr(entity, 'node_rotate', None)

        self._debug(f"scope transform (applied to coordinates): {node_scope_transform}")
        self._debug(f"node rotate (local): {node_local_rotation}")

        # Stage 1: apply coordinate system transforms to 'at' coordinate
        at_with_units = self.apply_transforms([entity.at], coordinate_system, None)[0]

        # Stage 2: apply scope transforms to 'at' coordinate
        if node_scope_transform:
            transformed_at = self.apply_transform_only([at_with_units], node_scope_transform, axis_scales, z_vector_base)[0]
        else:
            transformed_at = at_with_units

        self._debug(f"transformed 'at' coordinate: {transformed_at}")

        # calculate text dimensions
        font = self._load_font()
        cleaned_text = self.clean_latex_text(entity.text)
        self._debug(f"cleaned text: '{entity.text}' → '{cleaned_text}'")
        xmin, ymin, xmax, ymax = font.getbbox(cleaned_text)
        text_width = self.to_pt(xmax - xmin, "px")
        text_height = self.to_pt(ymax - ymin, "px")
        self._debug(f"text dimensions: {text_width:.1f} x {text_height:.1f} pt")

        # position node so anchor sits on transformed_at
        anchor_offset = self.get_anchor_offset(anchor_label, text_width, text_height)
        node_center = (transformed_at[0] - anchor_offset[0], transformed_at[1] - anchor_offset[1])
        self._debug(f"anchor '{anchor_label}' TikZ offset: {anchor_offset}, node center: {node_center}")

        # build bounding box
        bbox_corners = [
            (node_center[0] - text_width / 2, node_center[1] - text_height / 2),  # bottom-left
            (node_center[0] + text_width / 2, node_center[1] - text_height / 2),  # bottom-right
            (node_center[0] + text_width / 2, node_center[1] + text_height / 2),  # top-right
            (node_center[0] - text_width / 2, node_center[1] + text_height / 2),  # top-left
        ]

        # apply rotate around anchor
        if node_local_rotation is not None:
            self._debug(f"applying node rotation {node_local_rotation}° around anchor at {transformed_at}")
            final_corners = [
                self.rotate_point_around_center(corner, node_local_rotation, transformed_at)
                for corner in bbox_corners
            ]
        else:
            final_corners = bbox_corners

        self._debug(f"final text bounding box: {final_corners}")
        return Polygon(final_corners)

    else:
        raise ValueError(f"Unrecognized entity type: {entity.type}")


def _to_float_default(self, value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_transform_scale(self, transform):
    if transform is None or getattr(transform, 'scale', None) is None:
        return (1.0, 1.0, 1.0)
    scale_field = transform.scale
    if isinstance(scale_field, list):
        sx = self._to_float_default(scale_field[0], 1.0)
        sy = self._to_float_default(scale_field[1], sx) if len(scale_field) >= 2 else sx
        sz = self._to_float_default(scale_field[2], sx) if len(scale_field) >= 3 else sx
    else:
        sx = sy = sz = self._to_float_default(scale_field, 1.0)
    return (sx, sy, sz)


def _resolve_transform_shift(self, transform):
    shift = [0.0, 0.0, 0.0]
    if transform and getattr(transform, 'shift', None):
        for idx, val in enumerate(transform.shift[:3]):
            shift[idx] = self._to_float_default(val, 0.0)
    return tuple(shift)


def transformed_3d_vertices(self, vertices, transform):
    if not vertices:
        return []
    sx, sy, sz = self._resolve_transform_scale(transform)
    shift_x, shift_y, shift_z = self._resolve_transform_shift(transform)
    transformed = []
    for vertex in vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) < 3:
            continue
        vx = self._to_float_default(vertex[0], 0.0) * sx + shift_x
        vy = self._to_float_default(vertex[1], 0.0) * sy + shift_y
        vz = self._to_float_default(vertex[2], 0.0) * sz + shift_z
        transformed.append((vx, vy, vz))
    return transformed

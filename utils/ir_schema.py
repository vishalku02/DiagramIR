from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict, conlist

# Reusable types
Coord2D = conlist(float, min_length=2, max_length=2)
Coord3D = conlist(float, min_length=3, max_length=3)
Coord2Dor3D = Union[Coord2D, Coord3D] # [x, y] or [x, y, z]
ScaleType = Union[float, conlist(float, min_length=2, max_length=3)] # number or [sx, sy, (sz)]

class CoordinateSystem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    scale: Optional[ScaleType] = None


class Transform(BaseModel):
    model_config = ConfigDict(extra='forbid')
    shift: Optional[Coord2Dor3D] = None       
    scale: Optional[ScaleType] = None         
    xshift: Optional[str] = None
    yshift: Optional[str] = None
    rotate: Optional[float] = None    

class Clip(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['rectangle'] # extend to other clip shapes later if needed
    corner1: Coord2D
    corner2: Coord2D
    transform: Optional[Transform] = None

class Shape2D(BaseModel):
    model_config = ConfigDict(extra='forbid')
    vertices: List[Coord2D]
    type: Literal['triangle', 'polygon']
    cycle: bool
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class Shape3DPart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['3D-part']
    id: int
    vertices: List[Coord3D]
    cycle: bool
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class Ucube(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['Ucube']
    id: int
    size: Coord3D
    transform: Optional[Transform] = None
    fill: Optional[str] = None


Shapes = Annotated[Union[Shape2D, Shape3DPart, Ucube], Field(discriminator='type')]

class RectanglePrimitive(BaseModel):
    model_config = ConfigDict(extra='forbid')
    corner1: Coord2D
    corner2: Coord2D
    is_right_angle_symbol: bool
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class LineSegment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    from_: Coord2Dor3D = Field(alias='from')
    to: Coord2Dor3D
    style: Optional[Literal['solid', 'dashed', 'dotted']] = 'solid'
    text: Optional[str] = None
    transform: Optional[Transform] = None

class Node(BaseModel):
    model_config = ConfigDict(extra='forbid')
    at: Coord2Dor3D
    text: str
    anchor: Optional[Literal[
        'mid', 
        'above', 'below', 'left', 'right',
        'above left', 'above right', 'below left', 'below right'
    ]] = None
    node_rotate: Optional[float] = None
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class Circle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    center: Coord2D
    radius: float
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class Arc(BaseModel):
    model_config = ConfigDict(extra='forbid')
    center: Coord2Dor3D
    start_angle: float
    end_angle: float
    radius: float
    transform: Optional[Transform] = None
    fill: Optional[str] = None

class TikzIR(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tikzpicture_options: Optional[CoordinateSystem] = None
    clips: Optional[List[Clip]] = None
    shapes: Optional[List[Shapes]] = None
    line_segments: Optional[List[LineSegment]] = None
    nodes: Optional[List[Node]] = None
    circles: Optional[List[Circle]] = None
    rectangle_primitives: Optional[List[RectanglePrimitive]] = None
    arcs: Optional[List[Arc]] = None

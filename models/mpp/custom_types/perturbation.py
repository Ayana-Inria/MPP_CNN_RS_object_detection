from dataclasses import dataclass
from typing import Type, Union, List, Optional, Dict, Any

from base.shapes.base_shapes import Point


@dataclass
class Perturbation:
    type: Type
    removal: Union[None, Point, List[Point]] = None
    addition: Union[None, Point, List[Point]] = None
    data: Optional[Dict[str, Any]] = None
   
from __future__ import annotations
from dataclasses import dataclass
import threading
from geowave.workflow.recipe import Recipe

@dataclass(frozen=True)
class RecipeRevision:
    recipe_id: str
    revision: int
    recipe: Recipe
    active: bool

class RecipeCatalog:

    def __init__(self):
        self._items: dict[str, list[RecipeRevision]] = {}
        self._lock = threading.RLock()

    def publish(self, recipe_id: str, recipe: Recipe) -> RecipeRevision:
        with self._lock:
            versions = self._items.setdefault(recipe_id, [])
            revision = len(versions) + 1
            item = RecipeRevision(recipe_id, revision, recipe, True)
            versions.append(item)
            return item

    def latest(self, recipe_id: str) -> RecipeRevision | None:
        with self._lock:
            versions = self._items.get(recipe_id, [])
            return versions[-1] if versions else None

    def get(self, recipe_id: str, revision: int) -> RecipeRevision | None:
        with self._lock:
            for item in self._items.get(recipe_id, []):
                if item.revision == revision:
                    return item
        return None

    def history(self, recipe_id: str) -> tuple[RecipeRevision, ...]:
        with self._lock:
            return tuple(self._items.get(recipe_id, []))

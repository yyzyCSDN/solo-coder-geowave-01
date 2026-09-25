from __future__ import annotations
from dataclasses import dataclass
from geowave.workflow.recipe import Recipe
from geowave.workflow.compare import compare_recipes, potentially_breaking

@dataclass(frozen=True)
class RecipeRelease:
    name: str
    major: int
    minor: int
    patch: int
    recipe: Recipe

    @property
    def version(self) -> str:
        return f'{self.major}.{self.minor}.{self.patch}'

def next_release(previous: RecipeRelease | None, recipe: Recipe) -> RecipeRelease:
    if previous is None:
        return RecipeRelease(recipe.name, 1, 0, 0, recipe)
    changes = compare_recipes(previous.recipe, recipe)
    if not changes:
        return RecipeRelease(recipe.name, previous.major, previous.minor, previous.patch + 1, recipe)
    if potentially_breaking(changes):
        return RecipeRelease(recipe.name, previous.major + 1, 0, 0, recipe)
    return RecipeRelease(recipe.name, previous.major, previous.minor + 1, 0, recipe)

def compatible(required_major: int, release: RecipeRelease) -> bool:
    return release.major == required_major

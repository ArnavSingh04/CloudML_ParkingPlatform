"""Car-park registry.

Owns the static catalogue of configured car parks and how each maps to its
camera endpoint. Car parks are deterministic (``CBD_001`` .. ``CBD_0NN``) so the
API service and the camera simulator agree on identifiers without any shared
state.

The ``CBD_nnn`` id format and the human-readable street names mirror the
assignment's COREAPI1 example output (§4.1: ``CBD_001``, ``CBD_042`` /
"Market Street East").
"""

from __future__ import annotations

import random

from ..models.schemas import CarParkInfo

# Id format. Must stay in sync with camera_service.main (which builds the same
# ids independently — the two services share no code).
CARPARK_ID_PREFIX = "CBD"
CARPARK_ID_DIGITS = 3

# Street names for the simulated CBD. Cycled when more car parks are configured
# than there are names, with a numeric suffix to keep every name unique.
_STREET_NAMES: tuple[str, ...] = (
    "Market Street East",
    "Collins Street West",
    "Flinders Lane North",
    "Bourke Street Mall",
    "Queen Street Central",
    "Elizabeth Street South",
    "Little Lonsdale Lane",
    "Spencer Street Terminal",
    "King Street Riverside",
    "William Street Courts",
    "Exhibition Street North",
    "Russell Street Precinct",
    "Swanston Street Civic",
    "La Trobe Street East",
    "Franklin Street Depot",
    "Latrobe Terrace Annex",
    "Harbour Esplanade",
    "Docklands Quay",
    "Southbank Promenade",
    "Federation Wharf",
    "Carlton Gardens Edge",
    "Chinatown Laneway",
    "Parliament Hill",
    "Treasury Gardens West",
    "Batman Park",
    "Sandridge Crossing",
    "Yarra Bank",
    "Rialto Underground",
    "Emporium Rooftop",
    "State Library Square",
    "Melbourne Central Loop",
    "Birrarung Marr",
    "Olympic Boulevard",
)


def carpark_id(index: int) -> str:
    """Canonical id for the *1-based* car park ``index`` (e.g. 1 -> 'CBD_001')."""
    return f"{CARPARK_ID_PREFIX}_{index:0{CARPARK_ID_DIGITS}d}"


def carpark_name(index: int) -> str:
    """Human-readable name for the *1-based* car park ``index``."""
    name = _STREET_NAMES[(index - 1) % len(_STREET_NAMES)]
    cycle = (index - 1) // len(_STREET_NAMES)
    return name if cycle == 0 else f"{name} {cycle + 1}"


class CarParkRegistry:
    """In-memory catalogue of car parks and their camera URLs."""

    def __init__(self, num_carparks: int, camera_base_url: str) -> None:
        base = camera_base_url.rstrip("/")
        self._carparks: dict[str, CarParkInfo] = {}
        for i in range(1, num_carparks + 1):
            cid = carpark_id(i)
            self._carparks[cid] = CarParkInfo(
                id=cid,
                name=carpark_name(i),
                camera_url=f"{base}/cameras/{cid}/api/takephoto",
            )

    def __len__(self) -> int:
        return len(self._carparks)

    @property
    def count(self) -> int:
        return len(self._carparks)

    def all(self) -> list[CarParkInfo]:
        return list(self._carparks.values())

    def ids(self) -> list[str]:
        return list(self._carparks.keys())

    def get(self, carpark_id: str) -> CarParkInfo | None:
        return self._carparks.get(carpark_id)

    def sample(self, k: int, rng: random.Random | None = None) -> list[CarParkInfo]:
        """Return ``k`` *distinct* random car parks.

        ``k`` must be ``<=`` the catalogue size; callers cap at ``count``
        (``find-carparks`` uses ``min(2*n, count)``).
        """
        if k > len(self._carparks):
            raise ValueError(
                f"Requested {k} distinct car parks but only "
                f"{len(self._carparks)} are configured"
            )
        chooser = rng or random
        return chooser.sample(self.all(), k)

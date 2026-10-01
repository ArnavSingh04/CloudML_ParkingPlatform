"""On-demand matplotlib rendering for the operational dashboard (OPS-REQ-2).

matplotlib is a blocking, CPU-bound library, so — exactly like model inference
— the render must never run directly on the event loop. The route offloads it
with ``asyncio.to_thread`` (see ``api/operations.py``). We select the headless
'Agg' backend so no display server / GUI is required in a container.

matplotlib is imported lazily inside the function so importing this module (and
running the rest of the app/tests) stays cheap when no plot is requested.
"""

from __future__ import annotations

import io

from .models.schemas import CarParkStatus

# Colours mirror the HTML dashboard for visual consistency.
_AVAILABLE = "#3fb950"
_OCCUPIED = "#d29922"


def render_availability_png(
    statuses: list[CarParkStatus],
    window_seconds: int,
    recent_user_count: int,
) -> bytes:
    """Render a stacked bar chart (available vs occupied) as PNG bytes.

    Only car parks with a successful ('ok') status are plotted; when none exist
    yet a friendly placeholder is drawn instead of an empty axes.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display/GUI needed
    import matplotlib.pyplot as plt

    ok = [s for s in statuses if s.status == "ok"]

    fig, ax = plt.subplots(figsize=(max(6.0, len(ok) * 0.5), 4.0), dpi=100)

    if ok:
        # Strip the "CBD_" prefix so 24+ x-axis labels stay legible.
        labels = [s.carpark_id.split("_", 1)[-1] for s in ok]
        empty = [s.empty_count for s in ok]
        occupied = [s.occupied_count for s in ok]
        positions = range(len(ok))

        ax.bar(positions, empty, color=_AVAILABLE, label="Available")
        ax.bar(positions, occupied, bottom=empty, color=_OCCUPIED, label="Occupied")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Car park")
        ax.set_ylabel("Parking spaces")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No data yet — call /api/find-carparks",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()

    ax.set_title(
        f"SmartPark availability · {recent_user_count} "
        f"user(s) in last {window_seconds}s"
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)  # release the figure so repeated calls don't leak memory
    return buffer.getvalue()

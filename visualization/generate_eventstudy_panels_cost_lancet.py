#"both_ages18-120_quarterly_eventtimes-12-12_cal2018-01-01to2024-12-31_basevarying_UNBAL_pre4_post2_wall"

# call from terminal, e.g.: python3 diff-in-diff/generate_eventstudy_panels_cost.py results/did_subgroups_bootstrap/both_ages18-120_quarterly_eventtimes-12-12_cal2018-01-01to2024-12-31_baseuniversal_UNBAL_pre4_post2_wall

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.stats import norm
from matplotlib.legend_handler import HandlerBase

class Asterisk:
    def __init__(self, text):
        self.text = text

class HandlerAsterisk(HandlerBase):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height,
        fontsize, trans
    ):
        t = Text(
            x=xdescent + width / 2,
            y=ydescent + height / 2,
            text=orig_handle.text,
            color="red",
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            transform=trans,
        )
        return [t]

PRE_COLOR = "#5B8A72"      # muted green
POST_COLOR = "#4C78A8"     # muted blue

# ---------------------------------------------------------------------------
# Lancet number style
#   - decimal point is a mid-line dot (U+00B7):     0·5
#   - thousands: no separator for 4 digits (1000),
#     thin space (U+2009) from 5 digits on:         100 281
#   - minus is an en rule (U+2013):                 –0·5
# ---------------------------------------------------------------------------

LANCET_DECIMAL = "·"
LANCET_THOUSANDS = " "
LANCET_MINUS = "–"   # set to "−" if you prefer a true minus sign


def lancet_number(x, decimals=0, minus=LANCET_MINUS):
    """Format a single number in Lancet style."""
    if x is None or pd.isna(x):
        return "NA"
    s = f"{abs(float(x)):.{decimals}f}"
    int_part, _, frac_part = s.partition(".")
    if len(int_part) >= 5:
        groups = []
        while int_part:
            groups.insert(0, int_part[-3:])
            int_part = int_part[:-3]
        int_part = LANCET_THOUSANDS.join(groups)
    out = int_part + (LANCET_DECIMAL + frac_part if frac_part else "")
    if float(x) < 0 and float(s) != 0:   # avoid "–0·0"
        out = minus + out
    return out


class LancetFormatter(mticker.Formatter):
    """Tick formatter; picks the fewest decimals that show every tick exactly."""

    def __init__(self, decimals=None, minus=LANCET_MINUS):
        self.decimals = decimals
        self.minus = minus
        self._dec = 0 if decimals is None else decimals

    def set_locs(self, locs):
        super().set_locs(locs)
        if self.decimals is None:
            locs = np.asarray([l for l in locs if np.isfinite(l)])
            self._dec = 0
            if len(locs):
                for d in range(7):
                    if np.allclose(np.round(locs, d), locs, rtol=0, atol=1e-9):
                        self._dec = d
                        break
                else:
                    self._dec = 6

    def __call__(self, x, pos=None):
        return lancet_number(x, self._dec, self.minus)


def format_p(p):
    """p-value string without the leading comparator."""
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<" + lancet_number(0.001, 3)
    return lancet_number(p, 3)


def p_expr(p):
    """'=0·034' or '<0·001' — avoids the 'p=<0·001' construction."""
    s = format_p(p)
    return s if s.startswith("<") else "=" + s


def stars_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_chf(x):
    return lancet_number(x, 1)


def _require(path):
    """Fail with a listing of what the folder actually contains."""
    if not path.is_file():
        contents = sorted(p.name for p in path.parent.glob("*.csv"))
        raise FileNotFoundError(
            f"Missing: {path.name}\n"
            f"Folder: {path.parent.resolve()}\n"
            f"CSV files there:\n  " + "\n  ".join(contents or ["(none)"])
        )
    return path


def _base_suffix(folder):
    """'basevarying' -> 'varying', 'baseuniversal' -> 'universal'."""
    name = folder.resolve().name
    if "basevarying" in name:
        return "varying"
    if "baseuniversal" in name:
        return "universal"
    raise ValueError(f"Cannot infer base period (varying/universal) from '{name}'")


def plot_eventstudy_three_panels(folder_name, save_as=None):
    folder = Path(folder_name)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    suffix = _base_suffix(folder)

    eventstudy_files = {
        "Total cost": {
            "file": f"eventstudy_total_claim_cost_{suffix}.csv",
            "outcome": "total_claim_cost",
        },
        "Inpatient cost": {
            "file": f"eventstudy_inpatient_cost_{suffix}.csv",
            "outcome": "inpatient_cost",
        },
        "Outpatient cost": {
            "file": f"eventstudy_outpatient_cost_{suffix}.csv",
            "outcome": "outpatient_cost",
        },
    }

    att_results = pd.read_csv(_require(folder / "att_results_r_did.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=False)

    for ax, (title, spec) in zip(axes, eventstudy_files.items()):
        path = _require(folder / spec["file"])
        df = pd.read_csv(path).sort_values("event_time").copy()

        required_cols = {"event_time", "att", "se", "ci95_low", "ci95_high"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        z = (df["att"] / df["se"]).abs()
        df["p_value"] = 2 * (1 - norm.cdf(z))
        df["stars"] = df["p_value"].apply(stars_from_p)

        pre = df[df["event_time"] < 0]
        post = df[df["event_time"] >= 0]

        for data, color, label in [
            (pre, PRE_COLOR, "Pre-adoption"),
            (post, POST_COLOR, "Post-adoption"),
        ]:
            yerr = [
                data["att"] - data["ci95_low"],
                data["ci95_high"] - data["att"],
            ]
            
            # Connect adjacent event-study estimates
            ax.plot(
                data["event_time"],
                data["att"],
                color=color,
                linewidth=1.2,
                zorder=1,
            )

            ax.errorbar(
                data["event_time"],
                data["att"],
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                markersize=4,
                linewidth=1.2,
                label=label,
                zorder=2
            )

            y_range = df["ci95_high"].max() - df["ci95_low"].min()
            star_offset = 0.012 * y_range
            
            for _, row in post.iterrows():
                if row["stars"]:
                    ax.text(
                        row["event_time"],
                        row["ci95_high"] + star_offset,
                        row["stars"],
                        color="red",
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                    )

        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.45)

        quarter_ticks = list(range(-12, 13, 2))
        ax.set_xticks(quarter_ticks)
        ax.set_xlim(-12.5, 12.5)

        ax.set_title(f"{title} (N={lancet_number(100281)})")
        ax.set_xlabel("Measurement period relative to app uptake (quarters)", fontsize=12)
        ax.set_ylabel("ATT (CHF)")

        secax = ax.secondary_xaxis(
            "bottom",
            functions=(lambda q: q * 3, lambda m: m / 3),
        )
        secax.spines["bottom"].set_position(("outward", 40))
        secax.set_xticks([q * 3 for q in quarter_ticks])
        secax.set_xlabel("Time relative to app uptake (months)", fontsize=12)

        ax.xaxis.set_major_formatter(LancetFormatter())
        ax.yaxis.set_major_formatter(LancetFormatter())
        secax.xaxis.set_major_formatter(LancetFormatter())

        overall = att_results[
            (att_results["outcome"] == spec["outcome"])
            & (att_results["type"] == "dynamic_overall")
        ]

        if overall.empty:
            raise ValueError(
                f"No dynamic_overall result found for outcome={spec['outcome']}"
            )

        overall = overall.iloc[0]

        # Numbers are kept outside mathtext so the ·, – and thin space render
        # as plain Unicode text.
        att_txt = format_chf(overall["att"])
        overall_text = (
            r"$\bf{3\!-\!year\ ATT:}$ " + rf"$\bf{{{att_txt}\ CHF}}$"
            "\n"
            f"95% CI {format_chf(overall['ci95_low'])} to "
            f"{format_chf(overall['ci95_high'])}\n"
            r"$\it{p}$" + p_expr(overall["p_value"])
        )

        ax.text(
            0.03,
            0.03,
            overall_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="lightgrey",
                alpha=0.9,
            ),
        )

        wald_p = overall["wald_pretrend_p"]
        wald_color = "red" if wald_p < 0.05 else "dimgray"

        ax.text(
            0.98,
            0.99,
            r"Wald-$\it{p}$" + p_expr(wald_p),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color=wald_color,
        )

        ax.grid(alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color=PRE_COLOR, marker="o", lw=1.2, label="Pre-adoption"),
        Line2D([0], [0], color=POST_COLOR, marker="o", lw=1.2, label="Post-adoption"),
        Line2D([], [], linestyle="none", label=""),
    
        Asterisk("*"),
        Asterisk("**"),
        Asterisk("***"),
    ]
    
    legend_labels = [
        "Pre-adoption",
        "Post-adoption",
        "",
        f"p<{lancet_number(0.05, 2)}",
        f"p<{lancet_number(0.01, 2)}",
        f"p<{lancet_number(0.001, 3)}",
    ]
    
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map={Asterisk: HandlerAsterisk()},
        loc="lower left",
        bbox_to_anchor=(0.02, 0.005),
        frameon=False,
        ncol=6,
        fontsize=12,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))

    if save_as is not None:
        save_as = Path(save_as)
        save_as.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate three-panel event study plot."
    )

    parser.add_argument(
        "folder",
        help="Folder containing event study CSVs and att_results_r_did.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output filename. Default: <folder>/eventstudy_three_panels.png",
    )

    args = parser.parse_args()

    output = (
        Path(args.folder) / "eventstudy_three_panels_cost.png"
        if args.output is None
        else Path(args.output)
    )

    plot_eventstudy_three_panels(
        folder_name=args.folder,
        save_as=output,
    )

    print(f"Saved figure to: {output}")
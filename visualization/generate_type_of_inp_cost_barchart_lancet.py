#!/usr/bin/env python3

# call from terminal, e.g.: python3 diff-in-diff/generate_type_of_inp_cost_barchart.py results/did_inpatient_type_cost/both_ages18-120_quarterly_eventtimes-12-12_cal2018-01-01to2024-12-31_basevarying_UNBAL_pre4_post2_wall/att_results_r_did.csv

#!/usr/bin/env python3

# call from terminal, e.g.:
# python3 diff-in-diff/generate_type_of_inp_cost_barchart.py \
# results/did_inpatient_type_cost/both_ages18-120_quarterly_eventtimes-12-12_cal2018-01-01to2024-12-31_basevarying_UNBAL_pre4_post2_wall/att_results_r_did.csv

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Lancet number style
#   - decimal point is a mid-line dot (U+00B7):     0·5
#   - thousands: thin space (U+2009) from 5 digits on
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

    def __init__(self, minus=LANCET_MINUS):
        self.minus = minus
        self._dec = 0

    def set_locs(self, locs):
        super().set_locs(locs)
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


COST_OUTCOME_ORDER = [
    "inpatient_cost",
    "inpatient_cost_emergency",
    "inpatient_cost_planned",
    "inpatient_cost_ill",
    "inpatient_cost_acc",
]

CLAIMS_OUTCOME_ORDER = [
    "n_inpatient_claims",
    "n_inpatient_emergency",
    "n_inpatient_planned",
    "n_inpatient_ill",
    "n_inpatient_acc",
]

OUTCOME_LABELS = {
    "inpatient_cost": "Overall\ninpatient",
    "inpatient_cost_emergency": "Emergency\nadmission",
    "inpatient_cost_planned": "Non-emergency\nadmission",
    "inpatient_cost_ill": "Illness\nclaim",
    "inpatient_cost_acc": "Accident\nclaim",
    "n_inpatient_claims": "Overall\ninpatient",
    "n_inpatient_emergency": "Emergency\nadmission",
    "n_inpatient_planned": "Non-emergency\nadmission",
    "n_inpatient_ill": "Illness\nclaim",
    "n_inpatient_acc": "Accident\nclaim",
}

DARK_BLUE = "#4C78A8"
LIGHT_BLUE = "#AFCBE6"
GREY = "#BDBDBD"
TEXT_BLUE = "#1F4E79"
TEXT_GREY = "#8C8C8C"


def format_p(p):
    if pd.isna(p):
        return "p=NA"
    if p < 0.001:
        return f"p<{lancet_number(0.001, 3)}***"
    if p < 0.01:
        return f"p={lancet_number(p, 3)}**"
    if p < 0.05:
        return f"p={lancet_number(p, 3)}*"
    return f"p={lancet_number(p, 2)} n.s."


def is_significant(p):
    return pd.notna(p) and p < 0.05


def prepare_panel_data(df, outcome_order, att_csv):
    panel = df[
        (df["type"] == "dynamic_overall")
        & (df["outcome"].isin(outcome_order))
    ].copy()

    missing = set(outcome_order) - set(panel["outcome"])
    if missing:
        raise ValueError(f"Missing outcomes in {att_csv}: {sorted(missing)}")

    panel["outcome"] = pd.Categorical(
        panel["outcome"],
        categories=outcome_order,
        ordered=True,
    )

    return panel.sort_values("outcome").reset_index(drop=True)


def add_bar_panel(ax, panel, outcome_order, ylabel, decimals, title):
    x = list(range(len(panel)))
    y = panel["att"]

    yerr = [
        panel["att"] - panel["ci95_low"],
        panel["ci95_high"] - panel["att"],
    ]

    colors = []
    for _, row in panel.iterrows():
        if row["outcome"] == outcome_order[0]:
            colors.append(DARK_BLUE)
        elif is_significant(row["p_value"]):
            colors.append(LIGHT_BLUE)
        else:
            colors.append(GREY)

    ax.bar(
        x,
        y,
        color=colors,
        edgecolor="white",
        width=0.58,
        zorder=2,
    )

    ax.axhline(
        0,
        color="#9A9A9A",
        linewidth=2,
        zorder=10,
        clip_on=False,
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="#4A4A4A",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=20,
    )

    ax.grid(axis="y", alpha=0.25, zorder=0)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in panel["outcome"]])

    y_min = min(panel["ci95_low"].min(), y.min())
    y_max = max(panel["ci95_high"].max(), 0)
    padding = 0.18 * (y_max - y_min)
    ax.set_ylim(y_min - padding, y_max + padding)

    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low

    for i, row in panel.iterrows():
        p_text = format_p(row["p_value"])
        p_color = TEXT_BLUE if is_significant(row["p_value"]) else TEXT_GREY

        ax.text(
            i,
            y_low + 0.04 * y_range,
            p_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color=p_color,
        )

        ax.text(
            i,
            0 + 0.06 * y_range,
            lancet_number(row["att"], decimals),
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT_BLUE if is_significant(row["p_value"]) else TEXT_GREY,
            fontweight="bold",
        )

    ax.yaxis.set_major_formatter(LancetFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_inpatient_att_bars(att_csv, output=None):
    att_csv = Path(att_csv)
    if att_csv.is_dir():
        att_csv = att_csv / "att_results_r_did.csv"
    df = pd.read_csv(att_csv)

    cost_panel = prepare_panel_data(df, COST_OUTCOME_ORDER, att_csv)
    claims_panel = prepare_panel_data(df, CLAIMS_OUTCOME_ORDER, att_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=False)

    add_bar_panel(
        ax=axes[0],
        panel=cost_panel,
        outcome_order=COST_OUTCOME_ORDER,
        ylabel="3-year ATT cost (CHF)",
        decimals=1,
        title="Inpatient costs",
    )

    add_bar_panel(
        ax=axes[1],
        panel=claims_panel,
        outcome_order=CLAIMS_OUTCOME_ORDER,
        ylabel="3-year ATT number of claims",
        decimals=4,
        title="Inpatient claims",
    )

    plt.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot two-panel inpatient 3-year ATT bar chart from att_results_r_did.csv."
    )

    parser.add_argument(
        "att_csv",
        help="Path to att_results_r_did.csv, or the folder containing it",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path. Default: same folder as CSV / inpatient_att_cost_claims_bars.png",
    )

    args = parser.parse_args()

    in_path = Path(args.att_csv)
    out_dir = in_path if in_path.is_dir() else in_path.parent
    output = (
        out_dir / "inpatient_att_cost_claims_bars.png"
        if args.output is None
        else Path(args.output)
    )

    plot_inpatient_att_bars(args.att_csv, output)

    print(f"Saved figure to: {output}")
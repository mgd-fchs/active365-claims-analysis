#!/usr/bin/env bash
# =============================================================================
# Run the staggered-adoption DiD pipeline (diff-in-diff/run_mdid_pipeline.py)
# for one or more subgroups / specifications.
#
# Run from the repository root:
#     bash diff-in-diff/run_mdid.sh
#
# Requires data preparation steps 1+2 (SOURCE_TABLE below); steps-diff
# stratification additionally requires step 3.
#
# Every combination of the loop lists (WEIGHT_GROUPS x GENDERS x age bands x
# calendar windows x event windows x steps-diff categories) is one run with its
# own output folder under ROOT_OUT. The folder name encodes the specification.
# Each run writes: did_input.parquet, unique_ids.csv, did_sample_summary.csv,
# preadoption_cost_summary.csv, winsorization_cutoffs.csv, support_event_time.*,
# stratum_summary.csv and the R estimation results.
#
# Booleans: write True / False (case-sensitive).
# =============================================================================
set -euo pipefail

DEBUG=False          # True: print every shell command (set -x)
DRY_RUN=False        # True: print the python commands instead of running them
[ "$DEBUG" = "True" ] && set -x

# ---- Paths ------------------------------------------------------------------
DB="./insurance.duckdb"
SCRIPT="diff-in-diff/run_mdid_pipeline.py"
ROOT_OUT="./results/did_placebo"        # parent folder for all run folders
SOURCE_TABLE="claims_person_month_summary_final_w_inpatient_type"   # data-prep step 2 output (step 3 adds steps_diff)

# ---- Outcomes ---------------------------------------------------------------
# Comma-separated column names of SOURCE_TABLE. Available:
#   costs : total_claim_cost, inpatient_cost, outpatient_cost,
#           inpatient_cost_planned, inpatient_cost_emergency,
#           inpatient_cost_ill, inpatient_cost_acc, outpatient_cost_ill, outpatient_cost_acc
#   counts: n_inpatient_claims, n_outpatient_claims, n_inpatient_emergency, n_inpatient_planned,
#           n_inpatient_ill, n_inpatient_acc
OUTCOMES="total_claim_cost, inpatient_cost, n_inpatient_claims, outpatient_cost, n_outpatient_claims"

# ---- Winsorization (cost outcomes only) ---------------------------------------
# WINSORIZE=True replaces every cost outcome in OUTCOMES by its winsorized
# version (<col>_w). Count outcomes are never winsorized.
# Cutoff = WINSOR_PCT quantile among positive values, pooled over treated/control
# and all periods, computed on each run's own (sub)sample after sample selection.
# Cutoffs are written to winsorization_cutoffs.csv in every run folder.
WINSORIZE=False
WINSOR_PCT=0.99      # 0 < WINSOR_PCT < 1

# ---- Time structure ---------------------------------------------------------
GRANULARITY="quarterly"   # quarterly | half-yearly | yearly | monthly

# Event-time window for the dynamic event study. Paired by index:
# MIN_E_LIST[i] goes with MAX_E_LIST[i].
MIN_E_LIST=(-12)
MAX_E_LIST=(12)

# Calendar window on t_period_start (YYYY-MM-DD). Paired by index.
CAL_MIN=("2018-01-01")
CAL_MAX=("2024-12-31")

# ---- Estimation -------------------------------------------------------------
CONTROL_GROUP="notyettreated"   # notyettreated | nevertreated
BASE_PERIOD="varying"           # varying | universal

# Panel handling
#   UNBALANCED=False: balanced panel (drops early-treated and incomplete ids)
#   UNBALANCED=True : unbalanced panel; each id needs >= MIN_PRE_PERIODS
#                     pre-treatment and >= MIN_POST_PERIODS post-treatment periods
UNBALANCED=True
MIN_PRE_PERIODS=4
MIN_POST_PERIODS=2

# Placebo: randomly permute treatment timing across treated units (fixed seed
# in the python script).
PLACEBO_TIME=True

# ---- Subgroups (looped) -----------------------------------------------------
# Age at adoption, inclusive. Paired by index: AGE_MIN[i] goes with AGE_MAX[i].
AGE_MIN=(18)
AGE_MAX=(120)

GENDERS=("both")          # both | male | female
WEIGHT_GROUPS=("all")     # all | underweight | normal | overweight | obese (WHO BMI at adoption)

# ---- Optional stratifications (one group range per run) ---------------------
# Engagement
#   ENGAGEMENT_DEFINITION : alltime_frequency | 30d_frequency | alltime_points |
#                           frequency_points | minimal_engager | session_user
#   ENGAGEMENT_GRANULARITY: quartile | decile (frequency/points definitions)
#                           none              (minimal_engager, session_user: binary 0/1)
#   ENGAGEMENT_GROUP_MIN/MAX: keep groups MIN..MAX (inclusive; for binary: value 0 or 1)
ENGAGEMENT_STRAT=False
ENGAGEMENT_DEFINITION="session_user"
ENGAGEMENT_GRANULARITY="none"
ENGAGEMENT_GROUP_MIN=1
ENGAGEMENT_GROUP_MAX=1

# Income index, empirical quantiles within the analysed subgroup
#   INCOME_GRANULARITY: tercile | quartile | decile; groups 1-based, inclusive
INCOME_STRAT=False
INCOME_GRANULARITY="tercile"
INCOME_GROUP_MIN=1
INCOME_GROUP_MAX=1

# Healthcare costs in the 12 months before adoption, empirical quantiles
#   PRIOR_COST_DEFINITION : total | inpatient
#   PRIOR_COST_GRANULARITY: tercile | quartile | decile; groups 1-based, inclusive
PRIOR_COST_STRAT=False
PRIOR_COST_DEFINITION="total"
PRIOR_COST_GRANULARITY="quartile"
PRIOR_COST_GROUP_MIN=1
PRIOR_COST_GROUP_MAX=1

# Pre-existing chronic condition (any PCG) / type 2 diabetes at adoption
#   value 0 = no pre-existing condition, 1 = has one
PCG_STRAT=False
PCG_VALUE=0
T2D_STRAT=False
T2D_VALUE=0

# Change in mean daily steps, days 91-180 vs 1-90 after adoption (data-prep step 3).
# Looped over the listed categories:
#   increase: >= +1000 | stable: between -1000 and +1000 | decrease: <= -1000
STEPS_DIFF_STRAT=False
STEPS_DIFF_CATEGORIES=("increase" "stable" "decrease")

# ---- Checks -----------------------------------------------------------------
if [ "${#AGE_MIN[@]}" -ne "${#AGE_MAX[@]}" ];       then echo "AGE_MIN/AGE_MAX differ in length";       exit 1; fi
if [ "${#CAL_MIN[@]}" -ne "${#CAL_MAX[@]}" ];       then echo "CAL_MIN/CAL_MAX differ in length";       exit 1; fi
if [ "${#MIN_E_LIST[@]}" -ne "${#MAX_E_LIST[@]}" ]; then echo "MIN_E_LIST/MAX_E_LIST differ in length"; exit 1; fi
if [ ! -f "$SCRIPT" ]; then echo "Script not found: $SCRIPT (run from the repository root)"; exit 1; fi

# ---- Outcomes (winsorized cost columns if requested) -----------------------
COST_COLS=" total_claim_cost inpatient_cost outpatient_cost inpatient_cost_planned inpatient_cost_emergency inpatient_cost_ill inpatient_cost_acc outpatient_cost_ill outpatient_cost_acc "
RUN_OUTCOMES=""
IFS=',' read -ra _OUTS <<< "$OUTCOMES"
for o in "${_OUTS[@]}"; do
  o="$(echo "$o" | xargs)"            # trim whitespace
  [ -z "$o" ] && continue
  if [ "$WINSORIZE" = "True" ] && [[ "$COST_COLS" == *" $o "* ]]; then
    o="${o}_w"
  fi
  RUN_OUTCOMES="${RUN_OUTCOMES:+$RUN_OUTCOMES,}$o"
done
WINSOR_TAG="$(awk -v p="$WINSOR_PCT" 'BEGIN { printf "%g", p * 100 }')"

# ---- Flags shared by all runs -----------------------------------------------
UNBAL_FLAG="";   [ "$UNBALANCED" = "True" ]       && UNBAL_FLAG="--unbalanced --min-pre-periods $MIN_PRE_PERIODS --min-post-periods $MIN_POST_PERIODS"
PLACEBO_FLAG=""; [ "$PLACEBO_TIME" = "True" ]     && PLACEBO_FLAG="--placebo-time"
ENG_FLAG="";     [ "$ENGAGEMENT_STRAT" = "True" ] && ENG_FLAG="--engagement-strat --engagement-definition $ENGAGEMENT_DEFINITION --engagement-granularity $ENGAGEMENT_GRANULARITY --engagement-group-min $ENGAGEMENT_GROUP_MIN --engagement-group-max $ENGAGEMENT_GROUP_MAX"
INC_FLAG="";     [ "$INCOME_STRAT" = "True" ]     && INC_FLAG="--income-strat --income-granularity $INCOME_GRANULARITY --income-group-min $INCOME_GROUP_MIN --income-group-max $INCOME_GROUP_MAX"
PCOST_FLAG="";   [ "$PRIOR_COST_STRAT" = "True" ] && PCOST_FLAG="--prior-cost-strat --prior-cost-definition $PRIOR_COST_DEFINITION --prior-cost-granularity $PRIOR_COST_GRANULARITY --prior-cost-group-min $PRIOR_COST_GROUP_MIN --prior-cost-group-max $PRIOR_COST_GROUP_MAX"
PCG_FLAG="";     [ "$PCG_STRAT" = "True" ]        && PCG_FLAG="--pcg-strat --pcg-value $PCG_VALUE"
T2D_FLAG="";     [ "$T2D_STRAT" = "True" ]        && T2D_FLAG="--t2d-strat --t2d-value $T2D_VALUE"

# steps-diff stratification off -> loop once without filter
if [ "$STEPS_DIFF_STRAT" = "True" ]; then
  STEPS_DIFF_LOOP=("${STEPS_DIFF_CATEGORIES[@]}")
else
  STEPS_DIFF_LOOP=("none")
fi

echo "Outcomes: $RUN_OUTCOMES"
echo "Starting loop..."

for weight in "${WEIGHT_GROUPS[@]}"; do
  for gender in "${GENDERS[@]}"; do
    for aidx in "${!AGE_MIN[@]}"; do
      age_min="${AGE_MIN[$aidx]}"; age_max="${AGE_MAX[$aidx]}"

      for cidx in "${!CAL_MIN[@]}"; do
        cal_min="${CAL_MIN[$cidx]}"; cal_max="${CAL_MAX[$cidx]}"

        for eidx in "${!MIN_E_LIST[@]}"; do
          MIN_E="${MIN_E_LIST[$eidx]}"; MAX_E="${MAX_E_LIST[$eidx]}"

          for steps_cat in "${STEPS_DIFF_LOOP[@]}"; do
            STEPS_DIFF_FLAG=""
            [ "$steps_cat" != "none" ] && STEPS_DIFF_FLAG="--steps-diff-strat --steps-diff-category $steps_cat"

            # ---- output folder name encodes the specification ----
            out_dir="${ROOT_OUT%/}/${gender}_ages${age_min}-${age_max}_${GRANULARITY}_eventtimes${MIN_E}-${MAX_E}_cal${cal_min}to${cal_max}_base${BASE_PERIOD}_ctrl${CONTROL_GROUP}"
            if [ "$UNBALANCED" = "True" ]; then
              out_dir="${out_dir}_UNBAL_pre${MIN_PRE_PERIODS}_post${MIN_POST_PERIODS}"
            else
              out_dir="${out_dir}_BAL"
            fi
            out_dir="${out_dir}_w${weight}"
            [ "$ENGAGEMENT_STRAT" = "True" ] && out_dir="${out_dir}_eng${ENGAGEMENT_DEFINITION}_${ENGAGEMENT_GRANULARITY}_${ENGAGEMENT_GROUP_MIN}-${ENGAGEMENT_GROUP_MAX}"
            [ "$INCOME_STRAT" = "True" ]     && out_dir="${out_dir}_inc${INCOME_GRANULARITY}_${INCOME_GROUP_MIN}-${INCOME_GROUP_MAX}"
            [ "$PRIOR_COST_STRAT" = "True" ] && out_dir="${out_dir}_pcost${PRIOR_COST_DEFINITION}_${PRIOR_COST_GRANULARITY}_${PRIOR_COST_GROUP_MIN}-${PRIOR_COST_GROUP_MAX}"
            [ "$PCG_STRAT" = "True" ]        && out_dir="${out_dir}_pcg${PCG_VALUE}"
            [ "$T2D_STRAT" = "True" ]        && out_dir="${out_dir}_t2d${T2D_VALUE}"
            [ "$steps_cat" != "none" ]       && out_dir="${out_dir}_stepsdiff${steps_cat}"
            [ "$WINSORIZE" = "True" ]        && out_dir="${out_dir}_winsp${WINSOR_TAG}"
            [ "$PLACEBO_TIME" = "True" ]     && out_dir="${out_dir}_placebo"

            echo "Running gender=$gender age=${age_min}-${age_max} weight=${weight} cal=${cal_min}..${cal_max} event=${MIN_E}..${MAX_E} base=${BASE_PERIOD} control=${CONTROL_GROUP} unbalanced=${UNBALANCED} steps_diff=${steps_cat} winsorize=${WINSORIZE} placebo=${PLACEBO_TIME}"
            echo "  -> $out_dir"

            # flag strings are intentionally unquoted so they split into separate arguments
            cmd=(python3 "$SCRIPT"
              --db "$DB"
              --out-dir "$out_dir"
              --source-table "$SOURCE_TABLE"
              --work-table "$SOURCE_TABLE"
              --cal-min "$cal_min" --cal-max "$cal_max"
              --age-min "$age_min" --age-max "$age_max"
              --gender "$gender"
              --weight-group "$weight"
              --granularity "$GRANULARITY"
              --min-e "$MIN_E" --max-e "$MAX_E"
              --control-group "$CONTROL_GROUP"
              --base-period "$BASE_PERIOD"
              --outcomes "$RUN_OUTCOMES"
              --winsor-pct "$WINSOR_PCT"
              $UNBAL_FLAG $ENG_FLAG $INC_FLAG $PCOST_FLAG $PCG_FLAG $T2D_FLAG $STEPS_DIFF_FLAG $PLACEBO_FLAG
            )

            if [ "$DRY_RUN" = "True" ]; then
              printf '  %q' "${cmd[@]}"; echo
            else
              mkdir -p "$out_dir"
              "${cmd[@]}"
            fi
          done
        done
      done
    done
  done
done

echo "Done."
#!/usr/bin/env Rscript
options(error = function() {
  traceback(2)
  q(status = 1)
})

args <- commandArgs(trailingOnly = TRUE)
cat("ARGS:\n"); print(args)

required_packages <- c("arrow", "did", "readr", "ggplot2")

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org", quiet = FALSE)
}

suppressPackageStartupMessages({
  library(arrow)
  library(did)
  library(readr)
  library(ggplot2)
})

cat("All packages loaded successfully.\n")
cat("Package versions:\n")
for (pkg in required_packages) {
  cat(sprintf("  %s: %s\n", pkg, as.character(packageVersion(pkg))))
}

in_path <- args[1]
out_dir <- args[2]
min_e <- as.integer(args[3])
max_e <- as.integer(args[4])
control_group <- args[5]  # "notyettreated" or "nevertreated"
base_period <- args[6]    # "universal" or "varying"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(in_path)

# Column names in your exported parquet
idname <- "id_num"
tname  <- "t_index"
gname  <- "g_index"

# Basic sanity checks
req_cols <- c(idname, tname, gname, "age", "gender_num")
missing <- setdiff(req_cols, names(df))
if (length(missing) > 0) {
  stop(paste0("Missing required columns in parquet: ", paste(missing, collapse = ", ")))
}

# Ensure expected types
df[[idname]] <- as.integer(df[[idname]])
df[[tname]]  <- as.integer(df[[tname]])
df[[gname]]  <- as.integer(df[[gname]]) 

# --- DEBUG: show 1-2 individuals (short print) ---
cat("\n===== DEBUG: sample individuals (raw df) =====\n")
uids <- unique(df[[idname]])
uids <- uids[!is.na(uids)]
uids <- head(uids, 2)
for (uid in uids) {
  cat("\n--- id =", uid, " ---\n")
  sub <- df[df[[idname]] == uid, c(idname, tname, gname, "age", "gender_num")]
  sub <- sub[order(sub[[tname]]), ]
  print(head(sub, 12))
}
cat("===== END DEBUG individuals =====\n\n")

# Define covariates and explicitly exclude non-outcome/id-ish variables
covars <- c("age", "gender_num")
exclude_cols <- unique(c(
  idname, tname, gname, covars,
  "eth_id", "t_period_start", "t_year", "t_sub", "g_year", "g_sub",
  "t_period_code", "g_period_code", "event_time"
))

# Define outcomes
outcomes_arg <- args[7]
if (is.na(outcomes_arg) || outcomes_arg == "") {
  stop("No outcomes provided via CLI arg 7.")
}
requested_outcomes <- trimws(strsplit(outcomes_arg, ",")[[1]])

numeric_cols <- names(df)[vapply(df, is.numeric, logical(1))]
available_outcomes <- setdiff(numeric_cols, exclude_cols)

missing_requested <- setdiff(requested_outcomes, available_outcomes)
if (length(missing_requested) > 0) {
  stop(paste0("Requested outcomes not found/not numeric in parquet: ",
              paste(missing_requested, collapse = ", ")))
}

outcomes <- requested_outcomes

cat("R outcomes:\n"); print(outcomes)

alpha <- 0.05
z <- qnorm(1 - alpha/2)

# significance marker helper
sig_marker <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  return("-")
}

rows <- list()

for (y in outcomes) {
  cat("\n==============================\n")
  cat("R outcome:", y, "\n")
  cat("==============================\n")

  cat("Outcome NA count:", sum(is.na(df[[y]])), " / ", nrow(df), "\n")

  res <- att_gt(
    yname = y,
    tname = tname,
    idname = idname,
    gname = gname,
    xformla = ~ age + gender_num,
    data = df,
    panel = TRUE,
    control_group = control_group,
    base_period = base_period,
    allow_unbalanced_panel = TRUE,
    anticipation = 0,
    bstrap = FALSE,
    cband = FALSE,
    # bstrap = TRUE,
    # biters = 5,
    print_details = FALSE,
    faster_mode = TRUE,
    est_method = "dr"
  )

  if (is.null(res$att) || is.null(res$se)) {
    stop("res$att or res$se is NULL. Cannot export group-time effects.")
  }

  gt_r <- data.frame(
    g   = as.integer(res$group),
    t   = as.integer(res$t),
    att = as.numeric(res$att),
    se  = as.numeric(res$se)
  )
  gt_r <- gt_r[!is.na(gt_r$att), ]
  gt_r <- gt_r[order(gt_r$g, gt_r$t), ]

  out_gt <- file.path(out_dir, paste0("gt_r_", y, "_", base_period, ".csv"))
  write.csv(gt_r, out_gt, row.names = FALSE)
  cat("Wrote:", out_gt, "\n")

  # Wald pre-test of parallel trends
  pre_stat <- NA_real_
  pre_p    <- NA_real_
  if (!is.null(res$W))     pre_stat <- as.numeric(res$W)
  if (!is.null(res$Wpval)) pre_p    <- as.numeric(res$Wpval)
  pre_sig  <- sig_marker(pre_p)

  # Dynamic aggregation
  dyn <- aggte(res, type = "dynamic", min_e = min_e, max_e = max_e)
  summary(dyn)

  # Plot
  p <- ggdid(dyn)
  outfile <- file.path(out_dir, paste0("did_plot_", y, "_", base_period, ".png"))
  ggsave(filename = outfile, plot = p, width = 8, height = 6, dpi = 300)

  # Save event-study table
  ev <- data.frame(
    outcome = y,
    event_time = dyn$egt,
    att = dyn$att.egt,
    se  = dyn$se.egt,
    ci95_low  = dyn$att.egt - z * dyn$se.egt,
    ci95_high = dyn$att.egt + z * dyn$se.egt
  )
  write_csv(ev, file.path(out_dir, paste0("eventstudy_", y, "_", base_period, ".csv")))

  # Overall effect p-value + sig
  overall_att <- as.numeric(dyn$overall.att)
  overall_se  <- as.numeric(dyn$overall.se)
  overall_p <- NA_real_
  if (!is.na(overall_att) && !is.na(overall_se) && overall_se > 0) {
    overall_p <- 2 * (1 - pnorm(abs(overall_att / overall_se)))
  }
  overall_sig <- sig_marker(overall_p)

  rows[[length(rows) + 1]] <- data.frame(
    outcome = y,
    type = "dynamic_overall",
    att = overall_att,
    se  = overall_se,
    ci95_low  = overall_att - z * overall_se,
    ci95_high = overall_att + z * overall_se,
    p_value   = overall_p,
    sig       = overall_sig,
    wald_pretrend_stat = pre_stat,
    wald_pretrend_p    = pre_p,
    wald_pretrend_sig  = pre_sig
  )
}

att_df <- do.call(rbind, rows)
write_csv(att_df, file.path(out_dir, "att_results_r_did.csv"))
cat("\nWrote:", file.path(out_dir, "att_results_r_did.csv"), "\n")
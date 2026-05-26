#!/usr/bin/env Rscript
# Regenerate publication-quality figures for the XMH paper (Information Sciences resubmission).
# Port of generate_paper_figures.py to R/ggplot2 — fixes audit issues:
#   - colorblind-safe Okabe-Ito palette
#   - consistent serif typography
#   - no overlapping cell labels in heatmaps
#   - clean axis labels (no LaTeX escape artifacts)
#   - improved legend placement (no data occlusion)
#
# Usage:
#   Rscript generate_paper_figures.R
#
# Reads JSONs from ./data/ and writes PDFs to ./figures_R/ (kept separate
# from Python output to allow side-by-side review).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(patchwork)
  library(scales)
  library(viridis)
  library(stringr)
  library(forcats)
  library(rlang)
})

# --------------------------------------------------------------------------
# Paths and global config
# --------------------------------------------------------------------------
HERE     <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
DATA_DIR <- file.path(HERE, "data")
OUT_DIR  <- file.path(HERE, "figures_R")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Page widths (Elsevier double-column)
W_FULL <- 7.16   # inches, full width
W_HALF <- 3.50

# Okabe-Ito 8-color colorblind-safe palette (Wong 2011)
OI <- c(
  black    = "#000000",
  orange   = "#E69F00",
  skyblue  = "#56B4E9",
  green    = "#009E73",
  yellow   = "#F0E442",
  blue     = "#0072B2",
  vermill  = "#D55E00",
  purple   = "#CC79A7"
)

# Operator colors — consistent across all figures
OP_COL <- c(
  Selection = unname(OI["vermill"]),
  Crossover = unname(OI["blue"]),
  Mutation  = unname(OI["green"]),
  Velocity  = unname(OI["purple"]),
  Topology  = unname(OI["orange"]),
  Position  = unname(OI["yellow"])
)

# Algorithm colors
ALG_COL <- c(
  DE  = unname(OI["blue"]),
  GA  = unname(OI["green"]),
  PSO = unname(OI["vermill"])
)

# --------------------------------------------------------------------------
# Theme: serif, publication-quality
# --------------------------------------------------------------------------
theme_pub <- function(base_size = 9) {
  theme_bw(base_size = base_size, base_family = "serif") +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.10), hjust = 0.5),
      plot.subtitle    = element_text(size = rel(0.95), hjust = 0.5),
      axis.title       = element_text(size = rel(1.05)),
      axis.text        = element_text(size = rel(0.90), color = "black"),
      legend.title     = element_text(size = rel(0.95)),
      legend.text      = element_text(size = rel(0.90)),
      legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
      legend.key       = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
      strip.background = element_rect(fill = "grey95", color = "black", linewidth = 0.3),
      strip.text       = element_text(face = "bold", size = rel(1.0))
    )
}

theme_set(theme_pub())

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
read_json_safe <- function(path) {
  if (!file.exists(path)) return(NULL)
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  # Python json.dump emits NaN / Infinity / -Infinity (not valid JSON);
  # jsonlite is strict, so we coerce these to null before parsing.
  txt <- gsub("\\bNaN\\b",       "null", txt)
  txt <- gsub("\\bInfinity\\b",  "null", txt)
  txt <- gsub("-Infinity",        "null", txt, fixed = TRUE)
  fromJSON(txt, simplifyVector = FALSE)
}

exact_data    <- read_json_safe(file.path(DATA_DIR, "exact_shapley_results.json"))
method_data   <- read_json_safe(file.path(DATA_DIR, "method_comparison.json"))
overhead_data <- read_json_safe(file.path(DATA_DIR, "overhead_results.json"))
exp8a         <- read_json_safe(file.path(DATA_DIR, "exp8_part_a.json"))
exp8b         <- read_json_safe(file.path(DATA_DIR, "exp8_part_b.json"))
exp9a         <- read_json_safe(file.path(DATA_DIR, "exp9_sweep_a.json"))
exp9b         <- read_json_safe(file.path(DATA_DIR, "exp9_sweep_b.json"))

FUNCTIONS  <- c("sphere", "rosenbrock", "zakharov", "dixon_price",
                "rastrigin", "schwefel", "ackley", "griewank", "levy")
FUNC_LABS  <- c("Sphere", "Rosenbrock", "Zakharov", "Dixon-P.",
                "Rastrigin", "Schwefel", "Ackley", "Griewank", "Levy")
DIMS       <- c("10", "30", "50")

# Helper: extract Shapley as % for one (alg, func, dim) cell
get_shapley_pct <- function(alg, func, dim) {
  cfg <- exact_data[[alg]][[func]][[as.character(dim)]]
  if (is.null(cfg)) return(NULL)
  sv  <- cfg$shapley_values
  tot <- sum(abs(unlist(sv)))
  if (tot < 1e-15) return(setNames(rep(0, length(sv)), names(sv)))
  setNames(100 * unlist(sv) / tot, names(sv))
}

# Classify operator key -> short label
op_short <- function(key) {
  k <- tolower(key)
  if (grepl("selection|greedy", k))  return("Selection")
  if (grepl("crossover|binomial", k)) return("Crossover")
  if (grepl("mutation|rand", k))     return("Mutation")
  if (grepl("velocity", k))           return("Velocity")
  if (grepl("topology", k))           return("Topology")
  if (grepl("position", k))           return("Position")
  str_to_title(str_split(key, "_")[[1]][1])
}

# Build long-format Shapley table for an algorithm
build_shap_table <- function(alg, funcs = FUNCTIONS, dims = DIMS) {
  rows <- list()
  for (func in funcs) {
    for (dim in dims) {
      pct <- get_shapley_pct(alg, func, dim)
      if (is.null(pct)) next
      for (k in names(pct)) {
        rows[[length(rows) + 1L]] <- data.frame(
          algorithm = alg,
          function_name = func,
          dimension = dim,
          op_key = k,
          op_label = op_short(k),
          shapley_pct = unname(pct[k]),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (length(rows) == 0) return(NULL)
  do.call(rbind, rows)
}

cat("Loaded:",
    sum(c(!is.null(exact_data), !is.null(method_data), !is.null(overhead_data),
          !is.null(exp8a), !is.null(exp8b), !is.null(exp9a), !is.null(exp9b))),
    "/ 7 data files\n")

# ========================================================================
# FIGURE 1: GA Shapley heatmap
# ========================================================================
fig1_ga_heatmap <- function() {
  df <- build_shap_table("GA") |>
    mutate(
      function_name = factor(function_name, levels = FUNCTIONS, labels = FUNC_LABS),
      dimension     = factor(paste0("D = ", dimension),
                             levels = paste0("D = ", DIMS)),
      op_label      = factor(op_label, levels = c("Selection", "Crossover", "Mutation"))
    )

  p <- ggplot(df, aes(x = function_name, y = op_label, fill = shapley_pct)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.0f", shapley_pct),
                  color = shapley_pct > 40),
              size = 3.0, fontface = "bold", show.legend = FALSE) +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "black")) +
    scale_fill_distiller(
      palette = "YlOrRd", direction = 1, limits = c(0, 60),
      name = "Shapley value (%)",
      guide = guide_colorbar(barwidth = 0.5, barheight = 6)
    ) +
    facet_wrap(~ dimension, nrow = 1) +
    scale_y_discrete(limits = rev) +
    labs(
      x = NULL, y = NULL,
      title = "GA: Exact Shapley values by function and dimension"
    ) +
    theme_pub() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid  = element_blank()
    )

  ggsave(file.path(OUT_DIR, "fig1_ga_shapley_heatmap.pdf"),
         p, width = W_FULL, height = 3.5, device = cairo_pdf)
  cat("  ok fig1_ga_shapley_heatmap.pdf\n")
}

# ========================================================================
# FIGURE 2: PSO Shapley heatmap
# ========================================================================
fig2_pso_heatmap <- function() {
  df <- build_shap_table("PSO") |>
    filter(!grepl("position", tolower(op_key))) |>
    mutate(
      function_name = factor(function_name, levels = FUNCTIONS, labels = FUNC_LABS),
      dimension     = factor(paste0("D = ", dimension),
                             levels = paste0("D = ", DIMS)),
      op_label      = factor(op_label, levels = c("Velocity", "Topology"))
    )

  p <- ggplot(df, aes(x = function_name, y = op_label, fill = shapley_pct)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.0f", shapley_pct),
                  color = shapley_pct > 50),
              size = 3.0, fontface = "bold", show.legend = FALSE) +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "black")) +
    scale_fill_distiller(
      palette = "YlGnBu", direction = 1, limits = c(0, 55),
      name = "Shapley value (%)",
      guide = guide_colorbar(barwidth = 0.5, barheight = 6)
    ) +
    facet_wrap(~ dimension, nrow = 1) +
    scale_y_discrete(limits = rev) +
    labs(
      x = NULL, y = NULL,
      title = "PSO: Exact Shapley values by function and dimension"
    ) +
    theme_pub() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid  = element_blank()
    )

  ggsave(file.path(OUT_DIR, "fig2_pso_shapley_heatmap.pdf"),
         p, width = W_FULL, height = 3.2, device = cairo_pdf)
  cat("  ok fig2_pso_shapley_heatmap.pdf\n")
}

# ========================================================================
# FIGURE 3: Dimensionality effect (GA + PSO line plots)
# ========================================================================
fig3_dimensionality_effect <- function() {
  ga <- build_shap_table("GA") |>
    mutate(dim_int = as.integer(dimension)) |>
    group_by(op_label, dim_int) |>
    summarise(mean_pct = mean(shapley_pct), .groups = "drop") |>
    filter(op_label %in% c("Selection", "Crossover", "Mutation"))

  pso <- build_shap_table("PSO") |>
    filter(!grepl("position", tolower(op_key))) |>
    mutate(dim_int = as.integer(dimension)) |>
    group_by(op_label, dim_int) |>
    summarise(mean_pct = mean(shapley_pct), .groups = "drop") |>
    filter(op_label %in% c("Velocity", "Topology"))

  pA <- ggplot(ga, aes(x = dim_int, y = mean_pct,
                       color = op_label, shape = op_label, group = op_label)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.4, fill = "white", stroke = 0.9) +
    scale_color_manual(values = OP_COL, name = NULL) +
    scale_shape_manual(values = c(Selection = 16, Crossover = 15, Mutation = 17), name = NULL) +
    scale_x_continuous(breaks = c(10, 30, 50)) +
    coord_cartesian(ylim = c(0, 55)) +
    labs(x = "Dimension", y = "Shapley value (%)",
         title = "GA: Operator attribution vs dimension") +
    theme(legend.position = c(0.98, 0.50),
          legend.justification = c(1, 0.5))

  pB <- ggplot(pso, aes(x = dim_int, y = mean_pct,
                        color = op_label, shape = op_label, group = op_label)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.4, fill = "white", stroke = 0.9) +
    scale_color_manual(values = OP_COL, name = NULL) +
    scale_shape_manual(values = c(Velocity = 16, Topology = 15), name = NULL) +
    scale_x_continuous(breaks = c(10, 30, 50)) +
    coord_cartesian(ylim = c(0, 55)) +
    labs(x = "Dimension", y = "Shapley value (%)",
         title = "PSO: Operator attribution vs dimension") +
    theme(legend.position = c(0.98, 0.50),
          legend.justification = c(1, 0.5))

  p <- pA + pB + plot_layout(ncol = 2)

  ggsave(file.path(OUT_DIR, "fig3_dimensionality_effect.pdf"),
         p, width = W_FULL, height = 3.0, device = cairo_pdf)
  cat("  ok fig3_dimensionality_effect.pdf\n")
}

# ========================================================================
# FIGURE 4: Method comparison (cosine + rank agreement)
# ========================================================================
fig4_method_comparison <- function() {
  methods <- c("QuickSHAP", "KernelSHAP", "Tracking")
  algos   <- c("DE", "GA", "PSO")

  rows <- lapply(method_data, function(r) {
    if (is.null(r$method) || is.null(r$algorithm)) return(NULL)
    if (!(r$method %in% methods) || !(r$algorithm %in% algos)) return(NULL)
    data.frame(
      method    = r$method,
      algorithm = r$algorithm,
      cosine    = ifelse(is.null(r$cosine), NA_real_, r$cosine),
      rank_agr  = ifelse(is.null(r$rank_agreement), NA_real_, r$rank_agreement),
      stringsAsFactors = FALSE
    )
  })
  df <- do.call(rbind, rows)

  agg <- df |>
    group_by(method, algorithm) |>
    summarise(
      cosine_mean = mean(cosine,   na.rm = TRUE),
      rank_mean   = mean(rank_agr, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      method    = factor(method, levels = methods),
      algorithm = factor(algorithm, levels = algos)
    )

  pA <- ggplot(agg, aes(x = method, y = cosine_mean, fill = algorithm)) +
    geom_col(position = position_dodge(0.75), width = 0.7, alpha = 0.9) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.4) +
    scale_fill_manual(values = ALG_COL, name = NULL) +
    coord_cartesian(ylim = c(0, 1.15)) +
    labs(x = NULL, y = "Cosine similarity",
         title = "Accuracy: cosine similarity vs Exact")

  pB <- ggplot(agg, aes(x = method, y = rank_mean, fill = algorithm)) +
    geom_col(position = position_dodge(0.75), width = 0.7, alpha = 0.9) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.4) +
    scale_fill_manual(values = ALG_COL, name = NULL) +
    coord_cartesian(ylim = c(0, 1.15)) +
    labs(x = NULL, y = "Rank agreement",
         title = "Accuracy: rank agreement vs Exact")

  p <- pA + pB + plot_layout(ncol = 2, guides = "collect") &
    theme(legend.position = "top")

  ggsave(file.path(OUT_DIR, "fig4_method_comparison.pdf"),
         p, width = W_FULL, height = 3.1, device = cairo_pdf)
  cat("  ok fig4_method_comparison.pdf\n")
}

# ========================================================================
# FIGURE 5: Overhead comparison (log-scale horizontal bars)
# ========================================================================
fig5_overhead <- function() {
  rows <- lapply(overhead_data, function(r) {
    data.frame(
      instr  = if (!is.null(r$instrumentation_overhead_pct)) r$instrumentation_overhead_pct else NA_real_,
      quick  = if (!is.null(r$quickshap_overhead_pct))       r$quickshap_overhead_pct       else NA_real_,
      track  = if (!is.null(r$tracking_overhead_pct))        r$tracking_overhead_pct        else NA_real_,
      kernel = if (!is.null(r$kernelshap_multiplier))        r$kernelshap_multiplier        else NA_real_,
      exact  = if (!is.null(r$exact_multiplier))             r$exact_multiplier             else NA_real_
    )
  })
  df <- do.call(rbind, rows)

  # Median for robustness against single-config outliers
  instr_pos <- df$instr[df$instr > 0 & !is.na(df$instr)]
  med_instr <- if (length(instr_pos) > 0) median(instr_pos) else 15
  med_quick <- median(df$quick,  na.rm = TRUE)
  med_track <- median(df$track,  na.rm = TRUE)
  med_kern  <- median(df$kernel, na.rm = TRUE)
  med_exact <- median(df$exact,  na.rm = TRUE)

  df_plot <- data.frame(
    component = factor(
      c("Instrumentation", "QuickSHAP", "Tracking", "KernelSHAP", "Exact Shapley"),
      levels = rev(c("Instrumentation", "QuickSHAP", "Tracking", "KernelSHAP", "Exact Shapley"))
    ),
    cost  = c(1 + med_instr/100, 1 + med_quick/100, 1 + med_track/100, med_kern, med_exact),
    label = c(sprintf("%.0f%%", med_instr),
              sprintf("%.3f%%", med_quick),
              sprintf("%.1f%%", med_track),
              sprintf("%.0fx", med_kern),
              sprintf("%.0fx", med_exact)),
    stringsAsFactors = FALSE
  )

  bar_cols <- c(
    Instrumentation = unname(OI["blue"]),
    QuickSHAP       = unname(OI["green"]),
    Tracking        = unname(OI["skyblue"]),
    KernelSHAP      = unname(OI["orange"]),
    `Exact Shapley` = unname(OI["vermill"])
  )

  p <- ggplot(df_plot, aes(x = cost, y = component, fill = component)) +
    geom_col(width = 0.65, alpha = 0.9, show.legend = FALSE) +
    geom_text(aes(label = label, x = pmax(cost * 1.18, 1.6)),
              hjust = 0, fontface = "bold", size = 2.7, family = "serif") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.4) +
    scale_fill_manual(values = bar_cols) +
    scale_x_log10(limits = c(0.9, 120),
                  breaks = c(1, 2, 5, 10, 20, 50, 100),
                  labels = function(x) paste0(x, "x")) +
    labs(x = "Cost (x single run)", y = NULL,
         title = "Computational cost by component") +
    theme(panel.grid.major.y = element_blank())

  ggsave(file.path(OUT_DIR, "fig5_overhead.pdf"),
         p, width = 4.6, height = 3.0, device = cairo_pdf)
  cat("  ok fig5_overhead.pdf\n")
}

# ========================================================================
# FIGURE 6: DE invariance — all configs ~50/50/0
# ========================================================================
fig6_de_invariance <- function() {
  df <- build_shap_table("DE") |>
    mutate(
      function_name = factor(function_name, levels = FUNCTIONS, labels = FUNC_LABS),
      dim_lab       = factor(paste0("D=", dimension),
                             levels = paste0("D=", DIMS)),
      op_label      = factor(op_label, levels = c("Mutation", "Crossover", "Selection"))
    )

  p <- ggplot(df, aes(x = dim_lab, y = shapley_pct, fill = op_label)) +
    geom_col(position = position_dodge(0.85), width = 0.75, alpha = 0.9) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "grey60", linewidth = 0.4) +
    scale_fill_manual(values = OP_COL, name = NULL) +
    facet_wrap(~ function_name, nrow = 1) +
    coord_cartesian(ylim = c(0, 65)) +
    labs(x = NULL, y = "Shapley value (%)",
         title = "DE: invariant 50/50/0 attribution across all configurations") +
    theme(
      axis.text.x = element_text(size = rel(0.75)),
      legend.position = "top",
      legend.margin = margin(b = -5),
      panel.spacing = unit(0.15, "lines")
    )

  ggsave(file.path(OUT_DIR, "fig6_de_invariance.pdf"),
         p, width = W_FULL, height = 2.6, device = cairo_pdf)
  cat("  ok fig6_de_invariance.pdf\n")
}

# ========================================================================
# FIGURE 7: Summary — three algorithms side by side (stacked bars by dim)
# ========================================================================
fig7_summary <- function() {
  algos <- c("DE", "GA", "PSO")
  titles <- c(
    DE  = "DE\n(Coupled)",
    GA  = "GA\n(Landscape-dependent)",
    PSO = "PSO\n(Dimension-dependent)"
  )

  all_rows <- lapply(algos, function(alg) {
    df <- build_shap_table(alg)
    if (is.null(df)) return(NULL)
    df |>
      filter(!grepl("position", tolower(op_key))) |>
      # Re-normalize per (alg, func, dim) so the remaining operators sum to 100%
      group_by(algorithm, function_name, dimension) |>
      mutate(shapley_pct = if (sum(shapley_pct) > 0) shapley_pct / sum(shapley_pct) * 100 else shapley_pct) |>
      ungroup() |>
      group_by(algorithm, dimension, op_label) |>
      summarise(mean_pct = mean(shapley_pct), .groups = "drop")
  })
  df <- do.call(rbind, all_rows) |>
    mutate(
      algorithm = factor(algorithm, levels = algos, labels = titles[algos]),
      dimension = factor(paste0("D=", dimension), levels = paste0("D=", DIMS)),
      op_label  = factor(op_label, levels = c("Selection", "Crossover", "Mutation",
                                              "Velocity",  "Topology"))
    )

  p <- ggplot(df, aes(x = dimension, y = mean_pct, fill = op_label)) +
    geom_col(width = 0.55, alpha = 0.92) +
    geom_text(aes(label = ifelse(mean_pct > 3, sprintf("%.0f%%", mean_pct), ""),
                  color = mean_pct > 15),
              position = position_stack(vjust = 0.5),
              fontface = "bold", size = 2.7, family = "serif",
              show.legend = FALSE) +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "black")) +
    scale_fill_manual(values = OP_COL, name = NULL) +
    facet_wrap(~ algorithm, nrow = 1, scales = "free_x") +
    coord_cartesian(ylim = c(0, 105)) +
    labs(x = NULL, y = "Shapley value (%)") +
    theme(
      legend.position = "bottom",
      legend.margin = margin(t = -3)
    )

  ggsave(file.path(OUT_DIR, "fig7_summary_all_algorithms.pdf"),
         p, width = W_FULL, height = 3.6, device = cairo_pdf)
  cat("  ok fig7_summary_all_algorithms.pdf\n")
}

# ========================================================================
# FIGURE 8: Discriminant — XMH vs fANOVA
# ========================================================================
fig8_discriminant <- function() {
  if (is.null(exp8a) || is.null(exp8b)) {
    cat("  (skip fig8: exp8 data not found)\n")
    return(invisible(NULL))
  }

  # ---- Part A: per-cx_type Shapley breakdown ----
  cx_order <- c("sbx", "blx_alpha", "arithmetic", "uniform", "two_point", "single_point")
  cx_labs  <- c("SBX", "BLX-alpha", "Arith.", "Uniform", "Two-pt", "Single-pt")

  rows_a <- lapply(cx_order, function(cx) {
    case <- exp8a$per_cx_type[[cx]]
    if (is.null(case)) return(NULL)
    sv  <- unlist(case$shapley_values)
    tot <- sum(abs(sv))
    pct <- 100 * abs(sv) / tot
    cx_op  <- names(sv)[startsWith(names(sv), "crossover")][1]
    sel_op <- names(sv)[startsWith(names(sv), "selection")][1]
    mut_op <- names(sv)[startsWith(names(sv), "mutation")][1]
    ci <- case$confidence_intervals[[cx_op]]
    ci_lo <- 100 * (sv[cx_op] - ci[[1]]) / tot
    ci_hi <- 100 * (ci[[2]] - sv[cx_op]) / tot
    data.frame(
      cx_type  = cx,
      op_label = c("Selection", "Crossover", "Mutation"),
      shap_pct = c(pct[sel_op], pct[cx_op], pct[mut_op]),
      ci_lo    = c(NA, unname(ci_lo), NA),
      ci_hi    = c(NA, unname(ci_hi), NA),
      stringsAsFactors = FALSE
    )
  })
  dfA <- do.call(rbind, rows_a) |>
    mutate(
      cx_type  = factor(cx_type, levels = cx_order, labels = cx_labs),
      op_label = factor(op_label, levels = c("Selection", "Crossover", "Mutation"))
    )

  eta2 <- exp8a$anova$eta_squared * 100

  pA <- ggplot(dfA, aes(x = cx_type, y = shap_pct, fill = op_label)) +
    geom_col(position = position_dodge(0.85), width = 0.78, alpha = 0.92) +
    geom_errorbar(aes(ymin = pmax(shap_pct - ci_lo, 0),
                      ymax = shap_pct + ci_hi),
                  position = position_dodge(0.85), width = 0.18,
                  linewidth = 0.4, color = "grey20",
                  na.rm = TRUE) +
    scale_fill_manual(values = OP_COL, name = NULL) +
    coord_cartesian(ylim = c(0, 78)) +
    labs(
      x = "Crossover operator", y = "Shapley value (%)",
      title = "Part A: vary categorical crossover operator"
    ) +
    annotate("label",
             x = 0.55, y = 76, hjust = 0, vjust = 1,
             label = sprintf("fANOVA on hyperparameters: 0%%\n(nothing to decompose)\n\neta^2 on cx type: %.1f%%\n(between-type variance)", eta2),
             size = 2.3, family = "serif",
             fill = "#fffae6", label.size = 0.3, color = "#8a6d10") +
    theme(
      axis.text.x = element_text(angle = 20, hjust = 1),
      legend.position = c(0.99, 0.99),
      legend.justification = c(1, 1)
    )

  # ---- Part B: cx_rate sweep ----
  # JSON keys are formatted as "0.10", "0.20", ... so we keep them as strings
  # and only parse to numeric for plotting (as.character(0.10) -> "0.1" mismatch).
  rate_keys <- names(exp8b$per_cx_rate)
  rate_keys <- rate_keys[order(as.numeric(rate_keys))]
  rows_b <- lapply(rate_keys, function(rk) {
    case <- exp8b$per_cx_rate[[rk]]
    if (is.null(case)) return(NULL)
    sv  <- unlist(case$shapley_values)
    tot <- sum(abs(sv))
    cx_op <- names(sv)[startsWith(names(sv), "crossover")][1]
    if (is.na(cx_op)) return(NULL)
    fit <- unlist(case$final_fitness_per_run)
    fit <- fit[is.finite(fit)]
    data.frame(
      cx_rate = as.numeric(rk),
      cx_pct  = 100 * abs(sv[[cx_op]]) / tot,
      fitness = if (length(fit) > 0) mean(fit) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  dfB <- do.call(rbind, rows_b)
  row.names(dfB) <- NULL

  r2  <- exp8b$fanova_on_cx_rate$quadratic_r_squared * 100
  rho <- exp8b$xmh_cx_shapley_vs_rate$spearman_rho_rate_vs_cx_pct

  # Twin-axis trick: scale fitness to cx_pct range
  fit_min <- min(dfB$fitness); fit_max <- max(dfB$fitness)
  cx_min  <- min(dfB$cx_pct);  cx_max  <- max(dfB$cx_pct)
  scale_fn <- function(x) (x - fit_min) / (fit_max - fit_min) * (cx_max - cx_min) + cx_min
  inv_fn   <- function(y) (y - cx_min) / (cx_max - cx_min) * (fit_max - fit_min) + fit_min

  pB <- ggplot(dfB, aes(x = cx_rate)) +
    geom_line(aes(y = cx_pct, color = "XMH crossover phi (%)"),
              linewidth = 0.9) +
    geom_point(aes(y = cx_pct, color = "XMH crossover phi (%)"),
               size = 2.4) +
    geom_line(aes(y = scale_fn(fitness), color = "Final fitness (mean)"),
              linewidth = 0.8, linetype = "dashed") +
    geom_point(aes(y = scale_fn(fitness), color = "Final fitness (mean)"),
               size = 2.2, shape = 15) +
    scale_color_manual(
      values = c("XMH crossover phi (%)" = unname(OI["blue"]),
                 "Final fitness (mean)"  = unname(OI["vermill"])),
      name = NULL
    ) +
    scale_y_continuous(
      name = "XMH crossover phi (%)",
      sec.axis = sec_axis(~ inv_fn(.), name = "Final fitness (Rastrigin D=30)")
    ) +
    scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
    labs(
      x = "Crossover probability (cx rate)",
      title = "Part B: vary continuous cx rate (SBX fixed)"
    ) +
    annotate("label",
             x = 0.12, y = cx_max - 0.02 * (cx_max - cx_min),
             hjust = 0, vjust = 1,
             label = sprintf("fANOVA on cx rate: R^2 = %.1f%%\n(detects sensitivity)\n\nXMH rho(cx rate, phi_cx) = %.2f\n(smooth modulation)", r2, rho),
             size = 2.3, family = "serif",
             fill = "#e8f4fd", label.size = 0.3, color = "#1c5e8d") +
    theme(
      axis.title.y.left  = element_text(color = unname(OI["blue"])),
      axis.title.y.right = element_text(color = unname(OI["vermill"])),
      axis.text.y.left   = element_text(color = unname(OI["blue"])),
      axis.text.y.right  = element_text(color = unname(OI["vermill"])),
      legend.position = c(0.99, 0.45),
      legend.justification = c(1, 0.5),
      legend.background = element_rect(fill = alpha("white", 0.9), color = NA),
      legend.key.height = unit(0.35, "cm")
    )

  p <- pA + pB + plot_layout(ncol = 2, widths = c(1, 1)) +
    plot_annotation(
      title = "XMH vs fANOVA: complementary, not equivalent (GA, Rastrigin D=30)",
      theme = theme(plot.title = element_text(family = "serif", face = "bold",
                                              size = 11, hjust = 0.5))
    )

  ggsave(file.path(OUT_DIR, "fig8_discriminant.pdf"),
         p, width = W_FULL, height = 3.6, device = cairo_pdf)
  cat("  ok fig8_discriminant.pdf\n")
}

# ========================================================================
# FIGURE 9: Budget sensitivity (2x2)
# ========================================================================
fig9_budget <- function() {
  if (is.null(exp9a) || is.null(exp9b)) {
    cat("  (skip fig9: exp9 data not found)\n")
    return(invisible(NULL))
  }

  get_pct_op <- function(case) {
    sv  <- unlist(case$shapley_values)
    tot <- sum(abs(sv))
    if (tot < 1e-15) return(NULL)
    out <- list()
    for (op in names(sv)) {
      key <- if (startsWith(op, "selection")) "Selection"
             else if (startsWith(op, "crossover")) "Crossover"
             else if (startsWith(op, "mutation")) "Mutation"
             else op
      out[[key]] <- 100 * abs(sv[op]) / tot
    }
    out
  }

  # ---- (a) Convergence vs max_gen on Rastrigin ----
  rast_a   <- exp9a$per_func$rastrigin
  level_a  <- unlist(exp9a$levels)
  longest  <- as.character(max(level_a))
  traces_a <- do.call(rbind, lapply(rast_a[[longest]]$convergence_traces, unlist))
  gens_a   <- seq_len(ncol(traces_a)) - 1
  df_a <- data.frame(
    gen    = gens_a,
    median = apply(traces_a, 2, median),
    p25    = apply(traces_a, 2, function(x) quantile(x, 0.25)),
    p75    = apply(traces_a, 2, function(x) quantile(x, 0.75))
  )

  ylim_top <- max(df_a$p75, na.rm = TRUE)

  pA <- ggplot(df_a, aes(x = gen)) +
    geom_ribbon(aes(ymin = p25, ymax = p75), fill = "grey50", alpha = 0.25) +
    geom_line(aes(y = median), color = "grey20", linewidth = 0.7) +
    geom_vline(xintercept = level_a, color = "grey50",
               linetype = "dotted", linewidth = 0.3) +
    annotate("text",
             x = level_a, y = ylim_top * 0.85,
             label = paste0("g=", level_a),
             size = 2.0, family = "serif", angle = 90,
             hjust = 1, vjust = -0.3, color = "grey40") +
    scale_y_continuous(trans = scales::pseudo_log_trans(sigma = 0.1),
                       breaks = c(0, 1, 10, 100, 1000)) +
    labs(x = "Generation", y = "Best fitness (Rastrigin)",
         title = "(a) Convergence vs max gen (pop=50)")

  # ---- (b) Convergence vs pop_size on Rastrigin ----
  rast_b <- exp9b$per_func$rastrigin
  pops   <- unlist(exp9b$levels)

  df_b <- do.call(rbind, lapply(pops, function(pop) {
    tr <- do.call(rbind, lapply(rast_b[[as.character(pop)]]$convergence_traces, unlist))
    data.frame(
      gen    = seq_len(ncol(tr)) - 1,
      median = apply(tr, 2, median),
      pop    = factor(pop, levels = pops)
    )
  }))

  pB <- ggplot(df_b, aes(x = gen, y = median, color = pop)) +
    geom_line(linewidth = 0.7) +
    scale_color_viridis_d(name = "pop", option = "viridis", begin = 0.1, end = 0.85) +
    scale_y_continuous(trans = scales::pseudo_log_trans(sigma = 0.1),
                       breaks = c(0, 1, 10, 100, 1000)) +
    labs(x = "Generation", y = "Best fitness (Rastrigin)",
         title = "(b) Convergence vs pop size (max gen=200)") +
    theme(legend.position = c(0.99, 0.99),
          legend.justification = c(1, 1),
          legend.key.height = unit(0.25, "cm"))

  # ---- (c) Shapley% vs max_gen ----
  rows_c <- list()
  for (func in unlist(exp9a$functions)) {
    per_lvl <- exp9a$per_func[[func]]
    for (g in level_a) {
      pct <- get_pct_op(per_lvl[[as.character(g)]])
      if (is.null(pct)) next
      for (op in names(pct)) {
        rows_c[[length(rows_c) + 1L]] <- data.frame(
          func = func, gen = g, op = op, pct = pct[[op]],
          stringsAsFactors = FALSE
        )
      }
    }
  }
  dfC <- do.call(rbind, rows_c) |>
    mutate(
      op   = factor(op, levels = c("Selection", "Crossover", "Mutation")),
      func = factor(func, levels = c("sphere", "rastrigin"),
                    labels = c("Sphere", "Rastrigin"))
    )

  pC <- ggplot(dfC, aes(x = gen, y = pct, color = op,
                        linetype = func, shape = func,
                        group = interaction(op, func))) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.8, fill = "white") +
    scale_color_manual(values = OP_COL, name = "Operator") +
    scale_linetype_manual(values = c(Sphere = "dashed", Rastrigin = "solid"),
                          name = "Function") +
    scale_shape_manual(values = c(Sphere = 16, Rastrigin = 15),
                       name = "Function") +
    scale_x_log10(breaks = level_a) +
    coord_cartesian(ylim = c(0, 70)) +
    labs(x = "max generations", y = "Shapley value (%)",
         title = "(c) Shapley vs max gen") +
    guides(
      color    = guide_legend(order = 1, nrow = 1),
      linetype = guide_legend(order = 2, nrow = 1),
      shape    = guide_legend(order = 2, nrow = 1)
    )

  # ---- (d) Shapley% vs pop_size ----
  rows_d <- list()
  for (func in unlist(exp9b$functions)) {
    per_lvl <- exp9b$per_func[[func]]
    for (pop in pops) {
      pct <- get_pct_op(per_lvl[[as.character(pop)]])
      if (is.null(pct)) next
      for (op in names(pct)) {
        rows_d[[length(rows_d) + 1L]] <- data.frame(
          func = func, pop = pop, op = op, pct = pct[[op]],
          stringsAsFactors = FALSE
        )
      }
    }
  }
  dfD <- do.call(rbind, rows_d) |>
    mutate(
      op   = factor(op, levels = c("Selection", "Crossover", "Mutation")),
      func = factor(func, levels = c("sphere", "rastrigin"),
                    labels = c("Sphere", "Rastrigin"))
    )

  pD <- ggplot(dfD, aes(x = pop, y = pct, color = op,
                        linetype = func, shape = func,
                        group = interaction(op, func))) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.8, fill = "white") +
    scale_color_manual(values = OP_COL, name = "Operator", guide = "none") +
    scale_linetype_manual(values = c(Sphere = "dashed", Rastrigin = "solid"),
                          name = "Function", guide = "none") +
    scale_shape_manual(values = c(Sphere = 16, Rastrigin = 15),
                       name = "Function", guide = "none") +
    scale_x_log10(breaks = pops) +
    coord_cartesian(ylim = c(0, 90)) +
    labs(x = "pop size", y = "Shapley value (%)",
         title = "(d) Shapley vs pop size")

  p <- (pA + pB) / (pC + pD) +
    plot_layout(guides = "collect") +
    plot_annotation(
      title = "Budget sensitivity: operator attribution shifts with max gen and pop size",
      theme = theme(plot.title = element_text(family = "serif", face = "bold",
                                              size = 11, hjust = 0.5))
    ) &
    theme(legend.position = "bottom",
          legend.box = "vertical",
          legend.margin = margin(t = 0, b = 0),
          legend.box.margin = margin(t = -3, b = -3),
          legend.spacing.x = unit(0.25, "cm"),
          legend.spacing.y = unit(0, "cm"),
          legend.key.height = unit(0.20, "cm"),
          legend.key.width = unit(0.65, "cm"),
          legend.text = element_text(size = rel(0.80)),
          legend.title = element_text(size = rel(0.85)))

  ggsave(file.path(OUT_DIR, "fig9_budget.pdf"),
         p, width = W_FULL, height = 5.6, device = cairo_pdf)
  cat("  ok fig9_budget.pdf\n")
}

# ========================================================================
# Run all
# ========================================================================
main <- function() {
  cat("Generating R/ggplot2 figures for XMH paper...\n")
  fig1_ga_heatmap()
  fig2_pso_heatmap()
  fig3_dimensionality_effect()
  fig4_method_comparison()
  fig5_overhead()
  fig6_de_invariance()
  fig7_summary()
  fig8_discriminant()
  fig9_budget()
  cat(sprintf("\nAll figures saved to %s/\n", OUT_DIR))
}

if (!interactive()) main()

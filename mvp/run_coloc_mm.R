#!/usr/bin/env Rscript
# run_coloc_mm.R -- mismatch-aware coloc-SuSiE for one locus, exactly as configured in the benchmark
# ("coloc_susie_mm": susieR 0.16.6 susie_rss with R_finite = n_ref and R_mismatch = "eb_mix", then coloc 5.2.3
# coloc.bf_bf on the credible-set rows of lbf_variable; coloc.abf fallback), plus the validated three-way call.
#
# Provenance of every setting (paths relative to the repository root):
#   src/baselines/methods.py  class ColocSusie (mismatch=True) / ColocSusieMM: L = 10, p1 = p2 = 1e-4,
#                             p12 = 5e-6, R_finite = n_ref, R_mismatch = "eb_mix", max_iter = package default,
#                             n_t = median of the per-SNP N (src/baselines/locus.py scalar_n),
#                             LD symmetrised 0.5 * (R + t(R)) (src/baselines/locus.py dense_ld),
#                             fallback coloc.abf with W = 0.15, p12 = 5e-6 when a trait has no credible set.
#   src/baselines/r/worker.R  fit_susie / h_susie_coloc: susie_rss(z, R, n, L, R_finite, R_mismatch) and
#                             coloc.bf_bf(lbf_variable[cs_index, ], ...) "as coloc::coloc.susie".
#   src/baselines/abf.py      the coloc.abf formulas used for the fallback (validated against coloc 5.2.3).
#   src/eval/adapters.py      coloc_susie(): s = max_k PP4/(PP3+PP4); P_S = max_k PP4;
#                             P_D = (1 - max_k PP4) * max_k PP3; P_W = max(0, 1 - P_S - P_D);
#                             coloc_abf() (fallback): s = PP4/(PP3+PP4), P_S = PP4, P_D = PP3, P_W = PP0+PP1+PP2.
#   src/eval/metrics.py       calls_from_arrays(): S if P_S >= t_S; else D if P_D >= t_D (E-gate not applicable:
#                             this method reports no E1/E2); else "insufficient".
#   results/phase2_validation/m4_valid_baselines_v3/operating_points_p2.json (results/baselines_m4/m4_valid_v3/
#   in the original project): t_S = 0.8402818292911564, t_D = 0.7322585039086077.
#
# Usage:
#   Rscript mvp/run_coloc_mm.R --sumstats S.tsv[.gz] (--ld R.txt[.gz]|R.rds | --ld-factor X.txt[.gz]|X.rds) \
#       --n-ref N_REF [--rho-hat RHO] [--out PREFIX] [--estimate-sdy] [--allow-other-versions]
# See mvp/README.md for the input format.

suppressMessages(library(jsonlite))
options(warn = 1, stringsAsFactors = FALSE)

T_S <- 0.8402818292911564
T_D <- 0.7322585039086077
L_EFFECTS <- 10
P1 <- 1e-4
P2 <- 1e-4
P12 <- 5e-6
W_FALLBACK <- 0.15
R_MISMATCH <- "eb_mix"

usage <- function(msg = NULL) {
  if (!is.null(msg)) cat("error:", msg, "\n", file = stderr())
  cat("usage: Rscript run_coloc_mm.R --sumstats FILE (--ld FILE | --ld-factor FILE) --n-ref INT",
      "[--rho-hat NUM] [--out PREFIX] [--estimate-sdy] [--allow-other-versions]\n", file = stderr())
  quit(status = 2)
}

parse_args <- function(argv) {
  a <- list(estimate_sdy = FALSE, allow_other_versions = FALSE, rho_hat = NA_real_, out = NULL)
  i <- 1
  while (i <= length(argv)) {
    k <- argv[i]
    flag <- function() { if (i + 1 > length(argv)) usage(paste("missing value for", k)); argv[i + 1] }
    if (k == "--sumstats") { a$sumstats <- flag(); i <- i + 2 }
    else if (k == "--ld") { a$ld <- flag(); i <- i + 2 }
    else if (k == "--ld-factor") { a$ld_factor <- flag(); i <- i + 2 }
    else if (k == "--n-ref") { a$n_ref <- as.numeric(flag()); i <- i + 2 }
    else if (k == "--rho-hat") { a$rho_hat <- as.numeric(flag()); i <- i + 2 }
    else if (k == "--out") { a$out <- flag(); i <- i + 2 }
    else if (k == "--estimate-sdy") { a$estimate_sdy <- TRUE; i <- i + 1 }
    else if (k == "--allow-other-versions") { a$allow_other_versions <- TRUE; i <- i + 1 }
    else if (k %in% c("-h", "--help")) usage()
    else usage(paste("unknown argument", k))
  }
  if (is.null(a$sumstats)) usage("--sumstats is required")
  if (is.null(a[["ld"]]) == is.null(a[["ld_factor"]])) usage("give exactly one of --ld and --ld-factor")
  if (is.null(a$n_ref) || !is.finite(a$n_ref) || a$n_ref < 2) usage("--n-ref must be a number >= 2")
  a
}

# ------------------------------------------------------------------------------------------------ input
read_matrix <- function(path) {
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    m <- readRDS(path)
    m <- as.matrix(m)
    storage.mode(m) <- "double"
    return(list(m = m, names = colnames(m)))
  }
  first <- readLines(path, n = 1)
  tok <- strsplit(trimws(first), "[[:space:],]+")[[1]]
  has_header <- any(is.na(suppressWarnings(as.numeric(tok))))
  df <- utils::read.table(path, header = has_header, sep = "", check.names = FALSE, colClasses = "numeric")
  m <- as.matrix(df)
  dimnames(m) <- NULL
  list(m = m, names = if (has_header) tok else NULL)
}

read_sumstats <- function(path) {
  d <- utils::read.delim(path, check.names = FALSE, colClasses = c(snp = "character"))
  need <- c("snp", "beta1", "se1", "n1", "beta2", "se2", "n2")
  miss <- setdiff(need, names(d))
  if (length(miss)) stop("sumstats is missing columns: ", paste(miss, collapse = ", "))
  for (k in setdiff(intersect(c(need, "maf1", "maf2"), names(d)), "snp")) {
    d[[k]] <- as.numeric(d[[k]])
    if (any(!is.finite(d[[k]]))) stop("sumstats column ", k, " has non-finite values")
  }
  if (any(d$se1 <= 0) || any(d$se2 <= 0)) stop("non-positive standard errors")
  if (any(d$n1 <= 0) || any(d$n2 <= 0)) stop("non-positive sample sizes")
  if (anyDuplicated(d$snp)) stop("duplicated snp ids")
  if (nrow(d) < 2) stop("fewer than 2 variants")
  d
}

check_versions <- function(allow) {
  v <- c(susieR = as.character(packageVersion("susieR")), coloc = as.character(packageVersion("coloc")))
  want <- c(susieR = "0.16.6", coloc = "5.2.3")
  bad <- names(v)[v != want]
  if (length(bad)) {
    msg <- paste0(bad, " ", v[bad], " found, ", want[bad], " expected", collapse = "; ")
    if (!allow) stop(msg, " (point R_LIBS at the right library, or pass --allow-other-versions)")
    warning(msg, " (--allow-other-versions given: results may differ from the benchmark)")
  }
  as.list(v)
}

# ------------------------------------------------------------------------------------------------ coloc.abf
# Same formulas as src/baselines/abf.py (coloc 5.2.3 coloc.abf with prior effect SD W, sdY = 1).
logsum <- function(x) { m <- max(x); m + log(sum(exp(x - m))) }
logdiff <- function(x, y) { m <- max(x, y); m + log(exp(x - m) - exp(y - m)) }
adjust_prior <- function(p, nsnps) if (nsnps * p >= 1) 1 / (nsnps + 1) else p
labf <- function(z, se, W) { V <- se^2; r <- W^2 / (W^2 + V); 0.5 * (log1p(-r) + r * z^2) }
abf_pp <- function(z1, se1, z2, se2, W, p1, p2, p12) {
  n <- length(z1)
  a1 <- adjust_prior(p1, n); a2 <- adjust_prior(p2, n); a12 <- adjust_prior(p12, n)
  l1 <- labf(z1, se1, W); l2 <- labf(z2, se2, W)
  ls1 <- logsum(l1); ls2 <- logsum(l2); ls12 <- logsum(l1 + l2)
  lH <- c(0, log(a1) + ls1, log(a2) + ls2, log(a1) + log(a2) + logdiff(ls1 + ls2, ls12), log(a12) + ls12)
  exp(lH - logsum(lH))
}
abf_pp_sdy_estimated <- function(d) {
  # coloc's own sdY estimate from MAF, N and varbeta (option --estimate-sdy; not the benchmark configuration)
  if (!all(c("maf1", "maf2") %in% names(d))) stop("--estimate-sdy needs maf1 and maf2 columns")
  snp <- paste0("s", seq_len(nrow(d)))
  mk <- function(b, se, n, maf) list(beta = b, varbeta = se^2, N = stats::median(n), MAF = maf, type = "quant",
                                     snp = snp)
  # coloc.abf prints its summary and data-check warnings; keep stdout clean for the JSON result
  invisible(utils::capture.output(
    r <- suppressWarnings(suppressMessages(coloc::coloc.abf(mk(d$beta1, d$se1, d$n1, d$maf1),
                                                            mk(d$beta2, d$se2, d$n2, d$maf2),
                                                            p1 = P1, p2 = P2, p12 = P12)))))
  unname(as.numeric(r$summary[2:6]))
}

ratio_or_half <- function(num, other) {
  den <- num + other
  if (!is.finite(den) || den <= 0 || !is.finite(num)) 0.5 else num / den
}

# ------------------------------------------------------------------------------------------------ main
main <- function() {
  a <- parse_args(commandArgs(trailingOnly = TRUE))
  suppressMessages({ library(susieR); library(coloc) })
  versions <- check_versions(a$allow_other_versions)
  d <- read_sumstats(a$sumstats)
  p <- nrow(d)
  if (!is.null(a[["ld"]])) {
    lm <- read_matrix(a[["ld"]])
    R <- lm$m
    if (nrow(R) != ncol(R)) stop("LD matrix is not square: ", nrow(R), " x ", ncol(R))
    if (nrow(R) != p) stop("LD matrix has ", nrow(R), " variants, sumstats has ", p)
    if (!is.null(lm$names) && !identical(as.character(lm$names), d$snp)) stop("LD header SNP ids differ from sumstats snp order")
    if (any(!is.finite(R))) stop("LD matrix has non-finite values")
    if (max(abs(R - t(R))) > 1e-6) stop("LD matrix is not symmetric (max |R - t(R)| > 1e-6)")
    ld_source <- "dense"
  } else {
    lm <- read_matrix(a$ld_factor)
    X <- lm$m
    if (ncol(X) != p) stop("LD factor has ", ncol(X), " columns, sumstats has ", p, " variants")
    if (!is.null(lm$names) && !identical(as.character(lm$names), d$snp)) stop("LD factor header SNP ids differ from sumstats snp order")
    if (any(!is.finite(X))) stop("LD factor has non-finite values")
    if (nrow(X) != a$n_ref) stop("LD factor has ", nrow(X), " rows but --n-ref is ", a$n_ref)
    R <- crossprod(X) / a$n_ref        # R = X' X / n_ref (src/baselines/locus.py dense_ld, factor form)
    ld_source <- "factor"
  }
  R <- 0.5 * (R + t(R))
  z1 <- d$beta1 / d$se1
  z2 <- d$beta2 / d$se2
  n1 <- stats::median(d$n1)
  n2 <- stats::median(d$n2)

  warns <- character(0)
  t0 <- proc.time()[["elapsed"]]
  fit <- function(z, n) withCallingHandlers(
    suppressMessages(susieR::susie_rss(z = z, R = R, n = n, L = L_EFFECTS, R_finite = a$n_ref, R_mismatch = R_MISMATCH)),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  f1 <- fit(z1, n1)
  f2 <- fit(z2, n2)
  ncs <- function(f) if (is.null(f$sets$cs)) 0L else length(f$sets$cs)
  tinfo <- function(f) {
    k <- ncs(f)
    list(n_cs = k, converged = isTRUE(f$converged), niter = f$niter,
         cs_index = if (k) as.integer(f$sets$cs_index) else integer(0),
         cs_size = if (k) as.integer(sapply(f$sets$cs, length)) else integer(0), max_pip = max(f$pip))
  }
  t1 <- tinfo(f1); t2 <- tinfo(f2)
  snp <- paste0("s", seq_len(p))
  PP <- NULL
  pairs <- NULL
  if (t1$n_cs > 0 && t2$n_cs > 0) {
    bf1 <- f1$lbf_variable[t1$cs_index, , drop = FALSE]
    bf2 <- f2$lbf_variable[t2$cs_index, , drop = FALSE]
    colnames(bf1) <- snp
    colnames(bf2) <- snp
    cb <- suppressMessages(coloc::coloc.bf_bf(bf1, bf2, p1 = P1, p2 = P2, p12 = P12))
    sm <- as.data.frame(cb$summary)
    PP <- unname(as.matrix(sm[, c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")]))
    pairs <- data.frame(idx1 = as.integer(sm$idx1), idx2 = as.integer(sm$idx2),
                        hit1 = d$snp[match(sm$hit1, snp)], hit2 = d$snp[match(sm$hit2, snp)],
                        PP.H0 = PP[, 1], PP.H1 = PP[, 2], PP.H2 = PP[, 3], PP.H3 = PP[, 4], PP.H4 = PP[, 5])
    PP <- PP[apply(PP, 1, function(r) all(is.finite(r))), , drop = FALSE]
    if (nrow(PP) == 0) PP <- NULL
  }
  fallback <- is.null(PP)
  if (!fallback) {
    s <- max(apply(PP, 1, function(r) ratio_or_half(r[5], r[4])))
    PS <- max(PP[, 5])
    PD <- (1 - PS) * max(PP[, 4])
    PW <- max(0, 1 - PS - PD)
    fb <- NULL
  } else {
    fb <- if (a$estimate_sdy) abf_pp_sdy_estimated(d) else abf_pp(z1, d$se1, z2, d$se2, W_FALLBACK, P1, P2, P12)
    s <- ratio_or_half(fb[5], fb[4])
    PS <- fb[5]
    PD <- fb[4]
    PW <- fb[1] + fb[2] + fb[3]
  }
  call <- if (PS >= T_S) "S" else if (PD >= T_D) "D" else "insufficient"
  status <- if (t1$converged && t2$converged) "ok" else "nonconverged"
  out <- list(
    P_S = PS, P_D = PD, P_W = PW, s = s, call = call, status = status, fallback = fallback,
    fallback_PP = if (fallback) fb else NULL,
    fallback_sdY = if (fallback) (if (a$estimate_sdy) "estimated (coloc, from MAF and N)" else "1 (benchmark configuration)") else NULL,
    thresholds = list(t_S = T_S, t_D = T_D,
                      source = "results/phase2_validation/m4_valid_baselines_v3/operating_points_p2.json (coloc_susie_mm OP1 = OP2, M4 validation)"),
    trait1 = t1, trait2 = t2, pairs = pairs,
    inputs = list(n_variants = p, n1 = n1, n2 = n2, n_ref = a$n_ref, ld_source = ld_source, rho_hat = a$rho_hat,
                  rho_hat_note = "recorded only: coloc_susie_mm has no sample-overlap input (as in the benchmark)"),
    config = list(method = "coloc_susie_mm", L = L_EFFECTS, R_finite = a$n_ref, R_mismatch = R_MISMATCH,
                  p1 = P1, p2 = P2, p12 = P12, fallback_W = W_FALLBACK),
    versions = c(versions, list(R = R.version.string)),
    warnings = unique(warns), elapsed_s = proc.time()[["elapsed"]] - t0)
  js <- toJSON(out, auto_unbox = TRUE, digits = I(17), na = "null", null = "null", pretty = TRUE,
               dataframe = "rows")
  if (is.null(a$out)) {
    cat(js, "\n")
  } else {
    writeLines(js, paste0(a$out, ".json"))
    tsv <- data.frame(P_S = PS, P_D = PD, P_W = PW, s = s, call = call, status = status, fallback = fallback,
                      n_cs1 = t1$n_cs, n_cs2 = t2$n_cs, n_variants = p, n_ref = a$n_ref, rho_hat = a$rho_hat)
    utils::write.table(format(tsv, digits = 17), paste0(a$out, ".tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    cat(sprintf("call=%s P_S=%.6g P_D=%.6g fallback=%s -> %s.json, %s.tsv\n", call, PS, PD, fallback, a$out, a$out))
  }
}

tryCatch(main(), error = function(e) { cat("error:", conditionMessage(e), "\n", file = stderr()); quit(status = 1) })

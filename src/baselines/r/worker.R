# Persistent R worker for the baseline wrappers (src/baselines/rworker.py).
# Protocol: one JSON request per line on stdin:
#   {"id": int, "fn": "<handler>", "args": {...}, "arrays": {"name": {"file": path, "dim": [d1, (d2)]}}}
# Arrays are raw little-endian float64 files written column-major by Python.
# Reply: exactly one line on stdout "@@RESP <json>" with fields id, ok, result | error, warnings, messages, elapsed.
# Any other output produced by packages (cat/print) is diverted to the log (stderr) while a handler runs.
# The library set (susieR 0.14.2 main lib, or susieR 0.16.6 / mvsusieR libs) is chosen by R_LIBS at spawn time.
suppressMessages(library(jsonlite))
options(warn = 1, stringsAsFactors = FALSE)

read_arr <- function(spec) {
  d <- as.integer(unlist(spec$dim))
  n <- prod(d)
  con <- file(spec$file, "rb")
  on.exit(close(con))
  x <- readBin(con, what = "double", n = n, size = 8, endian = "little")
  if (length(x) != n) stop(sprintf("array %s: read %d of %d values", spec$file, length(x), n))
  if (length(d) == 2) dim(x) <- d
  x
}

lib_info <- function() {
  pk <- c("susieR", "coloc", "mvsusieR", "colocboost", "prop.coloc", "colocPropTest")
  v <- sapply(pk, function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else NA)
  path <- sapply(pk, function(p) tryCatch(find.package(p), error = function(e) NA))
  list(versions = as.list(v), paths = as.list(path), R = R.version.string,
       blas = tryCatch(extSoftVersion()[["BLAS"]], error = function(e) NA),
       pid = Sys.getpid())
}

# ---------------------------------------------------------------- coloc.abf (reference for validation)
h_coloc_abf <- function(a, A) {
  suppressMessages(library(coloc))
  snp <- paste0("s", seq_along(A$beta1))
  d1 <- list(beta = A$beta1, varbeta = A$varbeta1, type = "quant", sdY = a$sdY, snp = snp)
  d2 <- list(beta = A$beta2, varbeta = A$varbeta2, type = "quant", sdY = a$sdY, snp = snp)
  r <- coloc.abf(d1, d2, p1 = a$p1, p2 = a$p2, p12 = a$p12)
  # merge() may reorder rows by snp name; map back to input order
  o <- match(snp, r$results$snp)
  list(pp = unname(as.numeric(r$summary[2:6])), snp_pp_h4 = r$results$SNP.PP.H4[o])
}

# ---------------------------------------------------------------- coloc-SuSiE (both susieR versions)
fit_susie <- function(z, R, n, a) {
  args <- list(z = z, R = R, n = n, L = a$L)
  if (!is.null(a$max_iter)) args$max_iter <- a$max_iter
  if (isTRUE(a$mismatch)) {
    args$R_finite <- a$n_ref
    args$R_mismatch <- a$R_mismatch
  }
  do.call(susieR::susie_rss, args)
}

susie_summary <- function(f, R_mismatch_on) {
  ncs <- if (is.null(f$sets$cs)) 0L else length(f$sets$cs)
  out <- list(n_cs = ncs, converged = isTRUE(f$converged), niter = f$niter,
              cs_index = if (ncs) as.integer(f$sets$cs_index) else integer(0),
              purity_min = if (ncs) as.numeric(f$sets$purity[, 1]) else numeric(0),
              cs_size = if (ncs) as.integer(sapply(f$sets$cs, length)) else integer(0),
              max_pip = max(f$pip))
  if (R_mismatch_on && !is.null(f$R_finite_diagnostics)) {
    d <- f$R_finite_diagnostics
    keep <- c("B", "effective_rank", "r_over_B", "lambda_bias", "B_corrected", "Q_art", "artifact_flag",
              "R_sensitivity_flag", "R_reliability_flag", "SER_fallback")
    dd <- list()
    for (k in keep) if (!is.null(d[[k]]) && length(d[[k]]) == 1) dd[[k]] <- d[[k]]
    out$R_finite_diagnostics <- dd
  }
  out
}

h_susie_coloc <- function(a, A) {
  suppressMessages(library(coloc))
  p <- length(A$z1)
  snp <- paste0("s", seq_len(p))
  R <- A$R
  f1 <- fit_susie(A$z1, R, a$n1, a)
  f2 <- fit_susie(A$z2, R, a$n2, a)
  s1 <- susie_summary(f1, isTRUE(a$mismatch))
  s2 <- susie_summary(f2, isTRUE(a$mismatch))
  res <- list(susieR = as.character(packageVersion("susieR")), t1 = s1, t2 = s2, pairs = NULL)
  if (s1$n_cs > 0 && s2$n_cs > 0) {
    # as coloc::coloc.susie: lbf rows of the credible-set effects only, then coloc.bf_bf
    bf1 <- f1$lbf_variable[s1$cs_index, , drop = FALSE]
    bf2 <- f2$lbf_variable[s2$cs_index, , drop = FALSE]
    colnames(bf1) <- snp
    colnames(bf2) <- snp
    cb <- coloc.bf_bf(bf1, bf2, p1 = a$p1, p2 = a$p2, p12 = a$p12)
    sm <- as.data.frame(cb$summary)
    res$pairs <- list(
      idx1 = as.integer(sm$idx1), idx2 = as.integer(sm$idx2),
      hit1 = as.integer(match(sm$hit1, snp)), hit2 = as.integer(match(sm$hit2, snp)),
      PP = unname(as.matrix(sm[, c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")])))
  }
  res
}

# ---------------------------------------------------------------- mvSuSiE-RSS
h_mvsusie <- function(a, A) {
  suppressMessages({ library(susieR); library(mvsusieR) })
  Z <- A$Z
  V <- A$V
  prior <- create_mixture_prior(R = 2)
  args <- list(Z = Z, R = A$R, varY = V, prior_variance = prior, residual_variance = V,
               estimate_residual_variance = FALSE, estimate_prior_variance = TRUE, L = a$L)
  if (!is.null(a$N)) {  # native common-N path (used only by the equivalence test)
    args$varY <- NULL
    args$N <- a$N
  }
  fit <- do.call(mvsusie_rss, args)
  cs_eff <- integer(0)
  if (!is.null(fit$sets$cs) && length(fit$sets$cs) > 0) cs_eff <- as.integer(fit$sets$cs_index)
  sel <- fit$single_effect_lfsr
  list(mvsusieR = as.character(packageVersion("mvsusieR")), susieR = as.character(packageVersion("susieR")),
       single_effect_lfsr = unname(as.matrix(sel)), cs_index = cs_eff,
       converged = isTRUE(fit$converged), niter = fit$niter,
       V_est = if (!is.null(fit$V)) as.numeric(fit$V) else NULL,
       max_pip = max(fit$pip), pip = as.numeric(fit$pip))
}

# ---------------------------------------------------------------- coloc documented example (wrapper validation)
h_coloc_example <- function(a, A) {
  suppressMessages({ library(coloc); library(susieR) })
  data(coloc_test_data, package = "coloc", envir = environment())
  D3 <- coloc_test_data$D3
  D4 <- coloc_test_data$D4
  S3 <- runsusie(D3)
  S4 <- runsusie(D4)
  r <- coloc.susie(S3, S4)
  sm <- as.data.frame(r$summary)
  list(beta3 = D3$beta, varbeta3 = D3$varbeta, N3 = D3$N, beta4 = D4$beta, varbeta4 = D4$varbeta, N4 = D4$N,
       MAF3 = D3$MAF, MAF4 = D4$MAF, LD = as.numeric(D3$LD), LD4_equal = isTRUE(all.equal(D3$LD, D4$LD)),
       PP = unname(as.matrix(sm[, c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")])),
       hit1 = as.integer(match(sm$hit1, D3$snp)), hit2 = as.integer(match(sm$hit2, D4$snp)))
}

# ---------------------------------------------------------------- ColocBoost (pairwise)
# ColocBoost 1.0.10 bug workaround (README decision 18). In get_robust_colocalization, a CoS whose outcome set
# becomes empty gets the placeholder weight matrix(1, nrow) WITHOUT column names; get_integrated_weight then takes
# idx = integer(0) from the (missing) names and solve(residual_correlation[idx, idx]) stops with "'a' is 0-diml",
# so the whole fit fails whenever residual_correlation is given. The patch routes ONLY that unnamed placeholder
# through the package's own residual_correlation = NULL branch (what the package does when no correlation is
# given); every other call is unchanged. Uses are counted and returned (extra.cb_patch_uses).
.cb_patch <- new.env()
.cb_patch$uses <- 0L
patch_colocboost <- function() {
  ns <- asNamespace("colocboost")
  if (isTRUE(attr(get("get_integrated_weight", envir = ns), "baselines_patch"))) return(invisible())
  orig <- get("get_integrated_weight", envir = ns)
  patched <- function(weights, ...) {
    dots <- list(...)
    if (!is.null(dots$residual_correlation) && is.null(colnames(weights))) {
      .cb_patch$uses <- .cb_patch$uses + 1L
      dots$residual_correlation <- NULL
      return(do.call(orig, c(list(weights), dots)))
    }
    orig(weights, ...)
  }
  attr(patched, "baselines_patch") <- TRUE
  utils::assignInNamespace("get_integrated_weight", patched, ns = "colocboost")
  invisible()
}

# Self-check of the patch (tests/baselines/test_rworker.py): unnamed placeholder -> NULL-correlation branch;
# named weights -> identical to the unpatched function.
h_cb_patch_check <- function(a, A) {
  suppressMessages(library(colocboost))
  orig <- get("get_integrated_weight", envir = asNamespace("colocboost"))
  if (isTRUE(attr(orig, "baselines_patch"))) stop("patch already applied in this worker; check needs a fresh one")
  patch_colocboost()
  pat <- get("get_integrated_weight", envir = asNamespace("colocboost"))
  rc <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
  set.seed(1)
  w2 <- matrix(runif(20), 10, 2, dimnames = list(NULL, c("Y1", "Y2")))
  w2 <- sweep(w2, 2, colSums(w2), "/")
  ph <- matrix(1, nrow = 10)
  list(named_equal = isTRUE(all.equal(pat(w2, residual_correlation = rc), orig(w2, residual_correlation = rc))),
       placeholder = as.numeric(pat(ph, residual_correlation = rc)),
       placeholder_nullrc = as.numeric(orig(ph, residual_correlation = NULL)),
       orig_fails = inherits(try(orig(ph, residual_correlation = rc), silent = TRUE), "try-error"))
}

h_colocboost <- function(a, A) {
  suppressMessages(library(colocboost))
  patch_colocboost()
  uses0 <- .cb_patch$uses
  p <- length(A$z1)
  snp <- paste0("s", seq_len(p))
  R <- A$R
  dimnames(R) <- list(snp, snp)
  ss <- list(data.frame(z = A$z1, n = a$n1, variant = snp), data.frame(z = A$z2, n = a$n2, variant = snp))
  rc <- matrix(c(1, a$rho, a$rho, 1), 2, 2)
  args <- list(sumstat = ss, LD = R, residual_correlation = rc, output_level = 2)
  if (!is.null(a$M)) args$M <- a$M
  res <- do.call(colocboost, args)
  out <- list(colocboost = as.character(packageVersion("colocboost")), n_cos = 0L, cos_npc = numeric(0),
              min_npc_outcome = numeric(0), cos_outcomes = list(), top_idx = integer(0),
              ucos_outcomes = integer(0), max_vcp = if (length(res$vcp)) max(res$vcp) else 0,
              cb_patch_uses = .cb_patch$uses - uses0)
  cs <- res$cos_summary
  if (!is.null(cs) && nrow(cs) > 0) {
    out$n_cos <- nrow(cs)
    out$cos_npc <- as.numeric(cs$cos_npc)
    out$min_npc_outcome <- as.numeric(cs$min_npc_outcome)
    out$cos_outcomes <- as.list(as.character(cs$colocalized_outcomes))
    out$top_idx <- as.integer(match(cs$top_variable, snp))
  }
  u <- res$ucos_details
  if (!is.null(u) && !is.null(u$ucos_outcomes) && length(u$ucos_outcomes$outcome_index) > 0) {
    out$ucos_outcomes <- as.integer(unlist(u$ucos_outcomes$outcome_index))
  }
  out
}

# ---------------------------------------------------------------- prop.coloc (Patel & Burgess)
h_propcoloc <- function(a, A) {
  suppressMessages(library(prop.coloc))
  p <- length(A$b1)
  R <- A$R
  dimnames(R) <- list(paste0("s", seq_len(p)), paste0("s", seq_len(p)))
  r <- prop.coloc(b1 = A$b1, se1 = A$se1, b2 = A$b2, se2 = A$se2, n = c(a$n1, a$n2), ld = R, tau = a$tau,
                  alpha = a$alpha, figs = FALSE)
  num <- function(x) if (is.null(x) || length(x) != 1 || is.na(x)) NULL else as.numeric(x)
  list(p_full = num(r$p_full), eta_full = num(r$eta_full), p_eta = num(r$p_eta), p_naive = num(r$p_naive),
       p_cond = num(r$p_cond), eta_cond = num(r$eta_cond),
       convergence = if (is.null(r$convergence) || is.na(r$convergence[1])) NULL else as.logical(r$convergence[1]),
       top = if (all(is.na(r$top))) NULL else as.integer(sub("^s", "", r$top)))
}

# ---------------------------------------------------------------- colocPropTest (Wallace)
h_coloctest <- function(a, A) {
  suppressMessages({ library(coloc); library(colocPropTest) })
  p <- length(A$beta1)
  snp <- paste0("s", seq_len(p))
  R <- A$R
  dimnames(R) <- list(snp, snp)
  S1 <- list(beta = A$beta1, varbeta = A$varbeta1, snp = snp, type = "quant", sdY = 1, N = a$n1, MAF = A$maf1)
  S2 <- list(beta = A$beta2, varbeta = A$varbeta2, snp = snp, type = "quant", sdY = 1, N = a$n2, MAF = A$maf2)
  r <- run_proptests(S1, S2, LD = R)
  if (is.null(r) || nrow(r) == 0) return(list(n_tests = 0L))
  list(n_tests = nrow(r), min_fdr = min(r$fdr), min_p = min(r$p))
}

handlers <- list(info = function(a, A) lib_info(), coloc_abf = h_coloc_abf, coloc_example = h_coloc_example, susie_coloc = h_susie_coloc,
                 mvsusie = h_mvsusie, colocboost = h_colocboost, propcoloc = h_propcoloc,
                 coloctest = h_coloctest, cb_patch_check = h_cb_patch_check,
                 echo = function(a, A) lapply(A, function(x) list(dim = if (is.null(dim(x))) length(x) else dim(x),
                                                                   values = as.numeric(x))),
                 sleep = function(a, A) { Sys.sleep(a$seconds); list(slept = a$seconds) },
                 fail = function(a, A) stop("deliberate failure"))

respond <- function(x) {
  cat("@@RESP ", toJSON(x, auto_unbox = TRUE, digits = I(17), na = "null", null = "null"), "\n", sep = "")
  flush(stdout())
}

con <- file("stdin", "r")
logcon <- stderr()
repeat {
  line <- readLines(con, n = 1)
  if (length(line) == 0) break
  if (!nzchar(line)) next
  req <- fromJSON(line, simplifyVector = TRUE)
  if (identical(req$fn, "quit")) break
  t0 <- proc.time()[["elapsed"]]
  warns <- character(0)
  msgs <- character(0)
  sink(logcon)  # package chatter goes to the log, not to the protocol channel
  out <- tryCatch(
    withCallingHandlers({
      A <- lapply(req$arrays, read_arr)
      h <- handlers[[req$fn]]
      if (is.null(h)) stop(paste("unknown handler", req$fn))
      list(ok = TRUE, result = h(req$args, A))
    },
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") }),
    error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  sink()
  out$id <- req$id
  out$warnings <- head(unique(warns), 10)
  out$n_messages <- length(msgs)
  out$messages <- head(unique(msgs), 5)
  out$elapsed <- proc.time()[["elapsed"]] - t0
  respond(out)
}

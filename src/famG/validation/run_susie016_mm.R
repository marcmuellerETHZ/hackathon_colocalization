# Reference runner for the G-mm identity test: susieR 0.16.6 exactly as src/baselines/r/worker.R (fit_susie with
# mismatch = TRUE: susie_rss(z = z, R = R, n = n, L = 10, R_finite = n_ref, R_mismatch = "eb_mix"), max_iter not
# passed) followed by coloc.bf_bf on the CS rows of lbf_variable (h_susie_coloc), plus the full fit (PIP, alpha,
# V, lambda, convergence) that the wrapper does not return.
# Usage (R_LIBS=$P/env/rlib_susie016): Rscript run_susie016_mm.R <locus_dir> <out_dir> <id> [<id> ...]
#   <locus_dir>/<id>/{meta.json, z1.bin, z2.bin, R.bin}: float64 little-endian, R column-major p x p.
# Optional rounding-level perturbation (sensitivity check, DECISIONS P4-3): env MM_PERT = relative size (e.g. 1e-14),
# MM_SEED = seed; then z_t <- z_t * (1 + MM_PERT * u), u ~ U(-1, 1), drawn for z1 then z2 after set.seed(MM_SEED).
suppressMessages({ library(jsonlite); library(susieR); library(coloc) })
options(warn = 1)
args <- commandArgs(trailingOnly = TRUE)
ldir <- args[1]; odir <- args[2]; ids <- args[-(1:2)]
dir.create(odir, showWarnings = FALSE, recursive = TRUE)
rd <- function(f, n) { con <- file(f, "rb"); on.exit(close(con)); readBin(con, "double", n = n, size = 8, endian = "little") }

fit_one <- function(z, R, n, n_ref) {
  warns <- character(0)
  t0 <- proc.time()[["elapsed"]]
  f <- tryCatch(withCallingHandlers(
    susie_rss(z = z, R = R, n = n, L = 10, R_finite = n_ref, R_mismatch = "eb_mix"),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage")),
    error = function(e) e)
  el <- proc.time()[["elapsed"]] - t0
  if (inherits(f, "error")) return(list(fit = NULL, out = list(error = conditionMessage(f), elapsed = el)))
  ncs <- if (is.null(f$sets$cs)) 0L else length(f$sets$cs)
  d <- f$R_finite_diagnostics
  out <- list(error = NULL, elapsed = el, converged = isTRUE(f$converged), niter = f$niter,
              pip = as.numeric(f$pip), alpha = unname(f$alpha), V = as.numeric(f$V),
              lbf = as.numeric(f$lbf),
              n_cs = ncs, cs_index = if (ncs) as.integer(f$sets$cs_index) else integer(0),
              cs = if (ncs) unname(lapply(f$sets$cs, as.integer)) else list(),
              purity_min = if (ncs) as.numeric(f$sets$purity[, 1]) else numeric(0),
              lambda_bias = d$lambda_bias, ld_coherence = d$R_mismatch_init$ld_coherence,
              lambda_init = d$R_mismatch_init$lambda_bias,
              convergence_reason = if (is.null(f$convergence_reason)) NA else f$convergence_reason,
              warnings = head(unique(warns), 5))
  list(fit = f, out = out)
}

for (id in ids) {
  m <- fromJSON(file.path(ldir, id, "meta.json"))
  p <- m$p
  z1 <- rd(file.path(ldir, id, "z1.bin"), p); z2 <- rd(file.path(ldir, id, "z2.bin"), p)
  R <- matrix(rd(file.path(ldir, id, "R.bin"), p * p), p, p)
  pert <- as.numeric(Sys.getenv("MM_PERT", "0"))
  if (pert > 0) {
    set.seed(as.integer(Sys.getenv("MM_SEED", "1")))
    z1 <- z1 * (1 + pert * runif(p, -1, 1)); z2 <- z2 * (1 + pert * runif(p, -1, 1))
  }
  a <- fit_one(z1, R, m$n1, m$n_ref); b <- fit_one(z2, R, m$n2, m$n_ref)
  res <- list(id = id, susieR = as.character(packageVersion("susieR")), t1 = a$out, t2 = b$out, pairs = NULL)
  if (!is.null(a$fit) && !is.null(b$fit) && a$out$n_cs > 0 && b$out$n_cs > 0) {
    snp <- paste0("s", seq_len(p))
    bf1 <- a$fit$lbf_variable[a$out$cs_index, , drop = FALSE]
    bf2 <- b$fit$lbf_variable[b$out$cs_index, , drop = FALSE]
    colnames(bf1) <- snp; colnames(bf2) <- snp
    cb <- suppressMessages(coloc.bf_bf(bf1, bf2, p1 = 1e-4, p2 = 1e-4, p12 = 5e-6))
    sm <- as.data.frame(cb$summary)
    res$pairs <- list(idx1 = as.integer(sm$idx1), idx2 = as.integer(sm$idx2),
                      PP = unname(as.matrix(sm[, c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")])))
  }
  writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), na = "null", null = "null"),
             file.path(odir, paste0(id, ".json")))
}

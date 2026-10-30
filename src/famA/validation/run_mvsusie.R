# mvsusieR 0.3.0 reference for the Family A identity test (eta = 0, rho = 0, fixed mixture prior).
# Usage (R_LIBS=$P/env/rlib_susie016:$P/env/rlib_mvsusie):
#   Rscript run_mvsusie.R WORKDIR OUTPREFIX ROWS(comma-separated 1-based) U1 U2 PI0 PI1 PI2 PI3 TOL
# The prior matrices are diag(U1,0), diag(0,U2), diag(U1,U2) on the z scale; they are passed as U * N/(N-1)
# because mvsusie_ss rescales them by 1/N on the standardized-b scale (b_z = sqrt(N-1) b).
suppressPackageStartupMessages({library(mvsusieR); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE)
wd <- a[1]; outp <- a[2]; rows <- as.integer(strsplit(a[3], ",")[[1]])
u1 <- as.numeric(a[4]); u2 <- as.numeric(a[5]); pis <- as.numeric(a[6:9]); tol <- as.numeric(a[10])
idx <- read.delim(file.path(wd, "index.tsv"), stringsAsFactors = FALSE)
conR <- file(file.path(wd, "R.bin"), "rb"); conz <- file(file.path(wd, "z.bin"), "rb")
conb <- file(paste0(outp, ".bin"), "wb")
res <- list()
for (i in rows) {
  r <- idx[i, ]; p <- r$p; N <- r$n
  seek(conR, where = r$R_offset * 8); R <- matrix(readBin(conR, "double", p * p), p, p)
  seek(conz, where = r$z_offset * 8); z1 <- readBin(conz, "double", p); z2 <- readBin(conz, "double", p)
  sc <- N / (N - 1)
  mats <- list(matrix(c(u1, 0, 0, 0), 2, 2) * sc, matrix(c(0, 0, 0, u2), 2, 2) * sc, matrix(c(u1, 0, 0, u2), 2, 2) * sc)
  prior <- create_mixture_prior(mixture_prior = list(matrices = mats, weights = pis[2:4] / sum(pis[2:4])), null_weight = pis[1])
  t0 <- proc.time()[["elapsed"]]
  warns <- character(0)
  fit <- tryCatch(withCallingHandlers(
    mvsusie_rss(Z = cbind(z1, z2), R = R, N = N, varY = diag(2), prior_variance = prior,
                residual_variance = diag(2), estimate_residual_variance = FALSE,
                estimate_prior_variance = FALSE, estimate_prior_mixture_weights = FALSE,
                L = 10, max_iter = 100, tol = tol, verbose = FALSE),
    warning = function(w) {warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning")}),
    error = function(e) e)
  t1 <- proc.time()[["elapsed"]]
  if (inherits(fit, "error")) {
    res[[r$locus_id]] <- list(error = conditionMessage(fit), p = p)
    writeBin(rep(NA_real_, p + 10 * p + 10 * p * 4), conb)
    next
  }
  L <- nrow(fit$alpha)
  writeBin(as.double(fit$pip), conb)
  writeBin(as.double(t(fit$alpha)), conb)
  pmw <- fit$posterior_mixture_weights   # L x J x (K+1) incl. null column
  writeBin(as.double(aperm(pmw, c(3, 2, 1))), conb)   # row-major (L, J, K+1)
  res[[r$locus_id]] <- list(p = p, L = L, K1 = dim(pmw)[3], niter = fit$niter, converged = fit$converged,
                            elbo = fit$elbo, lbf = fit$lbf, time = t1 - t0, warnings = warns)
}
close(conR); close(conz); close(conb)
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), paste0(outp, ".json"))

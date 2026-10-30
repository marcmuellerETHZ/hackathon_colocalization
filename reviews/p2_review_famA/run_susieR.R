# Reviewer reference (p2.review_famA): susieR 0.14.2 susie_rss(z, R, n = n, L = 10) with all other defaults.
# Usage: Rscript run_susieR.R WORKDIR   (reads index.tsv, R.bin, z.bin; writes r_out.json, r_out.bin)
suppressPackageStartupMessages({library(susieR); library(jsonlite)})
stopifnot(as.character(packageVersion("susieR")) == "0.14.2")
wd <- commandArgs(trailingOnly = TRUE)[1]
idx <- read.delim(file.path(wd, "index.tsv"), stringsAsFactors = FALSE)
conR <- file(file.path(wd, "R.bin"), "rb"); conz <- file(file.path(wd, "z.bin"), "rb")
conb <- file(file.path(wd, "r_out.bin"), "wb")
res <- list()
set.seed(1)
for (i in seq_len(nrow(idx))) {
  r <- idx[i, ]; p <- r$p
  seek(conR, where = r$R_offset * 8); R <- matrix(readBin(conR, "double", p * p), p, p)
  seek(conz, where = r$z_offset * 8); z <- readBin(conz, "double", p)
  fit <- tryCatch(suppressWarnings(susie_rss(z, R, n = r$n, L = 10)), error = function(e) e)
  if (inherits(fit, "error")) {
    res[[r$fit_id]] <- list(error = conditionMessage(fit), L = 10L)
    writeBin(rep(NA_real_, p + 10 * p), conb); next
  }
  writeBin(as.double(fit$pip), conb)
  writeBin(as.double(t(fit$alpha)), conb)
  cs <- fit$sets$cs
  res[[r$fit_id]] <- list(L = nrow(fit$alpha), V = fit$V, elbo = fit$elbo, niter = fit$niter, converged = fit$converged,
                          cs = if (is.null(cs)) list() else unname(lapply(cs, function(x) x - 1L)))
  cat(r$fit_id, fit$niter, fit$converged, length(cs), "\n")
}
close(conR); close(conz); close(conb)
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), file.path(wd, "r_out.json"))

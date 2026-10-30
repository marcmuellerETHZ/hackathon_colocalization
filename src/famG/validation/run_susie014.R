# Run susieR 0.14.2 susie_rss on loci written by famG/validation/make_loci.py (reference for the G identity test).
# Usage: Rscript run_susie014.R WORKDIR OUTPREFIX FROM TO TRAIT N_PURITY USE_N
#   FROM/TO: 1-based row range of index.tsv; TRAIT: 1|2; N_PURITY: susie n_purity (default 100);
#   USE_N: 1 = susie_rss(z, R, n = n, L = 10) (standard pipeline), 0 = susie_rss(z, R, L = 10) (no n).
# Outputs: OUTPREFIX.json (per-locus scalars, sets, timing) and OUTPREFIX.bin (float64: pip, alpha, lbf_variable per locus).
suppressPackageStartupMessages({library(susieR); library(jsonlite)})
stopifnot(as.character(packageVersion("susieR")) == "0.14.2")
a <- commandArgs(trailingOnly = TRUE)
wd <- a[1]; outp <- a[2]; from <- as.integer(a[3]); to <- as.integer(a[4]); trait <- as.integer(a[5])
npur <- as.integer(a[6]); use_n <- as.integer(a[7]) == 1
idx <- read.delim(file.path(wd, "index.tsv"), stringsAsFactors = FALSE)
to <- min(to, nrow(idx))
conR <- file(file.path(wd, "R.bin"), "rb"); conz <- file(file.path(wd, "z.bin"), "rb")
conb <- file(paste0(outp, ".bin"), "wb")
res <- list()
set.seed(1)
for (i in from:to) {
  r <- idx[i, ]; p <- r$p
  seek(conR, where = r$R_offset * 8); R <- matrix(readBin(conR, "double", p * p), p, p)
  seek(conz, where = (r$z_offset + (trait - 1) * p) * 8); z <- readBin(conz, "double", p)
  n <- if (trait == 1) r$n else r$n2
  t0 <- proc.time()[["elapsed"]]
  warns <- character(0)
  fit <- tryCatch(withCallingHandlers({
    if (use_n) susie_rss(z, R, n = n, L = 10, n_purity = npur) else susie_rss(z, R, L = 10, n_purity = npur)
  }, warning = function(w) {warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning")}),
  error = function(e) e)
  t1 <- proc.time()[["elapsed"]]
  if (inherits(fit, "error")) {
    res[[r$locus_id]] <- list(error = conditionMessage(fit), time = t1 - t0, p = p)
    writeBin(rep(NA_real_, p + 2 * 10 * p), conb)
    next
  }
  L <- nrow(fit$alpha)
  writeBin(as.double(fit$pip), conb)
  writeBin(as.double(t(fit$alpha)), conb)          # row-major (L, p)
  writeBin(as.double(t(fit$lbf_variable)), conb)
  sets <- fit$sets
  res[[r$locus_id]] <- list(
    p = p, L = L, V = fit$V, lbf = fit$lbf, KL = fit$KL, elbo = fit$elbo, niter = fit$niter,
    converged = fit$converged, time = t1 - t0, warnings = warns,
    cs = if (is.null(sets$cs)) list() else unname(lapply(sets$cs, function(x) x - 1L)),
    cs_index = if (is.null(sets$cs_index)) integer(0) else sets$cs_index - 1L,
    purity = if (is.null(sets$purity)) list() else unname(as.list(as.data.frame(t(as.matrix(sets$purity)))))
  )
}
close(conR); close(conz); close(conb)
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), paste0(outp, ".json"))

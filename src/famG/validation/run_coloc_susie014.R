# Reference coloc-SuSiE pipeline (susieR 0.14.2 susie_rss + coloc 5.2.3 coloc.susie, coloc.abf fallback)
# on loci written by famG/validation/make_loci.py. Usage: Rscript run_coloc_susie014.R WORKDIR OUTJSON FROM TO
suppressPackageStartupMessages({library(susieR); library(coloc); library(jsonlite)})
stopifnot(as.character(packageVersion("susieR")) == "0.14.2", as.character(packageVersion("coloc")) == "5.2.3")
a <- commandArgs(trailingOnly = TRUE)
wd <- a[1]; outj <- a[2]; from <- as.integer(a[3]); to <- as.integer(a[4])
idx <- read.delim(file.path(wd, "index.tsv"), stringsAsFactors = FALSE)
to <- min(to, nrow(idx))
conR <- file(file.path(wd, "R.bin"), "rb"); conz <- file(file.path(wd, "z.bin"), "rb"); conm <- file(file.path(wd, "maf.bin"), "rb")
res <- list()
for (i in from:to) {
  r <- idx[i, ]; p <- r$p
  seek(conR, where = r$R_offset * 8); R <- matrix(readBin(conR, "double", p * p), p, p)
  seek(conz, where = r$z_offset * 8); z1 <- readBin(conz, "double", p); z2 <- readBin(conz, "double", p)
  seek(conm, where = (r$z_offset / 2) * 8); maf <- readBin(conm, "double", p)
  nm <- paste0("v", seq_len(p)); dimnames(R) <- list(nm, nm); names(z1) <- names(z2) <- nm
  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch(suppressWarnings({
    s1 <- susie_rss(z1, R, n = r$n, L = 10); s2 <- susie_rss(z2, R, n = r$n2, L = 10)
    cs <- suppressMessages(coloc.susie(s1, s2, p12 = 5e-6))
    if (is.null(cs$summary) || all(is.na(cs$summary$nsnps)) || !("PP.H4.abf" %in% names(cs$summary))) {
      se1 <- 1 / sqrt(2 * r$n * maf * (1 - maf)); se2 <- 1 / sqrt(2 * r$n2 * maf * (1 - maf))
      ab <- suppressMessages(coloc.abf(list(beta = z1 * se1, varbeta = se1^2, N = r$n, MAF = maf, type = "quant", sdY = 1, snp = nm),
                                        list(beta = z2 * se2, varbeta = se2^2, N = r$n2, MAF = maf, type = "quant", sdY = 1, snp = nm),
                                        p12 = 5e-6))
      list(fallback = TRUE, pp = matrix(as.numeric(ab$summary[2:6]), nrow = 1))
    } else {
      sm <- cs$summary
      list(fallback = FALSE, pp = as.matrix(sm[, c("PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")]),
           idx1 = sm$idx1 - 1L, idx2 = sm$idx2 - 1L)
    }
  }), error = function(e) list(error = conditionMessage(e)))
  out$time <- proc.time()[["elapsed"]] - t0
  res[[r$locus_id]] <- out
}
close(conR); close(conz); close(conm)
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), outj)

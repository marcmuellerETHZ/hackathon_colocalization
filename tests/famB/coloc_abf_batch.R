# Reference: coloc.abf (coloc 5.2.3) on many loci. Usage: Rscript coloc_abf_batch.R in.tsv out.tsv p1 p2 p12
suppressPackageStartupMessages(library(coloc))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(as.character(packageVersion("coloc")) == "5.2.3")
d <- read.delim(args[1], stringsAsFactors = FALSE)
p1 <- as.numeric(args[3]); p2 <- as.numeric(args[4]); p12 <- as.numeric(args[5])
out <- list()
for (lid in unique(d$locus)) {
  x <- d[d$locus == lid, ]
  d1 <- list(beta = x$beta1, varbeta = x$varbeta1, N = x$N1[1], MAF = x$MAF, type = "quant", sdY = 1, snp = x$snp)
  d2 <- list(beta = x$beta2, varbeta = x$varbeta2, N = x$N2[1], MAF = x$MAF, type = "quant", sdY = 1, snp = x$snp)
  r <- suppressMessages(coloc.abf(d1, d2, p1 = p1, p2 = p2, p12 = p12))
  s <- r$summary
  out[[length(out) + 1]] <- data.frame(locus = lid, nsnps = s[["nsnps"]], H0 = s[["PP.H0.abf"]], H1 = s[["PP.H1.abf"]],
                                       H2 = s[["PP.H2.abf"]], H3 = s[["PP.H3.abf"]], H4 = s[["PP.H4.abf"]])
}
write.table(do.call(rbind, out), args[2], sep = "\t", quote = FALSE, row.names = FALSE)

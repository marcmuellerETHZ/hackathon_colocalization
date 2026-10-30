# p5.verify_infra check (d): R coloc 5.2.3 coloc.abf on loci exported by coloc_r_check.py.
# args: <dir with locus_*.tsv> ; writes <dir>/r_pp.tsv
suppressMessages(library(coloc))
args <- commandArgs(TRUE)
d <- args[1]
out <- list()
for (f in sort(list.files(d, pattern = "^locus_.*\\.tsv$", full.names = TRUE))) {
  x <- read.delim(f)
  for (p12 in c(5e-6, 1e-5)) {
    r <- suppressMessages(coloc.abf(
      list(beta = x$beta1, varbeta = x$se1^2, N = x$n1[1], MAF = x$maf1, type = "quant", sdY = 1, snp = x$snp),
      list(beta = x$beta2, varbeta = x$se2^2, N = x$n2[1], MAF = x$maf2, type = "quant", sdY = 1, snp = x$snp),
      p1 = 1e-4, p2 = 1e-4, p12 = p12))
    s <- r$summary
    out[[length(out) + 1]] <- data.frame(file = basename(f), p12 = p12, nsnps = s[["nsnps"]],
      PP0 = s[["PP.H0.abf"]], PP1 = s[["PP.H1.abf"]], PP2 = s[["PP.H2.abf"]], PP3 = s[["PP.H3.abf"]],
      PP4 = s[["PP.H4.abf"]], coloc_version = as.character(packageVersion("coloc")))
  }
}
write.table(do.call(rbind, out), file.path(d, "r_pp.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# Sensitivity diagnostics for G-mm identity mismatches (non-converged fits): susieR 0.16.6 fit of one trait,
# same call as worker.R fit_susie (R_finite = n_ref, R_mismatch = "eb_mix"), stopped at max_iter = k for each k,
# with z unperturbed (pert = 0) or perturbed at rounding level: z * (1 + pert * u), u ~ U(-1, 1) (seed 1).
# Usage: Rscript run_traj_mm.R <locus_dir> <out_json> <id> <trait 1|2> <pert> <k1,k2,...>
suppressMessages({ library(jsonlite); library(susieR) })
a <- commandArgs(trailingOnly = TRUE)
ldir <- a[1]; out <- a[2]; id <- a[3]; tr <- as.integer(a[4]); pert <- as.numeric(a[5])
ks <- as.integer(strsplit(a[6], ",")[[1]])
rd <- function(f, n) { con <- file(f, "rb"); on.exit(close(con)); readBin(con, "double", n = n, size = 8, endian = "little") }
m <- fromJSON(file.path(ldir, id, "meta.json"))
p <- m$p
z <- rd(file.path(ldir, id, sprintf("z%d.bin", tr)), p)
R <- matrix(rd(file.path(ldir, id, "R.bin"), p * p), p, p)
n <- if (tr == 1) m$n1 else m$n2
if (pert > 0) { set.seed(1); z <- z * (1 + pert * runif(p, -1, 1)) }
res <- list()
for (k in ks) {
  f <- suppressWarnings(suppressMessages(
    susie_rss(z = z, R = R, n = n, L = 10, R_finite = m$n_ref, R_mismatch = "eb_mix", max_iter = k)))
  res[[as.character(k)]] <- list(k = k, niter = f$niter, converged = isTRUE(f$converged), pip = as.numeric(f$pip),
                                 lambda = f$R_finite_diagnostics$lambda_bias, V = as.numeric(f$V))
}
writeLines(toJSON(list(id = id, trait = tr, pert = pert, fits = unname(res)), auto_unbox = TRUE, digits = I(17)), out)

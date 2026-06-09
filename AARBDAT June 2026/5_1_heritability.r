# ============================================================
# 5_1_heritability.r  —  Heritability (H2) & Genetic Advance
# SALF AARBDAT 2026 — Day 5, Chapter 5
# 25 Maize Genotypes x 3 Replications — RCBD
# ============================================================

cat("==========================================================\n")
cat("  CHAPTER 5: HERITABILITY (H2) ANALYSIS\n")
cat("  25 Maize Genotypes x 3 Replications — RCBD\n")
cat("==========================================================\n\n")

# ---- Master function ----
heritability_full <- function(MS_T, MS_E, r, Xbar, tname, k=2.063){
  s2G  <- max((MS_T - MS_E)/r, 0)
  s2P  <- s2G + MS_E
  GCV  <- sqrt(s2G)/Xbar*100
  PCV  <- sqrt(s2P)/Xbar*100
  H2   <- s2G/s2P
  GA   <- k * sqrt(s2P) * H2
  GAM  <- GA/Xbar*100
  h2cat <- if(H2*100 < 30) "Low" else if(H2*100 < 60) "Moderate" else "High"
  gcvcat<- if(GCV < 10) "Low" else if(GCV < 20) "Moderate" else "High"
  list(Trait=tname, Mean=round(Xbar,2),
       MS_T=round(MS_T,3), MS_E=round(MS_E,3),
       s2G=round(s2G,3), s2P=round(s2P,3),
       GCV=round(GCV,2), PCV=round(PCV,2),
       H2=round(H2*100,2), H2cat=h2cat, GCVcat=gcvcat,
       GA=round(GA,3), GAM=round(GAM,2))
}

# ---- 1. Generate realistic maize dataset ----
set.seed(303)
n_gen <- 25; n_rep <- 3
df_maize <- expand.grid(
  Genotype = paste0("G",sprintf("%02d",1:n_gen)),
  Rep      = paste0("R",1:n_rep),
  stringsAsFactors = TRUE)

# Simulated trait values with varying heritability
G_ear  <- rnorm(n_gen, 16.5, 2.3)
G_rows <- rnorm(n_gen, 14.2, 1.5)
G_gy   <- rnorm(n_gen, 38.4, 8.6)
G_silk <- rnorm(n_gen, 62.0, 4.2)
G_pht  <- rnorm(n_gen,185.0,22.5)

g_idx <- as.integer(df_maize$Genotype)
df_maize$ear_length <- G_ear[g_idx]  + rnorm(nrow(df_maize),0,1.52)
df_maize$rows_ear   <- G_rows[g_idx] + rnorm(nrow(df_maize),0,1.45)
df_maize$grain_yield<- G_gy[g_idx]   + rnorm(nrow(df_maize),0,9.44)
df_maize$days_silk  <- G_silk[g_idx] + rnorm(nrow(df_maize),0,2.05)
df_maize$plant_ht   <- G_pht[g_idx]  + rnorm(nrow(df_maize),0,17.9)
print(df_maize)

traits_m <- c("ear_length","rows_ear","grain_yield","days_silk","plant_ht")
labels_m <- c("Ear Length (cm)","Rows per Ear","Grain Yield (q/ha)",
               "Days to Silking","Plant Height (cm)")

# ---- 2. ANOVA per trait ----
cat("--- 1. ANOVA TABLES FOR EACH TRAIT ---\n\n")
res_list <- list()
for(i in seq_along(traits_m)){
  frm <- as.formula(paste(traits_m[i],"~ Rep + Genotype"))
  mod <- aov(frm, data=df_maize)
  at  <- summary(mod)[[1]]
  MS_T <- at["Genotype","Mean Sq"]
  MS_E <- at["Residuals","Mean Sq"]
  Xb   <- mean(df_maize[[traits_m[i]]])
  cat(sprintf("Trait: %s\n", labels_m[i]))
  print(summary(mod))
  cat(sprintf("  Grand Mean = %.3f\n\n", Xb))
  res_list[[i]] <- heritability_full(MS_T, MS_E, n_rep, Xb, labels_m[i])
}

# ---- 3. Summary table ----
cat("--- 2. COMPLETE HERITABILITY & GENETIC ADVANCE TABLE ---\n")
cat(sprintf("  %-22s %7s %8s %7s %7s %7s %7s %8s %7s\n",
  "Trait","Mean","s2G","s2P","GCV%","PCV%","H2%","GA","GAM%"))
cat("  ",strrep("-",85),"\n",sep="")
for(r in res_list){
  cat(sprintf("  %-22s %7.2f %8.3f %7.3f %7.2f %7.2f %7.2f %8.3f %7.2f\n",
    substr(r$Trait,1,22), r$Mean, r$s2G, r$s2P, r$GCV, r$PCV, r$H2, r$GA, r$GAM))
}

cat("\n--- 3. DETAILED CALCULATION — Ear Length ---\n")
r1 <- res_list[[1]]
cat(sprintf("  Grand Mean (X-bar)    = %.3f cm\n", r1$Mean))
cat(sprintf("  MS(Genotype)          = %.3f\n", r1$MS_T))
cat(sprintf("  MS(Error)             = %.3f\n", r1$MS_E))
cat(sprintf("  r                     = %d\n", n_rep))
cat(sprintf("  Step 1: s2G = (%.3f - %.3f)/%d = %.3f\n", r1$MS_T,r1$MS_E,n_rep,r1$s2G))
cat(sprintf("  Step 2: s2P = %.3f + %.3f    = %.3f\n", r1$s2G,r1$MS_E,r1$s2P))
cat(sprintf("  Step 3: H2  = %.3f/%.3f      = %.4f = %.2f%%\n",
  r1$s2G, r1$s2P, r1$s2G/r1$s2P, r1$H2))
cat(sprintf("  Step 4: s_P = sqrt(%.3f)     = %.4f\n", r1$s2P, sqrt(r1$s2P)))
cat(sprintf("  Step 5: GA  = 2.063 x %.4f x %.4f = %.3f cm\n",
  sqrt(r1$s2P), r1$H2/100, r1$GA))
cat(sprintf("  Step 6: GAM = (%.3f/%.3f) x 100  = %.2f%%\n", r1$GA, r1$Mean, r1$GAM))

cat("\n--- 4. HERITABILITY CATEGORIES (Johnson et al. 1955) ---\n")
cat("  H2 < 30%    : Low      — Selection may not be effective\n")
cat("  H2 30-60%   : Moderate — Moderate selection response\n")
cat("  H2 > 60%    : High     — Selection highly effective\n\n")

cat("--- 5. JOINT INTERPRETATION (H2 + GAM) ---\n")
cat(sprintf("  %-22s %7s %10s %8s %10s %20s\n",
  "Trait","H2(%)","H2 Cat","GAM(%)","GCV Cat","Breeding Strategy"))
cat("  ",strrep("-",80),"\n",sep="")
strategy <- function(h2,gam){
  if(h2>60 && gam>20) "Direct selection (additive)" else
  if(h2>60 && gam<=20) "Hybridisation (non-additive)" else
  if(h2<=60 && gam>20) "Multi-env selection" else
  "Improve environment"
}
for(r in res_list){
  cat(sprintf("  %-22s %7.2f %10s %8.2f %10s %20s\n",
    substr(r$Trait,1,22), r$H2, r$H2cat, r$GAM, r$GCVcat,
    strategy(r$H2, r$GAM)))
}

# ---- 6. Plot ----
png("rplots/5_1_heritability_plot.png", width=850, height=540, res=110)
par(mar=c(8,5,4,2))
H2_vals  <- sapply(res_list, function(x) x$H2)
GAM_vals <- sapply(res_list, function(x) x$GAM)
GCV_vals <- sapply(res_list, function(x) x$GCV)
tn <- c("EarLen","RowsEar","GrainYld","DaysSilk","PlantHt")
x_loc <- barplot(rbind(H2_vals,GAM_vals,GCV_vals), beside=TRUE,
  col=c("#2d5a1b","#c4732a","#1a3a6b"),
  names.arg=rep("",5), ylim=c(0,115),
  main="Heritability (H2%), Genetic Advance as % Mean (GAM%) & GCV%\nMaize Germplasm",
  ylab="Percentage (%)", las=1, cex.main=0.98)
mtext(tn, side=1, at=colMeans(x_loc), line=1.5, cex=0.82)
abline(h=60, lty=2, col="#2d5a1b", lwd=1.5)
abline(h=20, lty=2, col="red",     lwd=1.5)
text(0.5, 61, "H2=60% threshold", cex=0.7, col="#1a3a6b", adj=0)
text(0.5, 21, "20% (High GCV/GAM)", cex=0.7, col="red",   adj=0)
legend("topright", legend=c("H2 (%)","GAM (%)","GCV (%)"),
  fill=c("#2d5a1b","#c4732a","#1a3a6b"), bty="n", cex=0.88)
box(); dev.off()
cat("\nPlot saved.\n")

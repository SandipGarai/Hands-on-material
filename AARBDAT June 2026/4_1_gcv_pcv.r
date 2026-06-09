# ============================================================
# 4_1_gcv_pcv.r  —  GCV & PCV Analysis
# SALF AARBDAT 2026 — Day 5, Chapter 4
# Chickpea germplasm — 20 genotypes, 3 reps, RCBD
# ============================================================

cat("==========================================================\n")
cat("  CHAPTER 4: GCV & PCV DATA ANALYSIS\n")
cat("  20 Chickpea Genotypes x 3 Replications — RCBD\n")
cat("==========================================================\n\n")

# ---- Master function ----
gcv_pcv_analysis <- function(MS_treat, MS_error, r, grand_mean, trait_name){
  sigma2_G <- max((MS_treat - MS_error)/r, 0)
  sigma2_P <- sigma2_G + MS_error
  GCV <- sqrt(sigma2_G) / grand_mean * 100
  PCV <- sqrt(sigma2_P) / grand_mean * 100
  classify <- function(v) if(v < 10) "Low" else if(v < 20) "Moderate" else "High"
  list(Trait=trait_name, Mean=round(grand_mean,2),
       MS_T=round(MS_treat,3), MS_E=round(MS_error,3),
       s2G=round(sigma2_G,3), s2P=round(sigma2_P,3),
       GCV=round(GCV,2), PCV=round(PCV,2),
       GCV_class=classify(GCV), PCV_class=classify(PCV))
}

# ---- 1. Raw data simulation ----
set.seed(200)
n_gen <- 20; n_rep <- 3
# True genotypic effects (SD≈22% of mean for high-variability traits)
true_means_yl   <- seq(900,1600,length.out=n_gen) + rnorm(n_gen,0,50)  # Seed yield
true_means_dttf <- seq(50,66,length.out=n_gen)                          # Days to flower
true_means_ph   <- seq(32,60,length.out=n_gen)                          # Plant height
true_means_sw   <- seq(12,28,length.out=n_gen)                          # 100-seed wt
true_means_pp   <- seq(18,42,length.out=n_gen)                          # Pods/plant

df_chickpea <- expand.grid(Genotype=paste0("G",sprintf("%02d",1:n_gen)),
                            Rep=paste0("R",1:n_rep), stringsAsFactors=TRUE)
g_idx <- as.integer(df_chickpea$Genotype)
df_chickpea$seed_yield  <- true_means_yl[g_idx]   + rnorm(nrow(df_chickpea),0,200)
df_chickpea$days_flower <- true_means_dttf[g_idx]  + rnorm(nrow(df_chickpea),0,2.5)
df_chickpea$plant_ht    <- true_means_ph[g_idx]    + rnorm(nrow(df_chickpea),0,4.5)
df_chickpea$seed_wt     <- true_means_sw[g_idx]    + rnorm(nrow(df_chickpea),0,1.8)
df_chickpea$pods_plant  <- true_means_pp[g_idx]    + rnorm(nrow(df_chickpea),0,3.2)
df_chickpea <- df_chickpea[,-(3:7)]  # keep only factors for now

# re-add cleaned values
df_chickpea$seed_yield  <- pmax(500, true_means_yl[g_idx]  + rnorm(nrow(df_chickpea),0,200))
df_chickpea$days_flower <- pmax(44,  true_means_dttf[g_idx] + rnorm(nrow(df_chickpea),0,2.5))
df_chickpea$plant_ht    <- pmax(20,  true_means_ph[g_idx]   + rnorm(nrow(df_chickpea),0,4.5))
df_chickpea$seed_wt     <- pmax(8,   true_means_sw[g_idx]   + rnorm(nrow(df_chickpea),0,1.8))
df_chickpea$pods_plant  <- pmax(10,  true_means_pp[g_idx]   + rnorm(nrow(df_chickpea),0,3.2))
traits <- c("seed_yield","days_flower","plant_ht","seed_wt","pods_plant")
labels <- c("Seed Yield (kg/ha)","Days to Flower","Plant Height (cm)",
            "100-Seed Weight (g)","Pods per Plant")
print(df_chickpea)
# ---- 2. ANOVA and GCV/PCV for each trait ----
cat("--- 1. ANOVA TABLES FOR EACH TRAIT ---\n\n")
results <- list()
for(i in seq_along(traits)){
  tr  <- traits[i]
  frm <- as.formula(paste(tr,"~ Rep + Genotype"))
  mod <- aov(frm, data=df_chickpea)
  aov_t <- summary(mod)[[1]]
  MS_T  <- aov_t["Genotype","Mean Sq"]
  MS_E  <- aov_t["Residuals","Mean Sq"]
  Xbar  <- mean(df_chickpea[[tr]])
  cat(sprintf("Trait: %s\n", labels[i]))
  print(summary(mod))
  cat(sprintf("  Grand Mean = %.2f\n\n", Xbar))
  results[[i]] <- gcv_pcv_analysis(MS_T, MS_E, n_rep, Xbar, labels[i])
}

# ---- 3. Summary table ----
cat("\n\n--- 2. GCV & PCV SUMMARY TABLE ---\n")
cat(sprintf("  %-22s %8s %8s %7s %7s %7s %7s %10s %10s\n",
  "Trait","Mean","s2G","s2P","GCV%","PCV%","GCV Cat","PCV Cat","GCV>PCV%"))
cat("  ", strrep("-",98),"\n",sep="")
for(res in results){
  gc <- res$GCV_class
  gc_col <- gc  # just text in terminal
  cat(sprintf("  %-22s %8.2f %8.3f %7.3f %7.2f %7.2f %10s %10s\n",
    substr(res$Trait,1,22), res$Mean, res$s2G, res$s2P,
    res$GCV, res$PCV, res$GCV_class, res$PCV_class))
}

cat("\n--- 3. DETAILED COMPUTATION (Seed Yield) ---\n")
r1 <- results[[1]]
cat(sprintf("  Grand Mean (X-bar) = %.2f kg/ha\n", r1$Mean))
cat(sprintf("  MS(Genotype)       = %.3f\n", r1$MS_T))
cat(sprintf("  MS(Error)          = %.3f\n", r1$MS_E))
cat(sprintf("  r (replications)   = %d\n", n_rep))
cat(sprintf("  s2G = (%.3f - %.3f)/%d = %.3f\n", r1$MS_T, r1$MS_E, n_rep, r1$s2G))
cat(sprintf("  s2P = %.3f + %.3f    = %.3f\n", r1$s2G, r1$MS_E, r1$s2P))
cat(sprintf("  GCV = sqrt(%.3f)/%.2f x 100 = %.2f%%\n", r1$s2G, r1$Mean, r1$GCV))
cat(sprintf("  PCV = sqrt(%.3f)/%.2f x 100 = %.2f%%\n", r1$s2P, r1$Mean, r1$PCV))
cat(sprintf("  GCV Category: %s\n", r1$GCV_class))
cat(sprintf("  PCV Category: %s\n", r1$PCV_class))

cat("\n--- 4. INTERPRETATION GUIDE ---\n")
cat("  GCV/PCV < 10%  : LOW      — Limited genetic variability\n")
cat("  GCV/PCV 10-20% : MODERATE — Moderate scope for selection\n")
cat("  GCV/PCV > 20%  : HIGH     — Excellent scope for selection\n")
cat("  GCV ≈ PCV      : Low environmental influence → High heritability expected\n")
cat("  PCV >> GCV     : High environmental influence → Selection less reliable\n")

# ---- Plot ----
png("/home/claude/rplots/4_1_gcv_pcv_barplot.png", width=800, height=520, res=110)
par(mar=c(8,5,4,2))
gcv_vals <- sapply(results, function(x) x$GCV)
pcv_vals <- sapply(results, function(x) x$PCV)
trt_names <- c("SeedYield","DaysFlower","PlantHt","SeedWt","Pods/Plant")
x_pos <- barplot(rbind(gcv_vals, pcv_vals),
  beside=TRUE, col=c("#2d5a1b","#c4732a"),
  names.arg=rep("",5), ylim=c(0, max(pcv_vals)*1.35),
  main="Genotypic (GCV) & Phenotypic (PCV) Coefficients of Variation\nChickpea Germplasm",
  ylab="Coefficient of Variation (%)", las=1, cex.main=0.98)
mtext(trt_names, side=1, at=colMeans(x_pos), line=1.5, cex=0.82)
abline(h=10, lty=2, col="blue",  lwd=1.5); abline(h=20, lty=2, col="red", lwd=1.5)
text(par("usr")[1]+0.3, 10.4, "10% (Low/Mod)", cex=0.72, col="blue", adj=0)
text(par("usr")[1]+0.3, 20.4, "20% (Mod/High)", cex=0.72, col="red",  adj=0)
legend("topright", legend=c("GCV","PCV"), fill=c("#2d5a1b","#c4732a"), bty="n", cex=0.9)
box(); dev.off()
cat("\nPlot saved.\n")

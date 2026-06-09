# ============================================================
# 3_1_pooled_anova.r  —  Pooled (Combined) ANOVA
# SALF AARBDAT 2026 — Day 5, Chapter 3
# ============================================================
library(agricolae)

cat("==========================================================\n")
cat("  CHAPTER 3: POOLED (COMBINED) ANOVA\n")
cat("  5 Sorghum Genotypes x 3 Environments x 3 Replications\n")
cat("==========================================================\n\n")

set.seed(100)
df_met <- expand.grid(Genotype=paste0("G",1:5),
                      Env=paste0("E",1:3), Rep=paste0("R",1:3),
                      stringsAsFactors=TRUE)
G_eff <- c(0,2.0,-0.8,2.8,-1.2)
E_eff <- c(0,4.5,-2.0)
GE_m  <- matrix(c(0.0,0.5,-0.3,0.2,-0.4,
                  -0.5,0.3,0.8,-0.6,0.0,
                   0.5,-0.8,-0.5,0.4,0.4), nrow=5)
for(i in seq_len(nrow(df_met))){
  g <- as.integer(df_met$Genotype[i])
  e <- as.integer(df_met$Env[i])
  df_met$yield[i] <- 22+G_eff[g]+E_eff[e]+GE_m[g,e]+rnorm(1,0,1.0)
}
df_met$yield <- round(df_met$yield,2)
print(df_met)
e_n <- 3; r_n <- 3; g_n <- 5

cat("--- 1. MEAN YIELD TABLE (Genotype x Environment) ---\n")
means_ge <- tapply(df_met$yield, list(df_met$Genotype,df_met$Env), mean)
# taaply(values to analyze, grouping factors, function to apply)
# Mean yield for each Genotype within each Environment
print(round(means_ge,2))
cat("\nGenotype Marginal Means:\n")
print(round(tapply(df_met$yield,df_met$Genotype,mean),2))
cat("\nEnvironment Marginal Means:\n")
print(round(tapply(df_met$yield,df_met$Env,mean),2))
cat(sprintf("Grand Mean: %.2f t/ha\n\n",mean(df_met$yield)))

# ---- ANOVA ----
cat("--- 2. POOLED ANOVA TABLE ---\n")
model_pool <- aov(yield ~ Env/Rep + Genotype + Genotype:Env, data=df_met)
# Env/Rep = Env + Env:Rep
aov_sum_raw <- summary(model_pool)
print(aov_sum_raw)
aov_p  <- aov_sum_raw[[1]]

# Use indices directly
# Row order in R's aov: Env, Genotype, Env:Rep, Genotype:Env, Residuals
MS_G   <- aov_p[2, "Mean Sq"]   # Genotype
MS_GE  <- aov_p[4, "Mean Sq"]   # Genotype:Env
MS_Err <- aov_p[5, "Mean Sq"]   # Residuals
df_G   <- aov_p[2, "Df"]
df_GE  <- aov_p[4, "Df"]
df_Err <- aov_p[5, "Df"]

cat(sprintf("\nMS_G=%.3f  MS_GxE=%.3f  MS_Err=%.3f\n\n", MS_G, MS_GE, MS_Err))

cat("--- 3. CORRECTED F-TESTS ---\n")
F_G  <- MS_G / MS_GE
F_GE <- MS_GE / MS_Err
p_G  <- pf(F_G,  df_G,  df_GE,  lower.tail=FALSE)
p_GE <- pf(F_GE, df_GE, df_Err, lower.tail=FALSE)

siglab <- function(p) if(p<0.001)"***" else if(p<0.01)"**" else if(p<0.05)"*" else "NS"
cat(sprintf("  Genotypes (tested vs MS_GxE): F=%6.2f  p=%.4f  %s\n",F_G,p_G,siglab(p_G)))
cat(sprintf("  G x E   (tested vs MS_Err) : F=%6.2f  p=%.4f  %s\n",F_GE,p_GE,siglab(p_GE)))

cat("\n--- 4. VARIANCE COMPONENTS ---\n")
sigma2_G  <- max((MS_G - MS_GE)/(e_n*r_n), 0)
sigma2_GE <- max((MS_GE - MS_Err)/r_n,     0)
sigma2_e  <- MS_Err
sigma2_P  <- sigma2_G + sigma2_GE/e_n + sigma2_e/(e_n*r_n)

cat(sprintf("  s2_G  = (%.3f - %.3f)/(3x3) = %.4f\n", MS_G, MS_GE, sigma2_G))
cat(sprintf("  s2_GE = (%.3f - %.3f)/3     = %.4f\n", MS_GE, MS_Err, sigma2_GE))
cat(sprintf("  s2_e  = %.4f\n", sigma2_e))
cat(sprintf("  s2_P  = %.4f (total phenotypic)\n", sigma2_P))
cat(sprintf("  %% Genetic  : %.1f%%\n", sigma2_G/sigma2_P*100))
cat(sprintf("  %% G x E    : %.1f%%\n", (sigma2_GE/e_n)/sigma2_P*100))
cat(sprintf("  %% Error    : %.1f%%\n\n", (sigma2_e/(e_n*r_n))/sigma2_P*100))

cat("--- 5. GENOTYPE RANKINGS ---\n")
g_means <- sort(tapply(df_met$yield,df_met$Genotype,mean),decreasing=TRUE)
cat(sprintf("  %-10s  %12s  %5s\n","Genotype","Mean(t/ha)","Rank"))
cat("  ", strrep("-",32),"\n",sep="")
for(i in seq_along(g_means))
  cat(sprintf("  %-10s  %12.2f  %5d\n",names(g_means)[i],g_means[i],i))
cat(sprintf("\n  Pooled Error CV = %.2f%%\n", sqrt(MS_Err)/mean(df_met$yield)*100))

# ---- Plots ----
png("rplots/3_1_pooled_anova_interaction.png", width=800,height=520,res=110)
par(mar=c(5,5.5,4,7),xpd=TRUE)
gcols <- c("#2d5a1b","#c4732a","#1a3a6b","#8b1a1a","#6a0dad")
matplot(1:3, t(means_ge), type="b", pch=15:19, lwd=2.5, lty=1:5,
  col=gcols, xaxt="n", las=1,
  main="G x E Interaction — Sorghum Multi-Environment Trial",
  xlab="Environment", ylab="Mean Grain Yield (t/ha)", cex.main=1.0,
  ylim=range(means_ge)+c(-1,2))
axis(1, at=1:3, labels=paste0("E",1:3))
legend(3.05, max(means_ge)+1.5, legend=paste0("G",1:5),
  col=gcols, pch=15:19, lwd=2, bty="n", cex=0.88, xpd=TRUE, lty=1:5)
box(); dev.off()
cat("\nInteraction plot saved.\n")

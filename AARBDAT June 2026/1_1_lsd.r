# ============================================================
# 1_1_lsd.r  —  LSD Test Analysis
# SALF AARBDAT 2026 — Day 5, Chapter 1
# Grain yield (t/ha) — 5 rice varieties, 4 reps, RCBD
# ============================================================
library(agricolae)

# ---- 1. Dataset (yields per replication) ----
variety <- factor(rep(paste0("V", 1:5), each = 4))
rep_fac <- factor(rep(paste0("R", 1:4), times = 5))
yield   <- c(
  3.98, 4.42, 4.10, 4.30,   # V1  mean ≈ 4.20
  5.40, 5.82, 5.60, 5.58,   # V2  mean ≈ 5.60
  3.55, 3.98, 3.82, 3.85,   # V3  mean ≈ 3.80
  5.88, 6.32, 6.08, 6.12,   # V4  mean ≈ 6.10
  4.68, 5.12, 4.92, 4.96    # V5  mean ≈ 4.92
)
df <- data.frame(variety, rep = rep_fac, yield)

cat("==========================================================\n")
cat("  CHAPTER 1: LEAST SIGNIFICANT DIFFERENCE (LSD) TEST\n")
cat("  5 Rice Varieties x 4 Replications — RCBD\n")
cat("==========================================================\n\n")

cat("--- 1. TREATMENT MEANS (t/ha) ---\n")
tmeans <- round(tapply(df$yield, df$variety, mean), 3)
print(tmeans)
cat("Grand Mean:", round(mean(df$yield), 3), "t/ha\n\n")

# ---- 2. ANOVA ----
cat("--- 2. ANALYSIS OF VARIANCE TABLE ---\n")
model <- aov(yield ~ rep + variety, data = df)
print(summary(model))

aov_sum <- summary(model)[[1]]
MSE     <- aov_sum["Residuals", "Mean Sq"]
dfe     <- aov_sum["Residuals", "Df"]
r_reps  <- 4

cat(sprintf("MSE = %.4f  |  df(error) = %d\n", MSE, dfe)) # f is float, d is integer
cat(sprintf("CV  = %.2f%%\n\n", sqrt(MSE)/mean(df$yield)*100))

# ---- 3. Manual LSD ----
SE_d    <- sqrt(2 * MSE / r_reps)
t_05    <- qt(0.975, dfe)
t_01    <- qt(0.995, dfe)
lsd05   <- t_05 * SE_d
lsd01   <- t_01 * SE_d

cat("--- 3. LSD COMPUTATION ---\n")
cat(sprintf("SE(d)              = sqrt(2 x %.4f / %d) = %.4f\n", MSE, r_reps, SE_d))
cat(sprintf("t(alpha/2, %d)     = %.3f  [at 5%%]\n", dfe, t_05))
cat(sprintf("t(alpha/2, %d)     = %.3f  [at 1%%]\n", dfe, t_01))
cat(sprintf("LSD (5%%)          = %.3f x %.4f = %.4f t/ha\n", t_05, SE_d, lsd05))
cat(sprintf("LSD (1%%)          = %.3f x %.4f = %.4f t/ha\n\n", t_01, SE_d, lsd01))

# ---- 4. LSD test ----
cat("--- 4. LSD TEST — MEAN GROUPINGS (5%) ---\n")
lsd_res <- LSD.test(model, "variety", alpha = 0.05, p.adj = "none", console = FALSE)
grp     <- lsd_res$groups
grp     <- grp[order(grp$yield, decreasing = TRUE), ]

cat(sprintf("  LSD(5%%) = %.4f t/ha\n\n", lsd_res$statistics$LSD))
cat(sprintf("  %-10s %10s %10s\n", "Variety", "Mean(t/ha)", "Group"))
# %s: character string; %-10s: Left-align string in a field of width 10; 
# %10s: Right-align string in a field of width 10

cat("  ", strrep("-", 32), "\n", sep="")
for(i in seq_len(nrow(grp))){
  cat(sprintf("  %-10s %10.3f %10s\n",
    rownames(grp)[i], grp$yield[i], grp$groups[i]))
}

# ---- 5. Pairwise table ----
cat("\n--- 5. PAIRWISE ABSOLUTE DIFFERENCES ---\n")
cat(sprintf("  %-22s %8s  %-10s  %s\n","Pair","| Diff |","vs LSD(5%)","Decision"))
cat("  ", strrep("-",54), "\n", sep="")
m  <- sort(tmeans, decreasing=TRUE); nm <- names(m)
for(i in 1:(length(nm)-1)){
  for(j in (i+1):length(nm)){
    d   <- abs(m[i]-m[j])
    sig <- if(d>=lsd01) "** (1%)" else if(d>=lsd05) "* (5%)" else "NS"
    cat(sprintf("  %-22s %8.3f  %-10s  %s\n",
      paste(nm[i],"vs",nm[j]), d,
      if(d>=lsd05) "> LSD" else "< LSD", sig))
  }
}
cat("\n  * = sig at 5%;  ** = sig at 1%;  NS = not significant\n")

# ---- 6. Plot ----
png("rplots/1_1_lsd_barplot.png", width=760, height=500, res=110)
par(mar=c(5,5.5,4,2))
cols <- c("#2d5a1b","#4a7c35","#7aad5a","#c4732a","#8b4513")
bp <- barplot(grp$yield, names.arg = rownames(grp),
  col = cols[1:nrow(grp)], ylim = c(0,8), las=1,
  main = "Rice Variety Mean Grain Yields with LSD Groupings (5%)",
  xlab = "Variety", ylab = "Grain Yield (t/ha)", cex.main=1.05)
abline(h = lsd05, lty=2, col="red", lwd=2)
for(k in seq_len(nrow(grp))){
  text(bp[k], grp$yield[k]+0.35,
    paste0(round(grp$yield[k],2)," (",grp$groups[k],")"),
    cex=0.88, font=2)
}
legend("topleft",
  legend = c(paste0("LSD(5%) = ",round(lsd05,3)," t/ha"),
             paste0("LSD(1%) = ",round(lsd01,3)," t/ha")),
  lty = c(2,3), col = c("red","blue"), lwd=2, bty="n", cex=0.85)
abline(h = lsd01, lty=3, col="blue", lwd=2)
box(); dev.off()
cat("\nPlot saved.\n")

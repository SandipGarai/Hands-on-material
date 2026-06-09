# ============================================================
# 2_1_duncan.r  —  Duncan's Multiple Range Test (DMRT)
# SALF AARBDAT 2026 — Day 5, Chapter 2
# Grain yield (t/ha) — 5 wheat genotypes, 4 reps, RCBD
# ============================================================
library(agricolae)

# ---- Same dataset as LSD chapter for comparison ----
variety <- factor(rep(paste0("V", 1:5), each = 4))
rep_fac <- factor(rep(paste0("R", 1:4), times = 5))
yield   <- c(
  3.98, 4.42, 4.10, 4.30,
  5.40, 5.82, 5.60, 5.58,
  3.55, 3.98, 3.82, 3.85,
  5.88, 6.32, 6.08, 6.12,
  4.68, 5.12, 4.92, 4.96
)
df <- data.frame(variety, rep = rep_fac, yield)
model <- aov(yield ~ rep + variety, data = df)
aov_sum <- summary(model)[[1]]
MSE  <- aov_sum["Residuals","Mean Sq"]
dfe  <- aov_sum["Residuals","Df"]
r    <- 4

cat("==========================================================\n")
cat("  CHAPTER 2: DUNCAN'S MULTIPLE RANGE TEST (DMRT)\n")
cat("  5 Wheat Genotypes x 4 Replications — RCBD\n")
cat("==========================================================\n\n")

cat("--- 1. ANOVA TABLE ---\n")
print(summary(model))
cat(sprintf("MSE = %.4f  |  df(error) = %d\n\n", MSE, dfe))

# ---- 2. SE of mean ----
SE_xbar <- sqrt(MSE / r)
cat("--- 2. STANDARD ERROR OF MEAN ---\n")
cat(sprintf("S(x̄) = sqrt(%.4f / %d) = %.4f t/ha\n\n", MSE, r, SE_xbar))

# ---- 3. Duncan test ----
cat("--- 3. DUNCAN'S TEST RESULTS ---\n")
dmrt_res <- duncan.test(model, "variety", alpha = 0.05, console = FALSE)
cat(sprintf("  SE(x̄)  = %.4f\n", dmrt_res$statistics$`Std.Err.`))
cat("\nMeans ranked in descending order with DMRT groups:\n")
grp <- dmrt_res$groups
grp <- grp[order(grp$yield, decreasing=TRUE),]
cat(sprintf("  %-10s %10s %10s\n","Genotype","Mean(t/ha)","DMRT Group"))
cat("  ", strrep("-",34),"\n", sep="")
for(i in seq_len(nrow(grp))){
  cat(sprintf("  %-10s %10.3f %10s\n",
    rownames(grp)[i], grp$yield[i], grp$groups[i]))
}

# ---- 4. LSR values and comparison table ----
cat("\n--- 4. LSR VALUES FOR EACH SPAN ---\n")
# Duncan's tabulated r values for alpha=0.05, df=12
# from standard table
duncan_r <- c(3.082, 3.225, 3.313, 3.370)
lsr_vals <- duncan_r * SE_xbar
cat(sprintf("  %-6s %12s %10s\n","Span p","r(0.05,p,12)","LSR_p"))
cat("  ", strrep("-",30),"\n",sep="")
for(i in 1:4){
  cat(sprintf("  %-6d %12.3f %10.4f\n", i+1, duncan_r[i], lsr_vals[i]))
}

cat("\n--- 5. PAIRWISE DMRT COMPARISONS ---\n")
m  <- sort(tapply(df$yield,df$variety,mean), decreasing=TRUE)
nm <- names(m)
t  <- length(m)
cat(sprintf("  %-22s %6s %8s %8s %8s\n",
  "Pair","Span","|Diff|","LSR_p","Decision"))
cat("  ",strrep("-",56),"\n",sep="")
for(i in 1:(t-1)){
  for(j in (i+1):t){
    span_p <- j - i + 1
    # LSR for this span
    lsr_p <- ifelse(span_p <= 5, lsr_vals[span_p-1], lsr_vals[4])
    d <- abs(m[i]-m[j])
    sig <- ifelse(d >= lsr_p, "*", "NS")
    cat(sprintf("  %-22s %6d %8.3f %8.4f %8s\n",
      paste(nm[i],"vs",nm[j]), span_p, d, lsr_p, sig))
  }
}
cat("\n  * = significant;  NS = not significant\n")

# ---- 5. Comparison: LSD vs DMRT ----
cat("\n--- 6. LSD vs DMRT COMPARISON ---\n")
lsd_res <- LSD.test(model,"variety",alpha=0.05,p.adj="none",console=FALSE)
cat(sprintf("  LSD(5%%)  critical value : %.4f t/ha (single value)\n",
  lsd_res$statistics$LSD))
cat(sprintf("  DMRT LSR range          : %.4f — %.4f t/ha (graduated)\n",
  min(lsr_vals), max(lsr_vals)))
cat("\nLSD groupings:\n")
lsd_grp <- lsd_res$groups[order(lsd_res$groups$yield,decreasing=TRUE),]
cat(sprintf("  %-10s %10s %10s\n","Variety","Mean","LSD Group"))
for(i in seq_len(nrow(lsd_grp)))
  cat(sprintf("  %-10s %10.3f %10s\n",rownames(lsd_grp)[i],lsd_grp$yield[i],lsd_grp$groups[i]))
cat("\nDMRT groupings:\n")
cat(sprintf("  %-10s %10s %10s\n","Variety","Mean","DMRT Group"))
for(i in seq_len(nrow(grp)))
  cat(sprintf("  %-10s %10.3f %10s\n",rownames(grp)[i],grp$yield[i],grp$groups[i]))

# ---- 6. Plot ----
png("rplots/2_1_duncan_barplot.png", width=760, height=500, res=110)
par(mar=c(5,5.5,4,2))
cols <- c("#1a3a6b","#2a5aab","#5a8adb","#c4732a","#8b4513")
bp <- barplot(grp$yield, names.arg=rownames(grp),
  col=cols, ylim=c(0,8), las=1,
  main="Wheat Genotype Mean Grain Yields — DMRT Groupings (5%)",
  xlab="Genotype", ylab="Grain Yield (t/ha)", cex.main=1.05)
for(k in seq_len(nrow(grp))){
  text(bp[k], grp$yield[k]+0.32,
    paste0(round(grp$yield[k],2)," (",grp$groups[k],")"),
    cex=0.88, font=2)
}
# Add LSR range band
abline(h = min(lsr_vals)+mean(grp$yield), lty=2, col="darkgreen", lwd=1.5)
legend("topleft",
  legend=paste0("LSR range: ",round(min(lsr_vals),3)," – ",round(max(lsr_vals),3)," t/ha"),
  lty=2, col="darkgreen", lwd=2, bty="n", cex=0.88) # average yield + minimum lsr value
box(); dev.off()
cat("\nPlot saved.\n")

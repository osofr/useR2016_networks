`%+%` <- function(a, b) paste0(a, b)
require("igraph")
require("simcausal")
options(simcausal.verbose=FALSE)

# -----------------------------------------------------------------------------------
# small world (Watts-Strogatz network) model:
# -----------------------------------------------------------------------------------
generate.smallwld <- function(n, dim, nei, p, ...) {
  g <- sample_smallworld(dim = 1, size = n, nei = nei, p = p, loops = FALSE, multiple = FALSE)
  g <- as.directed(g, mode = c("mutual"))
  sparse_AdjMat <- simcausal::igraph.to.sparseAdjMat(g) # From igraph object to sparse adj. matrix:
  NetInd_out <- simcausal::sparseAdjMat.to.NetInd(sparse_AdjMat) # From igraph object to simcausal/tmlenet input (NetInd_k, nF, Kmax):
  return(NetInd_out$NetInd_k)
}

# -----------------------------------------------------------------------------------
# SIMULATION WITH SIMCAUSAL
# -----------------------------------------------------------------------------------
D <- DAG.empty()
D <- D + network("Net", netfun = "generate.smallwld", dim = 1, nei = 9, p = 0.1)
D <- D + node("HUB", distr = "rconst", const = ifelse(nF >= 25, 1, 0))

D <- D +
    node("W1", distr = "rcat.b1", probs = c(0.0494, 0.1823, 0.2806, 0.2680,0.1651, 0.0546)) +
    node("W2", distr = "rbern", prob = plogis(-0.2)) +
    node("netW1W2", distr = "rconst", const = sum(W1[[1:Kmax]]*W2[[1:Kmax]]), replaceNAw0 = TRUE) +
    node("WNoise", distr = "rbern", prob = plogis(-0.4))

D <- D +
    node("PA", distr = "rbern", prob = W2*0.05 + (1-W2)*0.15) +
    node("nF.PA", distr = "rconst", const = sum(PA[[1:Kmax]]), replaceNAw0 = TRUE)

# Define exposure 0/1 as completely random:
D <- D + node("A", distr = "rbern", prob = 0.25)

# Defining the network summary measures based on A:
D <- D + node("sum.net.A", distr = "rconst",
           const = (sum(A[[1:Kmax]])*(HUB==0) + sum((W1[[1:Kmax]] > 4)*A[[1:Kmax]])*(HUB==1)), replaceNAw0 = TRUE)

D <- D +
    node("probY", distr = "rconst",
        const = plogis(ifelse(PA == 1,
                        +5 - 15*(nF.PA < 1), # the probability of maintaining gym membership drops if no friends are PA
                        -8.0 + 0.25*A) +
                        +0.5*sum.net.A + 0.25*nF.PA*sum.net.A + 0.5*nF.PA +
                        +0.5*(W1-1) - 0.58*W2 +
                        -0.5*(3.477-1) + 0.58*0.4496),
        replaceNAw0 = TRUE)
D <- D + node("Y", distr = "rbern", prob = probY)

Dset <- set.DAG(D, n.test = 200)

# -----------------------------------------------------------------------------------
# SIMULATE DATA
# -----------------------------------------------------------------------------------
sim_dat <- sim(Dset, n = 10000)
head(sim_dat)
# NETWORK
netind_cl <- attributes(sim_dat)$netind_cl
NetInd_mat <- attributes(sim_dat)$netind_cl$NetInd
head(NetInd_mat)

# VISUALIZING THE NETWORK
sim_dat_small <- sim(Dset, n = 50)
g <- sparseAdjMat.to.igraph(NetInd.to.sparseAdjMat(NetInd_k=attributes(sim_dat_small)$netind_cl$NetInd, nF=attributes(sim_dat_small)$netind_cl$nF))
par(mar=c(.1,.1,.1,.1))
plot.igraph(g,
  layout=layout.fruchterman.reingold,
  vertex.size=2,
  vertex.label.cex=.3,
  edge.arrow.size=.1)

# -----------------------------------------------------------------------------------
# ESTIMATION WITH TMLENET
# -----------------------------------------------------------------------------------
require("tmlenet")

# DEFINE SUMMARY MEASURES:
sW <-  def_sW(W1, W2) +
       def_sW(netW1W2 = sum(W1[[1:Kmax]]*W2[[1:Kmax]]),
              nF.PA = sum(PA[[1:Kmax]]),
              nFPAeq0.PAeq1 = (nF.PA < 1) * (PA == 1),
              replaceNAw0 = TRUE)

sA <-  def_sA(A, A.PAeq0 = A * (PA == 0)) +
       def_sA(sum.net.A = (sum(A[[1:Kmax]])*(HUB==0) + sum((W1[[1:Kmax]] > 4)*A[[1:Kmax]])*(HUB==1)),
              replaceNAw0 = TRUE)

# STATIC INTERVENTIONS ON A:
intervene_1 <-  def_new_sA(A = 0)
intervene_2 <-  def_new_sA(A = 1 - A)

# STOCHASTIC INTERVENTION ON A:
intervene_stoch <-  def_new_sA(A = rbinom(n = length(A), size = 1, prob = 0.35))

# DYNAMIC/STOCHASTIC INTERVENTION ON A, CONDITIONAL ON THE NUMBER OF FRIENDS (nF)
intervene_dyn <-  def_new_sA(A = rbinom(n = length(A), size = 1, prob = ifelse(nF >= 20, 0.9, 0.1)))

# -----------------------------------------------------------------------------------
# this syntax allows joint interventions (on more than one variable)
# allows defining direct treatment effects, etc...
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# regression formulas
# -----------------------------------------------------------------------------------
Qform <- "Y ~ nF.PA + A.PAeq0 + nFPAeq0.PAeq1 + sum.net.A + PA + W1 + W2"
hform.g0 <- "A + sum.net.A ~ HUB + PA + nF.PA + nFPAeq0.PAeq1"

# -----------------------------------------------------------------------------------
# PERFORM ESTIMATION WITH TMLENET
# -----------------------------------------------------------------------------------
options(tmlenet.verbose = TRUE)
res <- tmlenet(data = sim_dat, sW = sW, sA = sA,
              NETIDmat = NetInd_mat,
              Kmax = ncol(NetInd_mat),
              intervene1.sA = intervene_stoch,
              Qform = Qform,
              hform.g0 = hform.g0,
              hform.gstar = hform.g0,
              optPars = list(
                bootstrap.var = FALSE)
              )

res$EY_gstar1$estimates
res$EY_gstar1$vars
res$EY_gstar1$IC.CIs

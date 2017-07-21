vcos <- function(u, v) {
    (u %*% v)[1,1] / (sqrt(u %*% u)[1,1] * sqrt(v %*% v)[1,1])
}

sigmoid <- function(x) {
    1 / (1 + exp(-x));
}

vcos.matrix <- function(u, v) {
    r <- matrix(rep(0, nrow(u) * nrow(v)), nrow=nrow(u))
    for (i in 1:nrow(u)) {
        for (j in 1:nrow(v)) {
            r[i,j] <- vcos(u[i,], v[j,])
        }
    }
    r
}

sgns.matrix <- function(u, v) {
    sigmoid(u %*% t(v))
}

x <- c(.1, -.2)
y <- c(-.3, .2)
z <- c(.4, 0)

w <- matrix(rbind(x, y, z), nrow=3)
c <- matrix(rbind(z, y, x), nrow=3)

cw.sgns <- sgns.matrix(c, w)
cat('\ncw.sgns:\n')
print(cw.sgns)

nn.1 <- data.frame(
    x=c(1, 0, 0),
    y=c(0, 1, 0),
    z=c(0, 0, 1),
    nn=do.call(
        rbind,
        lapply(1:3, function(i) { which.max(cw.sgns[i,]) })
    )
)
cat('\nnn.1:\n')
print(nn.1)

nn.2 <- data.frame(
    x=c(2, 1, 1, 0, 0, 0),
    y=c(0, 1, 0, 2, 1, 0),
    z=c(0, 0, 1, 0, 1, 2)
)
nn.2$nn <- do.call(
    rbind,
    lapply(1:6, function(i) {
        which.max(nn.2$x[i] * cw.sgns[1,] +
                  nn.2$y[i] * cw.sgns[2,] +
                  nn.2$z[i] * cw.sgns[3,])
    })
)
cat('\nnn.2:\n')
print(nn.2)

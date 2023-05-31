# Charging probability data
charging_probability <- c(0.74, 0.72, 0.64, 0.55, 0.52, 0.46, 0.41, 0.38, 0.34) * 10

# Range data
range <- c(65.88, 68.09, 75.89, 84.13, 87.75, 93.7, 98.3, 102.3, 106.5)


# Objective data
objective <- c(6403675.70, 6386215.10, 5906155.91, 4581748.10, 4547870.42, 3569065.95, 3549481.79, 3548735.18, 2746600.25)

data <- data.frame(charging_probability, range, objective)

model <- lm(objective ~ range, data = data)

# Print the summary of the linear regression model
summary(model)

model$coefficients
##########################################################
data <- read.csv('hpTuning.csv', sep='\t', header = 0)

# Fit a linear regression model to determine the influence of w on g3
model_w <- lm(V6 ~ V3, data = data)

# Fit a linear regression model to determine the influence of c1 on g3
model_c1 <- lm(V6 ~ V1, data = data)

# Fit a linear regression model to determine the influence of c2 on g3
model_c2 <- lm(V6 ~ V2, data = data)

# Print the summary of the models
summary(model_w)
summary(model_c1)
summary(model_c2)

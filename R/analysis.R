library(fpp2)
library(urca)
library(ggplot2)
library(lubridate)
library(forecast)

crime_df <- read.csv("Data/Cleaned/CFW_Monthly_Crime_Count_Cleaned.csv")
crime_df$Date <- as.Date(crime_df$Date)

plot(crime_df)

# extract starting year and month
start_year  <- year(crime_df$Date[1])
start_month <- month(crime_df$Date[1])

crime_ts <- ts(crime_df$Crime_Count,
               start = c(start_year, start_month),
               frequency = 12)

# Divide into the training and testing set
train <- window(crime_ts, end=c(2024,12))
test  <- window(crime_ts, start=c(2025,1))

plot(train,
     main = "Monthly Crime Count – Fort Worth",
     ylab = "Crime Count",
     xlab = "Year")


# Checking Box Cox/ Log requirements
lamb = BoxCox.lambda(crime_df$Crime_Count)
lamb ## close to -1, not required

nsdiffs(train)
# Seasonal Diff
crime_diff <- diff(train, lag = 12, differences = 1)

summary(ur.df(crime_diff,type="drift",lags=12)) ## Shows unit root

crime_double_diff <- diff(crime_diff, differences = 1)

# 3. ADF test on the twice-differenced series
adf_result <- ur.df(crime_double_diff, type = "drift", lags = 12)
summary(adf_result)



ggAcf(crime_diff)
ggtsdisplay(crime_diff)


ggAcf(crime_double_diff)
ggtsdisplay(crime_double_diff)

fit_auto <- auto.arima(train, 
                  seasonal = TRUE, 
                  stepwise = FALSE, 
                  approximation = FALSE)
fit_auto
fc_auto   <- forecast(fit_auto,   h=length(test))
accuracy(fc_auto, test)

fc <- forecast(fit_auto, h = 12)

autoplot(crime_ts) +
  autolayer(fc$mean, series="Forecast", color="red") +
  autolayer(fc$lower[,2], series="95% Lower", linetype="dashed") +
  autolayer(fc$upper[,2], series="95% Upper", linetype="dashed") +
  ggtitle("Crime Count – Actual & Forecast -- Auto Arima") +
  ylab("Crime Count") +
  xlab("Year") +
  theme_minimal()


# -----------------------------------------------
fit_1 <- Arima(crime_ts,
                    order = c(1,1,1),       # (p,d,q)
                    seasonal = c(1,1,1),   # (P,D,Q)
                    #lambda = NULL
               )         # no Box-Cox unless you want one

fit_1
fc1 <- forecast(fit_1, h = 18)

autoplot(crime_ts) +
  autolayer(fc1$mean, series="Forecast", color="red") +
  autolayer(fc1$lower[,2], series="95% Lower", linetype="dashed") +
  autolayer(fc1$upper[,2], series="95% Upper", linetype="dashed") +
  ggtitle("Crime Count – Actual & Forecast") +
  ylab("Crime Count") +
  xlab("Year") +
  theme_minimal()






# -----------------------------------------------
fit_2 <- Arima(train,
               order = c(1,1,1),       # (p,d,q)
               seasonal = c(1,1,1),   # (P,D,Q)
               #lambda = NULL
)         # no Box-Cox unless you want one

fit_2
accuracy(fit_2)
fc2 <- forecast(fit_2, h = 12)

fc_manual <- forecast(fit_2, h=length(test) + 6)
accuracy(fc_manual, test)

autoplot(train) +
  autolayer(fc_manual$mean, series="Forecast", color="red") +
  autolayer(fc_manual$lower[,2], series="95% Lower", linetype="dashed") +
  autolayer(fc_manual$upper[,2], series="95% Upper", linetype="dashed") +
  ggtitle("Crime Count – Actual & Forecast") +
  ylab("Crime Count") +
  xlab("Year") +
  theme_minimal()



fc_final <- forecast(fit_2, h = 18)

# Combine into one ggplot object with proper color mapping
p <- autoplot(train) + 
  autolayer(train, series="Training Data") + 
  autolayer(test, series="Test Data") + 
  autolayer(fc_final$mean, series="Forecast") + 
  autolayer(fc_final$upper[,2], series="95% Upper PI", linetype="dashed") + 
  autolayer(fc_final$lower[,2], series="95% Lower PI", linetype="dashed") + 
  scale_color_manual(
    values = c( "Training Data" = "black", 
                "Test Data" = "blue", 
                "Forecast" = "red", 
                "95% Upper PI" = "darkgreen", 
                "95% Lower PI" = "orange" )) + 
  ggtitle("18-Month Forecast") + 
  ylab("Crime Count") + 
  xlab("Year") + 
  theme_minimal(base_size = 14) + 
  guides(color = guide_legend(title = "Series")) 

p





# --------------------------------------------------------
#ETS

fit_ets <- ets(crime_ts)
fit_ets

fc_ets <- forecast(fit_ets, h = 12)
fc_ets



# ARIMAX with Tempr data
library(dplyr)
tempr_df <- read.csv("Data/Cleaned/CFW_Avg_Tempr_Cleaned.csv")
head(tempr_df)
tempr_df$Date <- as.Date(tempr_df$Date)

df <- crime_df %>%
  inner_join(tempr_df, by = "Date") %>%
  arrange(Date)

head(df)

crime_ts <- ts(df$Crime_Count,
               start = c(2016, 2),
               frequency = 12)

temp_ts <- ts(df$Temp,
              start = c(2016, 2),
              frequency = 12)

temp_train <- temp_ts[1:length(train)]
temp_test  <- temp_ts[(length(train)+1):length(temp_ts)]

fit_xreg <- auto.arima(train,
                       xreg = temp_train,
                       stepwise = FALSE,
                       approximation = FALSE)
fit_xreg
checkresiduals(fit_xreg)


fc_arimax <- forecast(fit_xreg, xreg = temp_test, h = length(test))

acc_arimax <- accuracy(fc_arimax, test)
acc_arimax

p_arimax <- autoplot(train) +
  
  autolayer(train, series = "Training Data") +
  autolayer(test,  series = "Test Data") +
  autolayer(fc_arimax$mean,   series = "Forecast") +
  autolayer(fc_arimax$upper[,2], series = "95% Upper PI", 
            linetype = "dashed") +
  autolayer(fc_arimax$lower[,2], series = "95% Lower PI", 
            linetype = "dashed") +
  
  scale_color_manual(values = c(
    "Training Data"  = "black",
    "Test Data"      = "blue",
    "Forecast"       = "red",
    "95% Upper PI"   = "darkgreen",
    "95% Lower PI"   = "orange"
  )) +
  
  ggtitle("Crime Forecast Using ARIMAX (with Temperature)") +
  ylab("Crime Count") +
  xlab("Year") +
  theme_minimal(base_size = 14) +
  guides(color = guide_legend(title = "Series"))

p_arimax



acc_arimax <- accuracy(fc_arimax, test)
acc_arimax






fit_temp <- ets(temp_ts)
temp_future <- forecast(fit_temp, h = 12)$mean
fc_xreg <- forecast(fit_xreg,
                    xreg = temp_future,
                    h = 12)
fc_xreg

autoplot(fc_xreg) +
  ggtitle("Crime Forecast with Temperature (ARIMAX)") +
  xlab("Year") +
  ylab("Crime Count") +
  theme_minimal()




# ETS prediction -------------------------------------

# --- Fit ETS on training data ---
fit_ets <- ets(train)

summary(fit_ets)

checkresiduals(fit_ets)

fc_ets <- forecast(fit_ets, h = length(test))
accuracy(fc_ets, test)

p2 <- autoplot(train) +
  
  autolayer(train, series="Training Data") +
  autolayer(test,  series="Test Data") +
  autolayer(fc_ets$mean,  series="Forecast") +
  autolayer(fc_ets$upper[,2], series="95% Upper PI", linetype="dashed") +
  autolayer(fc_ets$lower[,2], series="95% Lower PI", linetype="dashed") +
  
  scale_color_manual(values = c(
    "Training Data" = "black",
    "Test Data"     = "blue",
    "Forecast"      = "purple",
    "95% Upper PI"  = "darkgreen",
    "95% Lower PI"  = "orange"
  )) +
  
  ggtitle("ETS Forecast Comparison") +
  ylab("Crime Count") +
  xlab("Year") +
  theme_minimal(base_size = 14) +
  guides(color = guide_legend(title = "Series"))

p2



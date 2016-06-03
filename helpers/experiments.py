#############################################
#               ONLY INDICES                #   
#############################################

#Test 1_1
df = df.filter(regex=("Log_Return_1_Close"))
#Test 1_2
df = df.filter(regex=("Log_Return_.*_Close"))
#Test 1_3
df = df.filter(regex=("RD1_Close"))
#Test 1_4
df = df.filter(regex=("RD.?_Close"))
#Test 1_5
df = df.filter(regex=("RD_P1_Close"))
#Test 1_6
df = df.filter(regex=("RD_P.?_Close"))
#Test 1_7
df = df.filter(regex=("RD_B1_Close"))
#Test 1_8
df = df.filter(regex=("RD_B.?_Close"))


## #Test 2_1
## df = df.filter(regex=("Log_Return_1_(Open|Close)"))
## #Test 2_2
## df = df.filter(regex=("Log_Return_.*_(Open|Close)"))
## #Test 2_3
## df = df.filter(regex=("RD1_(Open|Close)"))
## #Test 2_4
## df = df.filter(regex=("RD.?_(Open|Close)"))
## #Test 2_5
## df = df.filter(regex=("RD_P1_(Open|Close)"))
## #Test 2_6
## df = df.filter(regex=("RD_P.?_(Open|Close)"))
## #Test 2_7
## df = df.filter(regex=("RD_B1_(Open|Close)"))
## #Test 2_8
## df = df.filter(regex=("RD_B.?_(Open|Close)"))

#Test 3_1
df = df.filter(regex=("RD1_(Open|Close|High|Low|Volume)"))
#Test 3_2
df = df.filter(regex=("RD.?_(Open|Close|High|Low|Volume)"))
#Test 3_3
df = df.filter(regex=("RD_P1_(Open|Close|High|Low|Volume)"))
#Test 3_4
df = df.filter(regex=("RD_P.?_(Open|Close|High|Low|Volume)"))
#Test 3_5
df = df.filter(regex=("RD_B1_(Open|Close|High|Low|Volume)"))
#Test 3_6
df = df.filter(regex=("RD_B.?_(Open|Close|High|Low|Volume)"))



#############################################
#              ONLY COMMODITIES             #   
#############################################

#Test 4_1
df = df.filter(regex=("Log_Return_1_USD"))
#Test 4_2
df = df.filter(regex=("Log_Return_.*_USD"))
#Test 4_3
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD1"))
#Test 4_4
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD.?"))
#Test 4_5
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_P1"))
#Test 4_6
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_P.?"))
#Test 4_7
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_B1"))
#Test 4_8
df = df.filter(regex=("^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_B.?"))

#############################################
#                   All                     #   
#############################################

#Test 5_1
df = df.filter(regex=("Log_Return_1_(USD|Close)"))
#Test 5_2
df = df.filter(regex=("Log_Return_.*_(USD|Close)"))
#Test 5_3
df = df.filter(regex=("RD1_(USD|Close)"))
#Test 5_4
df = df.filter(regex=("RD.?_(USD|Close)"))
#Test 5_5
df = df.filter(regex=("RD_P1_(USD|Close)"))
#Test 5_6
df = df.filter(regex=("RD_P.?_(USD|Close)"))
#Test 5_7
df = df.filter(regex=("RD_B1_(USD|Close)"))
#Test 5_8
df = df.filter(regex=("RD_B.?_(USD|Close)"))


## #Test 6_1
## df = df.filter(regex=("Log_RetuLog_Return_1_(USD|Close|Open)"))
## #Test 6_2
## df = df.filter(regex=("Log_Return_.*_(USD|Close|Open)"))
## #Test 6_3
## df = df.filter(regex=("RD1_(USD|Close|Open)"))
## #Test 6_4
## df = df.filter(regex=("RD.?_(USD|Close|Open)"))
## #Test 6_5
## df = df.filter(regex=("RD_P1_(USD|Close|Open)"))
## #Test 6_6
## df = df.filter(regex=("RD_P.?_(USD|Close|Open)"))
## #Test 6_7
## df = df.filter(regex=("RD_B1_(USD|Close|Open)"))
## #Test 6_8
## df = df.filter(regex=("RD_B.?_(USD|Close|Open)"))


#Test 7_1
df = df.filter(regex=("Log_Return_1"))
#Test 7_2
df = df.filter(regex=("Log_Return_.*"))
#Test 7_3
df = df.filter(regex=("RD1"))
#Test 7_4
df = df.filter(regex=("RD_P.?_"))
#Test 7_5
df = df.filter(regex=("RD_P1"))
#Test 7_6
df = df.filter(regex=("RD_P.?_"))
#Test 7_7
df = df.filter(regex=("RD_B1"))
#Test 7_8
df = df.filter(regex=("RD_B.?_"))
#Test 7_9
df = df.filter(regex=(".*"))
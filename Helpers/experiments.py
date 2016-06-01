#############################################
#               ONLY INDICES                #   
#############################################

#Test 1_1
df = df.filter(regex=("Log_Return_Close_1"))
#Test 1_2
df = df.filter(regex=("Log_Return_Close_1"))
#Test 1_3
df = df.filter(regex=("(RD1_Close|" + colY + ")"))
#Test 1_4
df = df.filter(regex=("(RD.?_Close|" + colY + ")"))
#Test 1_5
df = df.filter(regex=("(RD_P1_Close|" + colY + ")"))
#Test 1_6
df = df.filter(regex=("(RD_P.?_Close|" + colY + ")"))
#Test 1_7
df = df.filter(regex=("(RD_B1_Close|" + colY + ")"))
#Test 1_8
df = df.filter(regex=("(RD_B.?_Close|" + colY + ")"))


#Test 2_1
df = df.filter(regex=("Log_Return_.*_1"))
#Test 2_2
df = df.filter(regex=("Log_Return_.*"))
#Test 2_3
df = df.filter(regex=("(RD1_(Open|Close)|" + colY + ")"))
#Test 2_4
df = df.filter(regex=("(RD.?_(Open|Close)|" + colY + ")"))
#Test 2_5
df = df.filter(regex=("(RD_P1_(Open|Close)|" + colY + ")"))
#Test 2_6
df = df.filter(regex=("(RD_P.?_(Open|Close)|" + colY + ")"))
#Test 2_7
df = df.filter(regex=("(RD_B1_(Open|Close)|" + colY + ")"))
#Test 2_8
df = df.filter(regex=("(RD_B.?_(Open|Close)|" + colY + ")"))

#Test 3_1
df = df.filter(regex=("(RD1_(Open|Close|High|Low|Volume)|" + colY + ")"))
#Test 3_2
df = df.filter(regex=("(RD.?_(Open|Close|High|Low|Volume)|" + colY + ")"))
#Test 3_3
df = df.filter(regex=("(RD_P1_(Open|Close|High|Low|Volume)|" + colY + ")"))
#Test 3_4
df = df.filter(regex=("(RD_P.?_(Open|Close|High|Low|Volume)|" + colY + ")"))
#Test 3_5
df = df.filter(regex=("(RD_B1_(Open|Close|High|Low|Volume)|" + colY + ")"))
#Test 3_6
df = df.filter(regex=("(RD_B.?_(Open|Close|High|Low|Volume)|" + colY + ")"))



#############################################
#              ONLY COMMODITIES             #   
#############################################

#Test 2_1
df = df.filter(regex=("Log_Return_.*_1"))
#Test 2_2
df = df.filter(regex=("Log_Return_.*"))
#Test 2_3
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD1|" + colY + ")"))
#Test 2_4
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD.?|" + colY + ")"))
#Test 2_5
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_P1|" + colY + ")"))
#Test 2_6
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_P.?|" + colY + ")"))
#Test 2_7
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_B1|" + colY + ")"))
#Test 2_8
df = df.filter(regex=("(^(GOLD|SILVER|PLAT|OIL_BRENT)_RD_B.?|" + colY + ")"))

#############################################
#                   All                     #   
#############################################

#Test 1_1
df = df.filter(regex=("Log_Return_Close_1"))
#Test 1_2
df = df.filter(regex=("Log_Return_Close_1"))
#Test 1_3
df = df.filter(regex=("(RD1_(USD|Close)|" + colY + ")"))
#Test 1_4
df = df.filter(regex=("(RD.?_(USD|Close)|" + colY + ")"))
#Test 1_5
df = df.filter(regex=("(RD_P1_(USD|Close)|" + colY + ")"))
#Test 1_6
df = df.filter(regex=("(RD_P.?_(USD|Close)|" + colY + ")"))
#Test 1_7
df = df.filter(regex=("(RD_B1_(USD|Close)|" + colY + ")"))
#Test 1_8
df = df.filter(regex=("(RD_B.?_(USD|Close)|" + colY + ")"))


#Test 2_1
df = df.filter(regex=("Log_Return_.*_1"))
#Test 2_2
df = df.filter(regex=("Log_Return_.*"))
#Test 2_3
df = df.filter(regex=("(RD1_(USD|Close|Open)|" + colY + ")"))
#Test 1_4
df = df.filter(regex=("(RD.?_(USD|Close|Open)|" + colY + ")"))
#Test 1_5
df = df.filter(regex=("(RD_P1_(USD|Close|Open)|" + colY + ")"))
#Test 1_6
df = df.filter(regex=("(RD_P.?_(USD|Close|Open)|" + colY + ")"))
#Test 1_7
df = df.filter(regex=("(RD_B1_(USD|Close|Open)|" + colY + ")"))
#Test 1_8
df = df.filter(regex=("(RD_B.?_(USD|Close|Open)|" + colY + ")"))


#Test 3_1
df = df.filter(regex=("(RD1|" + colY + ")"))
#Test 3_2
df = df.filter(regex=("(RD_P.?_|" + colY + ")"))
#Test 3_3
df = df.filter(regex=("(RD_P1|" + colY + ")"))
#Test 3_4
df = df.filter(regex=("(RD_P.?_|" + colY + ")"))
#Test 3_5
df = df.filter(regex=("(RD_B1|" + colY + ")"))
#Test 3_6
df = df.filter(regex=("(RD_B.?_|" + colY + ")"))
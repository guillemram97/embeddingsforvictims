ones = {"1":"One","2":"Two","3":"Three","4":"Four",
"5":"Five","6":"Six", "7":"Seven","8":"Eight","9":"Nine"}
ones_B = {"01":"One","02":"Two","03":"Three","04":"Four",
"05":"Five","06":"Six", "07":"Seven",
"08":"Eight","09":"Nine"}
afterones = {"10":"Ten","11":"Eleven","12":"Twelve",
"13":"Thirteen","14":"Fourteen","15":"Fifteen",
"16":"Sixteen", "17":"Seventeen",
"18":"Eighteen","19":"Nineteen"}
tens = {"2":"Twenty","3":"Thirty","4":"Fourty",
"5":"Fifty","6":"Sixty", "7":"Seventy","8":"Eighty",
"9":"Ninety"}
grand={0:" Billion, ",1:" Million, ",2:" Thousand, ",3:""}

#Function converting number to words
def num_to_wrds(val, lead=False, sign=' '):
    val=str(val)
    ans = ""
    if lead:
        if len(val)>6 and val[-6:]=='000000':
            aux=val[:-6]
            return str(aux)+ ' million'
        elif len(val)>3 and val[-3:]=='000':
            aux=val[:-3]
            return str(aux)+ ' thousand'
    if len(val)>6:
        if val[:-6]!='000':
            aux=val[:-6]
            while len(aux)<3: aux='0'+aux
            ans=num_to_wrds(aux, sign=sign)+ ' million '
        val=val[-6:]
    if len(val)>3:
        if val[:-3]!='000':
            aux=val[:-3]
            while len(aux)<3: aux='0'+aux
            ans=ans+num_to_wrds(aux, sign=sign)+ ' thousand '
        val=val[3:]
    while len(val)<3: val='0'+val
    if val[0] in ones:
        x = val
        ans = ans + ones[val[0]] + " hundred "
        #if val[1:]!='00':
            #ans = ans + " and "
    if val[1:] in afterones:
        ans = ans + afterones[val[1:]] + " "
    elif val[1] in tens:
        ans = ans + tens[val[1]] 
        if val[2:3] in ones or val[1:3] in ones_B: 
            ans=ans+sign
        if val[2:3] in ones:
            ans = ans + ones[val[2]]
    if val[1:3] in ones_B:
        ans = ans + ones[val[2]]
    if ans=='': return ans
    while ans[0]==' ': ans=ans[1:]
    while ans[-1]==' ': ans=ans[:-1]
    while ans!=ans.replace('  ', ' '):
        ans=ans.replace('  ', ' ')
    return str(ans.lower())

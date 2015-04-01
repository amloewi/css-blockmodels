library(data.table)
library(kernlab)
library(np)

adf = read.csv("~/Dropbox/css_models/adf.csv")
fdf = read.csv("~/Dropbox/css_models/fdf.csv")

find.threshold = function(y, p, tp.value=.5, tn.value=.5){
    values = array(dim=100)
    thresholds = seq(0,1,length=100)
    for(tix in 1:100){
        t = thresholds[tix]
        tp.rate = sum( y & (p >= t))/sum( y)
        tn.rate = sum(!y & (p <  t))/sum(!y)
        
        values[tix] = (tp.value*tp.rate + tn.value*tn.rate)
    }
    return(thresholds[which.max(values)])
}

# WHAT THE FUCK -- ... 
ca.f =   formula(adf$ca ~ data.matrix(adf[,-c(1:5)]))

ca.lm =       lm(ca.f)#adf$ca ~ adf[,-c(1:4)])
ca.lr =      glm(ca.f, family=binomial)
ca.gam =     gam(ca.f, family=binomial)

ca.svm =      ksvm(ca.f, type="C-svc")
ca.svm.vals = predict(ca.svm)

# THIS IS TAKING A MILLION YEARS
#ca.bw <- npregbw(xdat=data.matrix(adf[,-c(1:4)]), ydat=adf$ca)#, data=adf)
#ca.kr <- npreg(bws = ca.bw, gradients = TRUE)
#

ca.lm.t  = find.threshold(adf$ca, ca.lm$fitted.values)
ca.lr.t  = find.threshold(adf$ca, ca.lr$fitted.values)
ca.gam.t = find.threshold(adf$ca, ca.gam$fitted.values)

# All ~ 86%
sum(adf$ca & (ca.lm$fitted.values  >= ca.lm.t))  /sum(adf$ca)
sum(adf$ca & (ca.lr$fitted.values  >= ca.lr.t))  /sum(adf$ca)
sum(adf$ca & (ca.gam$fitted.values >= ca.gam.t)) /sum(adf$ca)

# All ~ 80%
sum(!adf$ca & (ca.lm$fitted.values  < ca.lm.t))  /sum(!adf$ca)
sum(!adf$ca & (ca.lr$fitted.values  < ca.lr.t))  /sum(!adf$ca)
sum(!adf$ca & (ca.gam$fitted.values < ca.gam.t)) /sum(!adf$ca)

# 80 and 98. -- Better, as a sum. How about -- without the block? Doesn't look important.
sum(adf$ca & ca.svm.vals) /sum(adf$ca)
sum(!adf$ca & !ca.svm.vals) /sum(!adf$ca)
# => 80, 98 ... 


# AND NOW -- ... um, also, CV? Uh, ever?
test.people = 1:5 # Not random ... yet. sample(1:21, 5)
test.data  = adf[,]
train.data =
ca.svm =      ksvm(ca.f, type="C-svc")
ca.svm.vals = predict(ca.svm)
sum(adf$ca & ca.svm.vals) /sum(adf$ca)
sum(!adf$ca & !ca.svm.vals) /sum(!adf$ca)


library(data.table)
library(kernlab)
library(np)

adf = read.csv("~/Dropbox/css_models/adf.csv")
fdf = read.csv("~/Dropbox/css_models/fdf.csv")

# Zeroing out people observing their own relationships

adf0 = adf
fdf0 = fdf
for(i in 1:nrow(adf)){
    sender   = paste("X", adf[i, "sender"]   +1, sep="")
    receiver = paste("X", adf[i, "receiver"] +1, sep="")
    #print(cat("Sender: ", sender, " Receiver: ", receiver, " what?"))
    adf0[i, sender] <- 0
    adf0[i, receiver] <- 0
}

ca.f.0 =   formula(adf0$ca ~ data.matrix(adf0[,-c(1:5)]))

ca.svm.0 =      ksvm(ca.f.0, type="C-svc")
ca.svm.vals.0 = predict(ca.svm.0)

sum( adf0$ca &  ca.svm.vals.0) /sum( adf0$ca)
sum(!adf0$ca & !ca.svm.vals.0) /sum(!adf0$ca)
# => ... 71, 98 (68, 97 w/out 'block')
# Versus 76, 98.

# So, not so bad.


# Did ANYbody say yes?
old.vals = (rowSums(adf0[,-c(1:6)]) > 3)+0
before = old.vals
# best.guss & ca, etc. (starting accurac)
model.formula = formula(old.vals ~ data.matrix(adf0[,-c(1:6)]))
while(TRUE){
    bg.svm =   ksvm(model.formula, type="C-svc")
    new.vals = predict(bg.svm)
    if(all(new.vals==old.vals)){
        break
    }
    old.vals = new.vals
    model.formula = formula(new.vals ~ data.matrix(adf0[,-c(1:6 )]))
}
sum(
    sum( adf$sa &  before) /sum( adf$sa) * 
    sum(!adf$sa & !before) /sum(!adf$sa)
)

sum(
    sum( adf$sa &  new.vals) /sum( adf$sa) *
    sum(!adf$sa & !new.vals) /sum(!adf$sa)
)


accuracies = array(dim=c(20,21))
# For different numbers of people (in order, for now)
for(n in 1:20){
    for(itr in 1:10){
        # Train on the personally observed edges
        sampled = sample(1:21, size=n)
        relevant = adf0$sender %in% sampled | adf0$receiver %in% sampled
        train = adf0[ relevant,]
        test  = adf0[!relevant,]
        # Test on the rest ... see how that goes
        model.formula = formula(train$sa ~ data.matrix(train[,-c(1:6)]))
        train.svm =   ksvm(model.formula, type="C-svc")
        out.sample = predict(train.svm, data.matrix(test[,-c(1:6)]))
        # Save and plot how ... accurate this is. Yeah.
        accuracies[itr,  n] = sum( out.sample &  test$sa)/sum( test$sa)
        accuracies[itr+1,n] = sum(!out.sample & !test$sa)/sum(!test$sa)
    }
}
plot(accuracies[1,])
for(i in 2:20){
    lines(accuracies[i,], col=ifelse(i%%2==0, 'red', 'black'))
}




# LAB NOTES, ASSHOLE










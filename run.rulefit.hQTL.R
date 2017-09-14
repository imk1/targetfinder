rm(list=ls())
args <- commandArgs(trailingOnly=TRUE); # Read Arguments from command line
nargs = length(args); # number of arguments
rm.target=T  # Set to true if you want to remove target variable from predictors
use.null=F # Set to T if you want to compute null models for feature interactions

# Print usage
print.usage <- function(){
  cat("Rscript run.model.posneg.R [posRdata] [negRdata] [outputFile] [workingDir]\n")
  cat("Combines a positive and negative set, learns a rulefit model\n")
  cat("   [posRdata]: .Rdata file containing association matrix\n")
  cat("   [negRdata]: .Rdata file containing association matrix\n")
  cat("   [outputFile]: Path to output File\n")
  cat("   [replaceFlag]: (OPTIONAL) If set to a F, then if output file exits, the run will quit and not replace it, DEFAULT:F\n")
  cat("   [workingDir]: (OPTIONAL) working directory, DEFAULT: tempfile()\n")
}

if (nargs < 3) {
  print.usage()
  q(save="no",status=1)
}

pos.Rdata.file <- args[[1]] # Association matrix Rdata file (Contains list called assoc.data)
if (! file.exists(pos.Rdata.file)) {
  cat("Positive Input matrix Rdata file ",pos.Rdata.file,"does not exist\n")
  q(save="no",status=1)
}

neg.Rdata.file <- args[[2]] # Association matrix Rdata file (Contains list called assoc.data)
if (! file.exists(neg.Rdata.file)) {
  cat("Negative Input matrix Rdata file ",neg.Rdata.file,"does not exist\n")
  q(save="no",status=1)
}

output.file <- args[[3]] # Output File name

replace.flag <- F # do not replace existing file
if (nargs > 3) {
  replace.flag <- as.logical(args[[4]])
}

work.dir <- tempfile() # Default work directory
if (nargs > 4) {
  work.dir <- args[[5]]
}
if (! file.exists(work.dir)) {
  dir.create(work.dir)
}

platform <- "linux"

if (! file.exists(Sys.getenv("RULEFITBASE"))) {
	stop( sprintf("Rulefit code base directory: %s does not exist", Sys.getenv("RULEFITBASE")) )
}
  
system( sprintf( "rm -rf %s/*", work.dir))
system( sprintf( "cp -r %s/* %s" , Sys.getenv("RULEFITBASE") , work.dir ) ) # copy rulefit package to working directory
rfhome <- work.dir
source(file.path(rfhome,"rulefit.r"))
library(akima,lib.loc=rfhome)

if (! is.character(output.file)) {
	stop('File name must be a string')
}
  
output.file.parts <- strsplit(as.character(output.file), .Platform$file.sep, fixed=TRUE)[[1]] # split on file separator
if (length(output.file.parts) == 1) { # if no path then just the file name itself
	output.file.path <- '.'
	output.file.fullname <- file.parts
} else {
	output.file.path <- paste(output.file.parts[1:(length(output.file.parts)-1)], collapse=.Platform$file.sep) # 1:last-1 token is path
	output.file.fullname <- output.file.parts[length(output.file.parts)] # last token is filename
}        
output.file.fullname.parts <- strsplit(output.file.fullname,'.',fixed=TRUE)[[1]] # split on .
if (length(output.file.fullname.parts) == 1) { # if no extension
	output.file.ext <- ''
	output.file.name <- file.fullname.parts
} else {
	output.file.ext <- paste('.', output.file.fullname.parts[length(output.file.fullname.parts)], sep="") # add the . to the last token
	output.file.name <- paste(output.file.fullname.parts[1:(length(output.file.fullname.parts)-1)], collapse=".")
}
output.dir <- output.file.path

if (! file.exists(output.dir)) {
  dir.create(output.dir, recursive=T)
}

if (! replace.flag && file.exists(output.file)) {
  cat("Output file exists: ",output.file,". Use replace.flag=T to replace the file\n")
  q(save="no",status=0)
}

# ================
# Run Rulefit
# ================
# $assoc.matrix(DATAFRAME): dataFrame that is the association matrix (rows: TF sites, cols: partner TFs)
#    rownames are peakIDs. Rows are sorted by the numeric part of PeakIDs. First column is target
# $target.name(STRING): name of target
# $assoc.mtrx.file <- path of association .mtrx file
# $assoc.R.file <- path of association .Rdata file
# $expr.val <- expression values (if not valid it is assigned NA)

# Compute optimal rulefit model and variable importance
#   rulefit.results$rfmod
#   rulefit.results$dataset
#   rulefit.results$vi
#   rulefit.results$int.strength
#   rulefit.results$pair.interactions

cat("Computing model and variable importance ...\n")

pos.assoc.data <- read.csv(pos.Rdata.file, row.names=1)
neg.assoc.data <- read.csv(neg.Rdata.file, row.names=1)
pos.names <- colnames(pos.assoc.data)
neg.names <- colnames(neg.assoc.data)
common.names <- intersect(pos.names, neg.names)
if ( (length(pos.names) != length(common.names)) | (length(neg.names) != length(common.names)) ) {
warning("All Positive and negative set features do not match. Using intersection")    
}
pos.assoc.data <- pos.assoc.data[,common.names]
neg.assoc.data <- neg.assoc.data[,common.names]
x.vals <- as.data.frame( rbind( pos.assoc.data, neg.assoc.data ) ) # put negative set below positive set

# Set labels
n.pos <- dim(pos.assoc.data)[[1]] # number of positive examples
n.neg <- dim(neg.assoc.data)[[1]] # number of negative examples
y.vals <- as.numeric(x.vals[,1])
y.vals[c(1 : n.pos)] <- 1
y.vals[c( (n.pos+1) : (n.pos+n.neg) )] <- -1

assoc.classf.data <- list(x.vals=x.vals,
						y.vals=y.vals,
						rm.target=T,
						target.name="hQTL" )
						
ntrue <- (assoc.classf.data$y.vals == 1)

rfmod <- rulefit(x=assoc.classf.data$x.vals ,
			model.type="both",
			inter.supp=1,
			xmiss=9e30, 
			y=assoc.classf.data$y.vals , 
			rfmode="class",
			max.rules=50000,
			tree.size=4,
			test.reps=0,
			quiet=T)

# Create place holder for variable importance
features.names <- colnames(assoc.classf.data$x.vals)
n.features <- length(common.names)
vi <- as.data.frame( matrix( data=NA, nrow=1, ncol=n.features) )
colnames(vi) <- features.names

# Create place holder for interaction strengths
int.strength <- data.frame(matrix( data=NA , nrow=1 , ncol=n.features ))
colnames(int.strength) <- features.names
# int.strength expected null
int.strength.null.mean <- int.strength
# int.strength std. null
int.strength.null.std <- int.strength

# Create place holder for pairwise interactions
pair.interactions <- data.frame(matrix( data=NA , nrow=n.features , ncol=n.features ))
rownames(pair.interactions) <- features.names
colnames(pair.interactions) <- features.names
pair.interactions.null.mean <- pair.interactions
pair.interactions.null.std <- pair.interactions

rulefit.results <- list(rfmod=rfmod,
					dataset=assoc.classf.data,
					vi=vi,
					int.strength=int.strength,
					int.strength.null.mean=int.strength.null.mean,
					int.strength.null.std=int.strength.null.std,
					pair.interactions=pair.interactions,
					pair.interactions.null.mean=pair.interactions.null.mean,
					pair.interactions.null.std=pair.interactions.null.std)

select.idx <- (rulefit.results$dataset$y.vals == 1)
vi <- varimp( impord=FALSE , plot=FALSE , x=rulefit.results$dataset$x.vals[ select.idx , ] )
vi <- t(vi$imp)
# Check if vi is all NaNs (happens when model fails due to very few examples)
if (all(!is.finite(vi))) {
vi <- as.data.frame( matrix(0, nrow=1, ncol=n.partners) )
} else {
vi <- as.data.frame(vi)
}
colnames(vi) <- features.names
rulefit.results$vi <- vi
						
save(list="rulefit.results",file=output.file)

# # Compute interaction strength
cat("Computing global interaction strength ...\n")

int.strength <- interact( features.names , plot=F ) 
rulefit.results$int.strength <- as.data.frame(t(int.strength))
colnames(rulefit.results$int.strength) <- features.names
title.name <- paste("Feature Interaction strength:", rulefit.results$dataset$target.name)

if ( (!is.data.frame(int.strength)) && (!is.matrix(int.strength)) ) { # If numeric data
	int.strength <- data.frame( vals=int.strength , names=c(1:length(int.strength)) ) # Convert to data frame with 2 columns    
} else {
	int.strength <- as.data.frame(int.strength)
	colnames(int.strength) <- "vals" # Rename the column of the data frame to vals
	int.strength$names <- rownames(int.strength) # add an extra name column
}

int.strength$names <- features.names # Add labels if necessary

resort.idx <- order(int.strength$vals)
int.strength <- int.strength[resort.idx,]

library(ggplot2)
axes.format <- theme(plot.title = element_text(size=12,vjust=1),                    
				axis.text.x = element_text(size=10,colour="black"),
				axis.text.y = element_text(size=10,colour="black",hjust=1),
				axis.title.x = element_text(size=12),
				axis.title.y = element_text(size=12,angle=90),
				legend.title = element_text(size=10,hjust=0),
				legend.text = element_text(size=10)                      
				)
                     
p1 <- ggplot(int.strength) + geom_bar( aes( x=reorder(names,int.strength) , y=int.strength ), stat="identity", fill=I("grey30") ) 
p1 <- p1 + axes.format + labs(title=title.name) + coord_flip()

if (nrow(int.strength) > 50) {
	p1 <- p1 + theme(axis.text.y = element_text(size=7,colour="black",hjust=1))
}

save(list="rulefit.results",file=output.file)

# Compute pairwise interactions
cat("Computing all pairwise interactions\n")
num.features <- length(rulefit.results$int.strength)
if ( (num.features > 50) && (! is.null(1e-7) ) ) {
	valid.idx <- which(rulefit.results$int.strength >= 1e-7)
} else {
	valid.idx <- c(1:num.features)
}
cat("Computing pairwise interactions for ",length(valid.idx), " of ", length(rulefit.results$int.strength), " predictors\n")
for (vidx in valid.idx) {
	cat("\t",vidx,"..\n")
	
	# Initialize pair.interactions if necessary
	if ( ! any( names(rulefit.results) == "pair.interactions" ) ) {
		rulefit.results$pair.interactions <- data.frame(matrix( data=NA , nrow=length(features.names) , ncol=length(features.names) ) )
		rownames(rulefit.results$pair.interactions) <- features.names
		colnames(rulefit.results$pair.interactions) <- features.names      
	}

	# Compute interaction strengths if int.strength is all NA
	if ( all( is.na(rulefit.results$int.strength) ) ) {
		rulefit.results <- get.int.strength(rulefit.results, use.null=use.null)    
	}
  
	opt.order <- order( rulefit.results$int.strength , decreasing=TRUE ) # sort features by decreasing interaction strength
	target.name <- rulefit.results$dataset$target.name
  
	# Get the predictor whose interactions you want to get
	if (is.null(vidx)){
		vidx <- opt.order[var.rank]
	}
  
	if (! is.numeric(vidx)){  
		vidx <- which(features.names %in% vidx)
	}
  
	# name of predictor whose interactions you want to get
	var.name <- features.names[vidx]
    
	# Get a filtered set of predictors to compare to
	if (length(features.names) > 50) {
		valid.other.idx <- which(rulefit.results$int.strength >= 1e-7)    
	} else {
		valid.other.idx <- c(1:length(features.names))    
	}
    
	other.idx <- setdiff(valid.other.idx,vidx) # All other features
  
	if (use.null) {
		temp.int2var <- twovarint(vidx, other.idx, plot=FALSE , import=T, null.mods=rulefit.results$null.models)
		int2var <- temp.int2var$int
	} else {
		int2var <- twovarint(vidx, other.idx, plot=FALSE , import=T) 
	}  

	if (length(features.names) > 50) {
		topN <- sum(int2var >= 1e-7)
	} else {
		topN <- NULL
	}
    
	title.name=paste("Pairwise Interactions (Ui =",T,") of",var.name,"given",target.name)
  
	rulefit.results$pair.interactions[ vidx , other.idx ] <- int2var
}

save(list="rulefit.results",file=output.file)

# Compute cross-validation error
if (is.character(rulefit.results)) {
	load(rulefit.results)    
}
 
rfrestore( rulefit.results$rfmod , x=rulefit.results$dataset$x.vals , y=rulefit.results$dataset$y.vals, wt=rep(1,nrow(rulefit.results$dataset$x.vals)) )
cat("Computing cross-validation error\n")
nfold = 10
rulefit.results$cv = rfxval (nfold=nfold, quiet=F)
cat("Cross-validation results have been computed\n")
if ( any(grepl( pattern="rmse", x=names(rulefit.results$cv), fixed=T)) ) {
	rulefit.results$cv$rsquare <- 1 - ( rulefit.results$cv$rmse^2 / 
		mean((rulefit.results$dataset$y.vals - mean(rulefit.results$dataset$y.vals,na.rm=T))^2 ,na.rm=T) )
	cat("\tR=",sqrt(rulefit.results$cv$rsquare),"\n")
}

save(list="rulefit.results",file=output.file)

require("animation")

get_agent_data <- function(gen) {
	dat = read.csv(paste("output/best_gen",gen,"_agent_run.csv", sep=''), header=T)
	return(dat)
}

a0 = get_agent_data(0)
a20 = get_agent_data(20)
a120 = get_agent_data(120)

animate_agent_run <- function(dat) {
	oopt = ani.options(interval = 0.02, nmax = nrow(dat))
	## use a loop to create images one by one
	xlim = c(0,50) #c(min(dat$food_x), max(dat$food_x))
	ylim = c(0,50)
	par(bg="white")
	plot(dat$agent_x, dat$agent_y, type='n', xlim=xlim, ylim=ylim)
	ani.record(reset=T)
	for (i in 1:ani.options("nmax")) {
		points(dat[i,]$agent_x, dat[i,]$agent_y, pch=21)
		points(dat[i,]$food_x, dat[i,]$food_y, pch=23) # diamond
		ani.record()
	}
	
	saveGIF(ani.replay(), movie.name="agent_run.gif", ani.width=500, ani.height=500)
	
	## restore the options
	ani.options(oopt)
	## see ?ani.record for an alternative way to set up an animation
	### 2. Animations in HTML pages ###
	saveHTML({
			ani.options(interval = 0.02, nmax = nrow(dat))
			par(mar = c(3, 3, 2, 0.5), mgp = c(2, 0.5, 0), tcl = -0.3, cex.axis = 0.8,
			cex.lab = 0.8, cex.main = 1)
			#brownian.motion(pch = 21, cex = 5, col = "red", bg = "yellow", main="Agent Run")
		}, img.name = "agent_run", title = "Agent Run",
		description = c("Description 1",
			"Description 2"))
}


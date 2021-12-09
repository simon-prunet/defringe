import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def create_movie(noisy, truth, mask, pred):
	'''
		Create animated movie with subplots, 
		from a sequence of 
		noisy: input image sequence
		truth: reference image sequence
		mask : mask image sequence
		pred : predicted image sequence
	'''

	nobs = noisy.shape[2]
	fig, axes = plt.subplots(2,2)
	frames = []
	for i in range(nobs):
		pl1 = axes[0,0].imshow(noisy[:,:,i],animated=True)
		pl2 = axes[0,1].imshow(truth[:,:,i],animated=True)
		pl3 = axes[1,0].imshow(mask[:,:,i] ,animated=True)
		pl4 = axes[1,1].imshow(pred[:,:,i], animated=True)
		frames.append([pl1,pl2,pl3,pl4])
	
	ani = animation.ArtistAnimation(fig,frames,interval=250,blit=True)
	return(ani)





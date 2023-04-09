
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Loading requirements
from core.lib.pyRequirment import *
from cv2 import imread, cvtColor, COLOR_BGR2GRAY

class ownLib():
    # Here you can put your functions. To use, call them using 'lib.mfun'.
    def test(self, *args):
        # [out] <- (this class, other inputs)
        print('Hello world')
        return args
    
    # Neuron synapses moedel which is obtained from an exponential distribution
    def createNeuronsConnections(self, params):
        # 3D connection of neurons and synaptic connections
        Post         = np.zeros((params.mneuro, params.nneuro, params.N_connections), dtype=np.int16)
        ties_stock   = 2000 * params.N_connections
        
        for i in range(0, params.mneuro):
            for j in range(0, params.nneuro):              
                XY      = np.zeros((2, ties_stock), dtype=np.int8)
                R       = np.random.exponential(scale=params.lambdagain, size=ties_stock)
                fi      = 2 * np.pi * np.random.rand(1, ties_stock)
                XY[0,:] = np.int16(R * np.cos(fi))
                XY[1,:] = np.int16(R * np.sin(fi))
                _, idx  = np.unique(XY, axis=1, return_index=True)
                XY      = XY[:, np.sort(idx)] # returns the same data with no repetition
                n       = 0
                
                for k in range(0, np.size(XY, 1)):
                    x = i + XY[0, k]
                    y = j + XY[1, k]
                    # This is distance condition, with sin and cos we applied a distance condition == 1
                    # this line will evaluate that == 0
                    pp = 1 if i==x and j == y else 0
                    if (x>=0 and y>=0 and x<params.mneuro and y<params.nneuro and pp==0):
                        # Returns the linear index equivalents to the rows(x) and columns(y) (based on MATLAB)
                        Post[i,j,n] = y*params.mneuro + x
                        n += 1
                    if n >= params.N_connections: break
        # End for

        Post2 = Post.transpose(2,0,1)
        Post_line = Post2.T.flatten()
        Pre_line = np.zeros(np.size(Post_line), dtype=np.int16)
        k = 0
        for i in range(0, np.size(Post_line), params.N_connections):
            Pre_line[i:(i + params.N_connections)] = k
            k = k + 1
        return [Pre_line.flatten(), Post_line.flatten()]
    # the end of the function

    # Astrocyte synapses moedel
    def createAstrocytesConnections(self, params):
        # Connection of astrocytes
        # Each agent is connected to its closest neighbours (4 sides)
        Post_line = np.zeros((params.quantity_astrocytes*4), dtype=np.int16)
        Pre_line  = np.zeros((params.quantity_astrocytes*4), dtype=np.int16)
        n = 0
        for i in range(0, params.mastro):
            for j in range(0, params.nastro):
                if (i == 0 and j == 0):                                 # Corner top left
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j + 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i + 1)
                    n += 2
                elif (i == params.mastro-1) and (j == params.nastro-1): # Corner bottom right
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    n += 2
                elif (i == 0) and (j == params.nastro-1):               # Corner top right
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i + 1)
                    n += 2
                elif (i == params.mastro-1) and (j == 0):               # Corner bottom left
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j + 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    n += 2
                elif i == 0:                                            # First top row
                    Pre_line[n:(n+3)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = (j + 1)*params.mastro + i
                    Post_line[n+2]    = j*params.mastro + (i + 1)
                    n += 3
                elif i == params.mastro-1:                              # Last bottom row
                    Pre_line[n:(n+3)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = (j + 1)*params.mastro + i
                    Post_line[n+2]    = j*params.mastro + (i - 1)
                    n += 3
                elif j == 0:                                            # First left column
                    Pre_line[n:(n+3)] = j*params.mastro + i
                    Post_line[n]      = (j + 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    Post_line[n+2]    = j*params.mastro + (i + 1)
                    n += 3
                elif j == params.nastro-1:                              # Last right column
                    Pre_line[n:(n+3)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    Post_line[n+2]    = j*params.mastro + (i + 1)
                    n += 3
                elif (i == 0 and j == 0):                               # Corner top left
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j + 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i + 1)
                    n += 2
                elif (i == params.mastro-1) and (j == params.nastro-1): # Corner bottom right
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    n += 2
                elif (i == 0) and (j == params.nastro-1):               # Corner top right
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i + 1)
                    n += 2
                elif (i == params.mastro-1) and (j == 0):               # Corner bottom left
                    Pre_line[n:(n+2)] = j*params.mastro + i
                    Post_line[n]      = (j + 1)*params.mastro + i
                    Post_line[n+1]    = j*params.mastro + (i - 1)
                    n += 2
                elif (i > 0) and (i < params.mastro-1) and (j > 0) and (j < params.nastro-1): # Middle nodes
                    Pre_line[n:(n+4)] = j*params.mastro + i
                    Post_line[n]      = (j - 1)*params.mastro + i
                    Post_line[n+1]    = (j + 1)*params.mastro + i
                    Post_line[n+2]    = j*params.mastro + (i - 1)
                    Post_line[n+3]    = j*params.mastro + (i + 1)
                    n += 4
        # End for
        Pre_line  = Pre_line[0:n]
        Post_line = Post_line[0:n]
        return [Pre_line.flatten(), Post_line.flatten()]
    # the end of the function


    ## Loading and preparing images ---------------------------------------------
    def load_images(self, params):
        images = np.zeros((params.mneuro, params.nneuro, len(params.image_names)))
        i = 0
        for name in params.image_names:
            image = imread(params.images_dir + '/' + name)
            image = cvtColor(image, COLOR_BGR2GRAY)
            images[:,:,i] = image
            i += 1
        return images

    def make_experiment(self, images, params):
        num_images = len(images)
        # Images are affected by noise and prepared
        if hasattr(params, 'learn_order'):
            learn_order = params.learn_order
        else:
            learn_order = self.make_image_order(num_images, 10, True)
        learn_signals = self.make_noise_signals(images, \
                                                learn_order, \
                                                params.mneuro, \
                                                params.nneuro, \
                                                params.variance_learn, \
                                                params.Iapp_learn)
        if hasattr(params, 'test_order'):
            test_order = params.test_order
        else:
            test_order = self.make_image_order(num_images, 1, True)
        test_signals = self.make_noise_signals(images, \
                                               test_order, \
                                               params.mneuro, \
                                               params.nneuro, \
                                               params.variance_test, \
                                               params.Iapp_test)
        I_signals = np.concatenate((learn_signals, test_signals), axis=2)
        I_signals = np.uint8(I_signals)

        # Make image signals
        learn_timeline = self.make_timeline(params.learn_start_time, \
                                            params.learn_impulse_duration, \
                                            params.learn_impulse_shift, \
                                            len(learn_order))
        test_timeline  = self.make_timeline(params.test_start_time, \
                                            params.test_impulse_duration, \
                                            params.test_impulse_shift, \
                                            len(test_order))
        
        full_timeline = np.concatenate((learn_timeline, test_timeline), axis=0)
        full_timeline = np.int0(full_timeline / params.step)
        full_timeline = np.uint16(full_timeline)
        
        timeline_signal_id = np.zeros((params.n), dtype=np.int8)
        timeline_signal_id_movie = np.zeros((params.n), dtype=np.int8)
        
        for i in range(0, I_signals.shape[2]):
            be = full_timeline[i, 0]
            en = full_timeline[i, 1]
            # For simulation
            timeline_signal_id[be : en] = i + 1
            # For video
            be = be - params.before_sample_frames
            en = en + params.after_sample_frames
            timeline_signal_id_movie[be : en] = i + 1
        return [I_signals, full_timeline, timeline_signal_id, timeline_signal_id_movie]
    
    def make_image_order(self, num_images, num_repetitions, need_shuffle):
        image_order = np.zeros((num_images * num_repetitions), dtype=np.int8)
        for id_image in range(0, num_images):
            image_order[(id_image*num_repetitions):((id_image+1)*num_repetitions)] = id_image
        if need_shuffle:
            image_order = image_order(np.random.permutation(num_images*num_repetitions))
        return image_order
    
    def make_noise_signals(self, images, order, height, width, variance, Iapp0):
        signals = np.zeros((height, width, len(order)))
        for i in range(0, len(order)):
            image_id         = order[i]
            signal           = self.make_noise_signal(images[:,:,image_id], height, width, variance, Iapp0)
            signals[:, :, i] = signal
        return signals
    
    def make_noise_signal(self, image, height, width, variance, Iapp0, thr=127):
        image = image[0 : height, 0 : width] < thr
        # rng('shuffle')
        p = np.random.permutation(width*height)
        b = p[0 : np.uint16(width * height * variance)]
        image[b] = ~image[b]
        image = np.double(image) * Iapp0
        return image

    def make_timeline(self, start, duration, step, num_samples):
        timeline = np.zeros((num_samples, 2))
        for i in range(0, num_samples):
            be             = start + step*i
            en             = be + duration
            timeline[i, :] = [be, en]
        return timeline
    ## Loading and preparing images (END) ---------------------------------------

    # Neuron to astro function
    def neuron2astrocyte(self, params, G, maskLine):
        mask = np.reshape(maskLine, (params.nneuro, params.mneuro)).T
        mask = np.single(mask)
        
        glutamate_above_thr = np.reshape(G, (params.nneuro, params.mneuro)).T
        
        neuron_astrozone_activity = np.zeros((params.mastro, params.nastro))
        neuron_astrozone_spikes   = np.zeros((params.mastro, params.nastro), dtype=np.int8)
        
        sj = int(0)
        for j in range(0, params.mneuro, params.az):
            sk = int(0)
            for k in range(0, params.nneuro, params.az):
                # The number of neurons that passed the glu threshold
                neuron_astrozone_activity[j - sj, k - sk] = \
                    np.add.reduce(glutamate_above_thr[j:(j+2), k:(k+2)], axis=(0,1))
                    # np.sum(glutamate_above_thr[j:(j+2), k:(k+2)])  # The 'reduce' order is faster
                # Number of neurons spiking
                neuron_astrozone_spikes[j - sj, k - sk] = \
                    np.add.reduce(mask[j:(j+2), k:(k+2)], axis=(0,1))
                    # np.sum(mask[j:(j+2), k:(k+2)])  # The 'reduce' order is faster
                sk = int((k + 2)/2)
            sj = int((j + 2)/2)
        return neuron_astrozone_activity.T.flatten(), neuron_astrozone_spikes.T.flatten()


# The end of the class
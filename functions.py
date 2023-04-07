
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       http://abolfazldelavar.com
## // --------------------------------------------------------------

# Loading requirements
from core.lib.pyRequirment import *

class ownLib():
    # Here you can put your functions. To use, call them using 'lib.mfun'.
    def test(self, *args):
        # [out] <- (this class, other inputs)
        print('Hello world')
        return args
    
    def createConnections(self, params):
        # 3D connection of neurons and synaptic connections
        Post         = np.zeros((params.mneuro, params.nneuro, params.N_connections), dtype=np.int16)
        ties_stock   = 2000 * params.N_connections
        
        for i in range(0, params.mneuro):
            for j in range(0, params.nneuro):
                # [samples] = fast_weighted_sampling(weights, m)
                
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

        Post2 = Post.transpose(2,1,0)
        Post_line = Post2.flatten()
        Pre_line = np.zeros(np.size(Post_line), dtype=np.int16)
        k = 0
        for i in range(0, np.size(Post_line), params.N_connections):
            Pre_line[i:(i + params.N_connections)] = k
            k = k + 1
        return [Pre_line.flatten(), Post_line.flatten()]
    # the end of the function

# The end of the class
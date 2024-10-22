import numpy as np
from time import sleep
from datetime import datetime

padded=None

def next_state_np(si,new_st):#0.11529006000000001 for H,w=100,100
    global padded
    if padded is None:
        # padded=np.pad(si,[[1,2],[3,4]],'constant',constant_values=[[5,-1],[7,-5]])
        # new_st=np.zeros_like(si)
        padded=np.pad(si,1,'constant',constant_values=0)
    else:
        padded[:,:]=np.pad(si,1,'constant',constant_values=0)
    #9.935e-05
    for i in range(si.shape[0]):
        for j in range(si.shape[1]):
            a=padded[i:i+3,j:j+3]
            # 0.022203249999999997

            n=a.sum()-a[1,1]
            c=a[1,1]
            # 0.23334804999999997
            if c==1:
                new_st[i,j]=1 if n in [2,3] else 0
            else:#if c==0:
                new_st[i,j]=1 if n==3 else 0
            # 0.45784515

import pyopencl as cl

cl_context=None
cl_queue=None

class next_state_opencl:
    def __init__(self, cl_context, cl_queue) -> None:
        self.ctx=cl_context
        self.queue=cl_queue
        self.pt_size=16 # work group size=16*16
        self.prog=cl.Program(self.ctx,'''
__kernel void step(__global const uchar *oldstate,__global uchar *newstate, int w){
    int k=get_global_id(0);
    int i=get_global_id(1);
    int j=get_global_id(2);
                             
    int cp=k*w*w+i*w+j;
                             
    if(i==0 || j==0 || i==w-1 || j==w-1){
        //newstate[cp]=0;
        return;
        }
                             
    uchar s= oldstate[cp-w-1] + oldstate[cp-w] + oldstate[cp-w+1] +
             oldstate[cp-1]   + oldstate[cp+1] +
             oldstate[cp+w-1] + oldstate[cp+w] + oldstate[cp+w+1];
    uchar c=oldstate[cp];
                             
    newstate[cp]=(c)&&(s==2 || s==3)||(!c)&&(s==3);
    //newstate[cp]=c?(s==2 || s==3):(s==3);
}
''').build()
        self.knl=self.prog.step

    def _proc(self,pts):
        R=self.pt_size

        res=np.zeros_like(pts)

        sin=cl.Buffer(self.ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=pts)
        sout=cl.Buffer(self.ctx,cl.mem_flags.WRITE_ONLY ,pts.nbytes)
        
        self.knl(self.queue,pts.shape,(1,R,R),sin,sout,np.int32(R))
        cl.enqueue_copy(self.queue,res,sout)
        return res

    def __call__(self,si,new_st):
      
        H,W=si.shape
        step=self.pt_size-2
        nH=(H-1)//step+1
        nW=(W-1)//step+1

        si1=np.pad(si,[[1,nH*step-H+1],[1,nW*step-W+1]],'constant',constant_values=0)
        big_patches=np.zeros([nH*nW,self.pt_size,self.pt_size],np.uint8)

        k=0
        for i in range(nH):
            for j in range(nW):
                big_patches[k]=si1[i*step:(i+1)*step+2,j*step:(j+1)*step+2]
                k+=1
        
        proc_patches=self._proc(big_patches)
        
        pp_unpad=proc_patches[:,1:1+step,1:1+step].reshape([nH,nW,step,step])
        pp_unpad1=np.concatenate(pp_unpad,1)
        pp_unpad2=np.concatenate(pp_unpad1,1)
        new_st[:,:]=pp_unpad2[:new_st.shape[0],:new_st.shape[1]]


def print_state(step,state):
    ls=['========================== STEP = '+str(step)]
    for i in range(state.shape[0]):
        l=''.join(['░░' if x ==0 else '██' for x in state[i]])
        ls.append(l)
    print('\n'.join(ls))

def main():
    global cl_context
    global cl_queue

    cl_context=cl.create_some_context(False)
    cl_queue=cl.CommandQueue(cl_context)

    next_state_opencla=next_state_opencl(cl_context,cl_queue)

    H,W=20,80
    
    N_STEPS=300

    state=np.zeros((H,W),np.uint8)
    state_b=state.copy()

    step=0
    ts=0

    np.random.seed(997)

    initial_st=(np.random.uniform(0.,1.,(H,W))>0.5)

    for step in range(N_STEPS+1):
        if step==0:
            state[:]=initial_st
        else:
            t0=datetime.now()
            next_state_opencla(state,state_b)
            state,state_b=state_b,state
            ts+=(datetime.now()-t0).total_seconds()
        print_state(step,state)
    print()

    print(ts/(N_STEPS))
    
if __name__=="__main__":
    main()

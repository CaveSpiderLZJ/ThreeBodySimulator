import cv2
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple

# set the simulation parameters
FS = 1000                                   # track samping frequency (Hz)
T = 450                                     # total simulation time (s)
G = 6.67e-11                                # gravity constant (SI)
M1 = 1.25e19                                # mass of star1 (kg)
M2 = 1.68e19                                # mass of star2 (kg)
M3 = 1.73e19                                # mass of star3 (kg)
M4 = 1e3                                    # mass of the planet (kg)
P1 = [-1000, -500, -800]                    # initial position of star1 (m)
P2 = [1200, -800, -2000]                    # initial position of star2 (m)
P3 = [-200, 1500, 1000]                     # initial position of star3 (m)
P4 = [-3100, -4000, 0]                      # initial position of the planet (m)
V1 = [0, 95, 0]                             # initial velocity of star1 (m/s)
V2 = [200, -100, 0]                         # initial velocity of star2 (m/s)
V3 = None                                   # V3 will be set s.t. the momentum of 3 starts sums to 0
V4 = [175, 0, 595]                          # initial velocity of the planet (m/s)

# set the output video parameters
SAVE_PATH = 'out.mp4'                       # the output video file path
T_OUT = 50                                  # the output video time (s)
FPS = 50                                    # the output video frame rate (Hz)
H, W = 960, 1280                            # the output video shape
TAIL = 250                                  # the star track length in frames
X_MIN, X_MAX = -8000, 12000                 # the x range of the bounding box (m)
Y_MIN, Y_MAX = -8000, 12000                 # the y range of the bounding box (m)
Z_MIN, Z_MAX = -8000, 12000                 # the z range of the bounding box (m)


def step(m:np.ndarray, v:np.ndarray, p:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ''' Calculate the next state of the N-star system.
    args:
        m: np.ndarray[(N,), np.float64], mass.
        v: np.ndarray[(N,3), np.float64], velocity.
        p: np.ndarray[(N,3), np.float64], position.
    returns:
        (next_v, next_p), the next state.
    '''
    N = m.shape[0]
    a = np.zeros_like(p)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            r = np.sqrt(np.sum(np.square(p[j,:]-p[i,:])))
            aij = G * m[j] / (r*r)
            dir = (p[j,:]-p[i,:]) / r
            a[i,:] += aij * dir
    next_p = p + (1/FS) * v + 0.5*a*((1/FS)**2)
    next_v = v + (1/FS) * a
    return (next_v, next_p)


def output_video(ps:np.ndarray) -> None:
    ''' Output the three body track video.
    args:
        p: np.ndarray[(T,N,3), np.float64], the start positions.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, (W, H))
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    colors = ['blue', 'orange', 'green', 'black']
    for i in tqdm.trange(len(ps)):
        p = ps[i,:,:]
        ax = plt.axes(projection='3d')
        start = max(0, i-TAIL)
        for j in range(3):
            ax.scatter(p[j,0], p[j,1], p[j,2], color=colors[j], s=80)
            ax.plot(ps[start:i+1,j,0], ps[start:i+1,j,1],
                ps[start:i+1,j,2], color=colors[j])
        ax.scatter(p[3,0], p[3,1], p[3,2], color=colors[3], s=20)
        ax.plot(ps[start:i+1,3,0], ps[start:i+1,3,1],
            ps[start:i+1,3,2], color=colors[3])
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_zlim(Z_MIN, Z_MAX)
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        ax.zaxis.set_major_locator(MultipleLocator(5000))
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f'Three-Body Problem Simulator', fontsize=16)
        buffer, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(buffer, np.uint8).reshape((h, w, 4))
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)
        fig.clear()
    video_writer.release()


if __name__ == '__main__':
    m = np.array([M1, M2, M3, M4], dtype=np.float64)
    p = np.array([P1, P2, P3, P4], dtype=np.float64)
    v = np.array([V1, V2, [0,0,0], V4], dtype=np.float64)  
    v[2,:] = -(m[0]*v[0,:] + m[1]*v[1,:]) / m[2]
    ps = []
    for i in tqdm.trange(T*FS):
        v, p = step(m, v, p)
        ps.append(p[None,:,:])
    ps = np.concatenate(ps, axis=0)
    output_video(ps[::int((T*FS)/(T_OUT*FPS))])
    print('Done!')

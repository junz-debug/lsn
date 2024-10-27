import matplotlib.pyplot as plt
import numpy as np
def show_pictures(data, epoch, idx, rows = 8):#64*3*32*32
    image_show = data.detach().cpu().numpy()
    image_show = image_show.swapaxes(1, 2)
    image_show = image_show.swapaxes(3, 2)#64*32*32*3
    N, H, W, C = image_show.shape[0], image_show.shape[1], image_show.shape[2], image_show.shape[3] 
    cols = int(N/rows)
    X=np.zeros([H*rows,W*cols,3],dtype=float)
    page=0
    for row in range(rows):
        for col in range(cols):
            if page>=N:
                break
            for i in range(H):
                for j in range(W):
                    X[row*H+i][col*W+j]=image_show[page][i][j]
            page+=1
    plt.imshow(X)
    plt.xticks([])#删除坐标刻度
    plt.yticks([])#删除坐标刻度
    plt.show()
    plt.close()

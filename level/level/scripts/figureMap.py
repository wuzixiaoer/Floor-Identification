from PIL import Image
from pylab import array
import matplotlib.pyplot as plt
# 读取图像到数组中
im = array(Image.open('E://learn//Senior//Huawei//J区地图20180614//J1F2.png'))
x = [100,200]
y = [100,200]
plt.plot(x,y,'r*')
# 绘制图像
plt.imshow(im)

plt.show()

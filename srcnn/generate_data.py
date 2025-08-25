"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation ("Microsoft") grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
from srcnn.utils import *
import os
import time
from msanomalydetector.util import average_filter


class gen():
    def __init__(self, win_siz, step, nums):
        self.control = 0
        self.win_siz = win_siz
        self.step = step
        self.number = nums

    # 核心代码：生成异常数据，共七步。
    def generate_train_data(self, value, back_k=0):
        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        if back_k <= 5:
            back = back_k
        else:
            back = 5
        length = len(value)
        tmp = []
        for pt in range(self.win_siz, length - back, self.step):
            head = max(0, pt - self.win_siz)
            tail = min(length - back, pt)

            # 步骤1：数据预处理
            data = np.array(value[head:tail])
            data = data.astype(np.float64)
            data = normalize(data)  # # 归一化到[0,3]范围，数值空间的大小直接影响异常的"表达能力"。在狭小的[0,1]空间里，异常很难"伸展拳脚"，而[0,3]给了异常足够的"发挥空间"。[0,3]归一化本质上是在"放大信号"，让神经网络更容易"听见"异常的声音。
            
            # 步骤2：确定异常点数量
            num = np.random.randint(1, self.number)
            
            # 步骤3：随机选择异常点位置
            ids = np.random.choice(self.win_siz, num, replace=False)

            # 步骤4：初始化标签数组（0代表正常，1代表正常，起手全0）
            lbs = np.zeros(self.win_siz, dtype=np.int64)
            
            # 步骤5：防偏差控制机制：
            # 监控窗口末尾附近（win_siz-6位置）是否有异常， 如果长期没有，强制添加一个防止模型偏差。
            # 这个机制确实是为了确保模型能检测出最新的异常。它解决的是异常检测中的一个经典问题："历史敏感，当下迟钝"
            # 通过强制在窗口末尾注入异常样本，确保模型具备实时异常检测的能力
            if (self.win_siz - 6) not in ids:
                self.control += np.random.random()
            else:
                self.control = 0
            if self.control > 100:
                ids[0] = self.win_siz - 6
                self.control = 0

            # 步骤6：计算异常生成参数，为下一步做准备
            mean = np.mean(data)
            dataavg = average_filter(data)
            var = np.var(data)

            # 步骤7：生成异常并设置标签
            for id in ids:  # 对每个选中的位置注入异常值
                data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)   # 异常强度基于：（平滑值 + 均值） * 高斯噪声 * 方差缩放
                # 将对应标签位置设为1（异常）
                lbs[id] = 1
            tmp.append([data.tolist(), lbs.tolist()])
        return tmp


def auto(dic):
    path_auto = os.getcwd() + '/auto.json'
    auto = {}
    for item, value in dic:
        if value != None:
            auto[item] = value
    with open(path_auto, 'w+') as f:
        json.dump(auto, f)


def get_path(data):
    dir_ = os.getcwd() + '/' + data + '/'
    fadir = [_ for _ in os.listdir(dir_)]
    print(fadir, 'fadir')
    files = []
    for eachdir in fadir:
        files += [dir_ + eachdir + '/' + _ for _ in os.listdir(dir_ + eachdir)]
    print(files, 'files')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--step', type=int, default=64, help='step')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--num', type=int, default=10, help='upper limit value for the number of anomaly points')
    args = parser.parse_args()
    np.random.seed(args.seed)
    auto(vars(args).items())
    files = get_path(args.data)

    train_data_path = os.getcwd() + '/' + args.data + '_' + str(args.window) + '_train.json'
    total_time = 0
    results = []
    print("generating train data")
    generator = gen(args.window, args.step, args.num)
    for f in files:
        print('reading', f)
        in_timestamp, in_value = read_csv(f)
        in_label = []
        if len(in_value) < args.window:
            print("value's length < window size", len(in_value), args.window)
            continue
        time_start = time.time()
        train_data = generator.generate_train_data(in_value)
        time_end = time.time()
        total_time += time_end - time_start
        results += train_data
    print('file num:', len(files))
    print('total fake data size:', len(results))
    with open(train_data_path, 'w+') as f:
        print(train_data_path)
        json.dump(results, f)

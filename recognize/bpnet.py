import random
import math
import recognize.get_inputs as getinput
import numpy as np

random.seed(0)


# 京：北京；津：天津；沪：上海；渝：重庆；
# 冀：河北；豫：河南；云：云南；辽：辽宁；黑：黑龙江；
# 湘：湖南；皖：安徽；闽：福建；鲁：山东；新：新疆；
# 苏：江苏；浙：浙江；赣：江西；鄂：湖北；桂：广西；
# 甘：甘肃；晋：山西；蒙：内蒙；陕：陕西；吉：吉林；
# 贵：贵州；粤：广东；青：青海；藏：西藏；川：四川；
# 宁：宁夏；琼：海南


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# def relu(x):
#     if x > 0:
#         return x
#     else:
#         return 0


def d_sigmoid(x):
    return x * (1 - x)


class bp_net:
    def __init__(self, ni, nh, no, prepare=False):

        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.ci = [[0.0 for j in range(self.nh)] for i in range(self.ni)]
        self.co = [[0.0 for j in range(self.no)] for i in range(self.nh)]

        if not prepare:
            self.wi = [[0.0 for j in range(self.nh)] for i in range(self.ni)]
            self.wo = [[0.0 for j in range(self.no)] for i in range(self.nh)]
            for i in range(self.ni):
                for j in range(self.nh):
                    self.wi[i][j] = random.uniform(-0.2, 0.2)
                    self.ci[i][j] = 0.0

            for i in range(self.nh):
                for j in range(self.no):
                    self.wo[i][j] = random.uniform(-2.0, 2.0)
                    self.co[i][j] = 0.0
        else:
            self.wi = np.loadtxt('./csv/wi.csv', delimiter=',')
            self.wo = np.loadtxt('./csv/wo.csv', delimiter=',')
            # self.wo = []
            #             # for i in wo:
            #             #     self.wo.append([i])

    def forward(self, sub_inputs):
        if len(sub_inputs) != self.ni - 1:
            print('illegal inputs')

        # self.ai = copy.deepcopy(sub_inputs)
        for i in range(self.ni - 1):
            self.ai[i] = sub_inputs[i]
        for j in range(self.nh):
            cnt = 0.0
            for i in range(self.ni):
                cnt += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(cnt)

        for k in range(self.no):
            cnt = 0.0
            for j in range(self.nh):
                cnt += (self.ah[j] * self.wo[j][k])
            self.ao[k] = sigmoid(cnt)
        return self.ao

    def back_propagation(self, targets, n, m):
        outputs_delta = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            outputs_delta[k] = error * d_sigmoid(self.ao[k])

        for j in range(self.nh):
            for k in range(self.no):
                change = outputs_delta[k] * self.ah[j]
                self.wo[j][k] += n * change + m * self.co[j][k]
                self.co[j][k] = change

        hidden_delta = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += outputs_delta[k] * self.wo[j][k]
            hidden_delta[j] = error * d_sigmoid(self.ah[j])

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_delta[j] * self.ai[i]
                self.wi[i][j] += n * change + m * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
            # print('target is {},predict is {}'.format(targets[k], self.ao[k]))
        return error

    def train(self, inputs, outputs, max_iterator=1000, n=0.5, m=0.1):
        for i in range(max_iterator):
            for j in range(len(inputs)):
                sub_input = inputs[j]
                sub_output = outputs[j]
                self.forward(sub_input)
                output = [sub_output] * self.no
                error = self.back_propagation(output, n, m)
                # if i % 50 == 0:
                #     print('Combined error', error)
        np.savetxt('./csv/wi.csv', self.wi, delimiter=',')
        np.savetxt('./csv/wo.csv', self.wo, delimiter=',')

    def test(self, inputs, output, length=1000):
        for i in range(length):
            sub_input = inputs[i]
            x = self.forward(sub_input)
            print('输入是：{}，预测是：{},实际输出是：{}'.format(math.floor(i / 50), self.close_to(output, x), x))

    def close_to(self, output, value):
        mi = 100
        index = 0
        for i in range(len(output)):
            if mi > math.fabs(value - output[i]):
                mi = math.fabs(value - output[i])
                index = i
        return math.floor(index / 50)


if __name__ == '__main__':
    # inputs, outputs = getinput.get_inputs('./data')
    # np.savetxt('./csv/inputs.csv', inputs, delimiter=',')
    # np.savetxt('./csv/outputs.csv', outputs, delimiter=',')
    inputs = np.loadtxt('./csv/inputs.csv', delimiter=',')
    outputs = np.loadtxt('./csv/outputs.csv', delimiter=',')
    # print(outputs)
    bp = bp_net(35, 35 * 2, 10)
    # bp = bp_net(35, 35*32, 1)
    bp.train(inputs, outputs, max_iterator=2)

    expect_bp = bp_net(35, 35 * 2, 1, True)
    expect_bp.test(inputs, outputs)

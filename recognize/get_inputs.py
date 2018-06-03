import matplotlib.image as maping
import matplotlib.pyplot as plt
import os
import math


def get_feature_list(path):
    img = maping.imread(path)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j][0] == 0:
    #             print(0, end=' ')
    #         else:
    #             print(1, end=" ")
    #     print()
    feature_list = []
    for i in range(int(img.shape[0] / 10)):
        for j in range(int(img.shape[1] / 10)):
            cnt = 0
            # print('%d %d %d %d', i * 10, i * 10 + 10, j * 10, j * 10 + 10)

            for m in range(i * 10, i * 10 + 10):
                for n in range(j * 10, j * 10 + 10):
                    if img[m][n][0] == 255:
                        cnt += 1
            feature_list.append(cnt / 100)
    # print(feature_list)
    # print(len(feature_list))
    if max(feature_list) == min(feature_list):
        print(path)
    return feature_list


def get_inputs(path):
    num = 0
    for a, b, c in os.walk('./data'):
        for d in c:
            num += 1
    # feature_list = [[0 for j in range(num)] for i in range(35)]
    feature_list = []
    cnt = 0
    outputs = [math.floor(j / 50) for j in range(num)]
    for root, dir_names, filename in os.walk(path):
        for name in filename:
            # print(root + '\\' + name)
            # img = maping.imread(root + '\\' + name)
            # plt.imshow(img)
            # plt.show()
            sub_list = get_feature_list(root + "\\" + name)
            # print(sub_list)
            # for i in range(len(sub_list)):
            #     feature_list[i][cnt] = sub_list[i]
            cnt += 1
            feature_list.append(sub_list)
            print('cnt is {}'.format(cnt))
    print(len(feature_list), len(feature_list[0]))
    # print(feature_list)
    # print(outputs)
    normal_feature, normal_output = normalization(feature_list, outputs)
    # print(normal_output)
    return normal_feature, normal_output


def normalization(feature, output):
    normal_feature = []
    cnt = 0
    for sub_feature in feature:
        min_mun = min(sub_feature)
        max_mun = max(sub_feature)
        if max_mun == min_mun:
            print('max==min,cnt is{}'.format(cnt), sub_feature)
            continue
        sub_normal_feature = []
        for x in sub_feature:
            xt = (x - min_mun) / (max_mun - min_mun)
            sub_normal_feature.append(xt)
        normal_feature.append(sub_normal_feature)
        cnt += 1
    normal_output = []
    mi = min(output)
    ma = max(output)
    for i in output:
        if mi == ma:
            break
        xt = (i - mi) / (ma - mi)
        normal_output.append(xt)
    return normal_feature, normal_output


if __name__ == '__main__':
    get_inputs('./data')

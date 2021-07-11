from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import matplotlib.pyplot as plt
import math
import random
import numpy
import tensorflow.compat.v1 as tf
import sys  # 导入sys模块
import time
import googletrans

sys.setrecursionlimit(20000)

# Tensorflow 特性处理
tf.disable_v2_behavior()

# Samples
sample_dict=["Parsley sage and thyme",
             "Hello World",
             "True or false",
             "The above condition can be stated in terms of a homomorphism",
             "A quick fox jumps over a lazy dog",
             "Simulated annealing is a probabilistic technique for approximating the global optimum of a given function"]

# 常量预定义
preset_action = 3  #  0-Trainning / 1-Testing / 2-ComplexTest / 3-GivenComplexTest
enable_nn_training = (preset_action == 0)  # 是否进行模型训练?
do_interference_preset = True  # 是否进行干扰
silent_mode = False
character_width = 60
character_height = 80
letter_dict_width = 48  # 保证是4的倍数
letter_dict_height = 60  # 保证是4的倍数
character_condensation = 25
character_cnt_preset = 1
character_size = 50
captcha_mode = 0
character_rotation_lbound = 0  # 旋转干扰下限
character_rotation_ubound = 3  # 旋转干扰上限
cfs_remove_noise_threshold = 20  # CFS 除噪阈值
captcha_dict = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ? "
sample_sentence = sample_dict[4]
various_font = False
do_interference = (not enable_nn_training)

tensorflow_model_checkpoint = 'C:\\Users\\Null\\Desktop\\tensorflow\\'
if enable_nn_training:
    denoise_threshold = [1]
    denoise_sample_length = [1]
else:
    denoise_threshold = [2, 10]
    denoise_sample_length = [1, 2]

if do_interference_preset == False:
    do_interference = False

if various_font:
    font_list = ["arial.ttf", "consola.ttf", "bahnschrift.ttf", "comic.ttf"]
else:
    font_list = ["consola.ttf"]

image_paste_mask = True

if captcha_mode == 0:
    character_cnt = character_cnt_preset
else:
    character_cnt = len(sample_sentence)


# 邻域降噪
def neighbourhood_noise_reduction(img, sampleLen, threshold):
    bin_arr = [[0 for row in range(img.size[1])] for col in range(img.size[0])]
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            for i2 in range(-sampleLen, sampleLen + 1):
                if i2 + i < 0 or i2 + i >= img.size[0]: continue;
                for j2 in range(-sampleLen, sampleLen + 1):
                    if j2 + j < 0 or j2 + j >= img.size[1]: continue;
                    bin_arr[i][j] += img.getpixel((i + i2, j + j2)) == 0 #  统计邻域内文字像素的个数

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if bin_arr[i][j] <= threshold:
                img.putpixel((i, j), 1)
            else:
                img.putpixel((i, j), 0)
    return img


# 图像分割
def vertical_projection(img):
    bin_arr = [0 for col in range(img.size[0])]
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if img.getpixel((i, j)) == 0:
                bin_arr[i] = bin_arr[i] + 1
    return bin_arr


def get_adjacent_pixel(img, x, y, direction):
    if direction == 0:  # Downward
        return img.getpixel((x, y - 1)) == 0
    elif direction == 1:  # Left
        return img.getpixel((x - 1, y)) == 0
    elif direction == 2:  # Right
        return img.getpixel((x + 1, y)) == 0
    elif direction == 3:  # LeftDown
        return img.getpixel((x - 1, y - 1)) == 0
    elif direction == 4:  # RightDown
        return img.getpixel((x + 1, y - 1)) == 0


# 滴水算法
def drip_fall(img, start):
    route = []
    cur_height = img.size[1]
    last_direction = 0  # 0-下,1-左,2-右,3-左下,4-右下
    cur_horz = start
    while cur_height > 0:
        route.append((cur_horz, cur_height)) # 记录路径
        occupied_adj_pixels = 0
        possible_direction = -1
        for i in range(5):
            if get_adjacent_pixel(img, cur_horz, cur_height, i):
                occupied_adj_pixels += 1
            else:
                possible_direction = i
        if occupied_adj_pixels == 5: # 下方邻域全为文字子块，则向下流动
            cur_height = cur_height - 1
            last_direction = 0
        else:
            if possible_direction == 0 or possible_direction == -1: # 下方位置不是文字区域，向下流动
                cur_height = cur_height - 1
                last_direction = 0
            elif possible_direction == 3: # 左下区域不是文字区域，向左下方流动
                cur_height = cur_height - 1
                cur_horz = cur_horz - 1
                last_direction = 3
            elif possible_direction == 4: # 游侠区域不是文字区域，向右下方流动
                cur_height = cur_height - 1
                cur_horz = cur_horz + 1
                last_direction = 4
            elif possible_direction == 1: # 左侧为空时的流动
                if last_direction == 2: # 防止水滴左右往返
                    cur_horz = cur_horz + 1
                    last_direction = 2
                else:
                    cur_horz = cur_horz - 1
                    last_direction = 1
            elif possible_direction == 2: # 右侧为空时的流动
                if last_direction == 1: # 防止水滴左右往返
                    cur_horz = cur_horz - 1
                    last_direction = 1
                else:
                    cur_horz = cur_horz + 1
                    last_direction = 2
    return route


cfs_bin_arr = []
cfs_vis_arr = []
cfs_area = []
cfs_left = []
cfs_right = []

# 对处于一个连通块内的所有字符进行标记
def cfs_dfs(img, x, y, cur_block):
    global cfs_vis_arr # vis_arr 记录DFS访问状态的数组
    global cfs_bin_arr # bin_arr 记录当前位置所处的连通块编号
    # 回溯和剪枝条件
    if x >= img.size[0] or x < 0: return
    if y >= img.size[1] or y < 0: return
    if cfs_vis_arr[x][y]: return
    if img.getpixel((x, y)) != 0: return
    if cfs_bin_arr[x][y] != 0: return
    #进行标记
    cfs_vis_arr[x][y] = True
    cfs_bin_arr[x][y] = cur_block
    cfs_area[cur_block] = cfs_area[cur_block] + 1
    # 遍历相邻区域
    cfs_dfs(img, x + 1, y, cur_block)
    cfs_dfs(img, x - 1, y, cur_block)
    cfs_dfs(img, x, y + 1, cur_block)
    cfs_dfs(img, x, y - 1, cur_block)
    cfs_vis_arr[x][y] = False
    return

# 连通块标记
def cfs_slice(img):
    global cfs_vis_arr
    global cfs_bin_arr
    global cfs_area
    # 初始化
    cfs_bin_arr = [[0 for row in range(img.size[1])] for col in range(img.size[0])]
    cfs_vis_arr = [[0 for row in range(img.size[1])] for col in range(img.size[0])]
    current_block = 0
    cfs_area = [0]
    # 遍历所有像素
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            # 找到一个未被标记连通块编号的像素，以此为基点标记于其相邻的连通块
            if img.getpixel((i, j)) == 0 and cfs_bin_arr[i][j] == 0:
                current_block = current_block + 1
                cfs_area.append(0)
                cfs_dfs(img, i, j, current_block)


def image_slice(img):
    vert_projection = vertical_projection(img)
    slice_starts = []
    block_begin_flag = False


def captcha_gen():
    global cfs_vis_arr
    global cfs_bin_arr
    global cfs_area
    global captcha_dict, sample_sentence, captcha_mode, font_list, character_width, character_height, character_condensation, character_cnt, character_size
    global denoise_threshold, denoise_sample_length
    global do_interference

    for cj in range(1):
        im = Image.new('RGBA', (
            character_cnt * character_width - character_condensation * (character_cnt - 1), character_height), 'white')
        result_capt = [];
        # Paint Texts
        for i in range(character_cnt):
            im_child = Image.new('RGBA', (character_width, character_height), (255, 255, 255))
            fn = ImageFont.truetype(font_list[random.randint(0, len(font_list) - 1)], character_size)
            drw_child = ImageDraw.Draw(im_child)
            if captcha_mode == 0:
                result_capt.append(captcha_dict[random.randint(0, len(captcha_dict) - 1)])
            else:
                result_capt.append(sample_sentence[i])

            drw_child.text((character_width / 2 - character_size / 2, character_height / 2 - character_size / 2),
                           result_capt[i], font=fn,
                           fill=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
            rot_angle = random.randint(character_rotation_lbound, character_rotation_ubound)
            if random.randint(1, 2) == 1: rot_angle = -rot_angle;
            rot_angle_rad = rot_angle / 180 * math.pi
            im_child_rot = im_child.rotate(rot_angle, expand=True, resample=Image.BICUBIC)
            dx1 = int(character_height * math.sin(abs(rot_angle_rad)) + 2)
            dy1 = int(character_width * math.sin(abs(rot_angle_rad)) + 2)
            dx2 = int(character_height * math.cos(abs(rot_angle_rad)) - 1)
            dy2 = int(character_width * math.cos(abs(rot_angle_rad)) - 1)
            im_child_crop = im_child_rot.crop((dy1, dx1, dy2, dx2))
            im_child_crop = im_child_crop.resize((character_width, character_height))

            if image_paste_mask == False:
                im.paste(im_child_crop, (i * character_width - character_condensation * i, 0,
                                         i * character_width + im_child_crop.size[0] - character_condensation * i,
                                         im_child_crop.size[1]))
            else:
                im_mask = im_child_crop.convert("L")
                im_mask = ImageOps.invert(im_mask)
                im_mask = im_mask.convert("1")

                im.paste(im_child_crop, (i * character_width - character_condensation * i, 0,
                                         i * character_width + im_child_crop.size[0] - character_condensation * i,
                                         im_child_crop.size[1]), mask=im_mask)

        # 产生验证码干扰
        if do_interference == True:
            # Draw Points
            drw = ImageDraw.Draw(im)
            rnd_points_max = (im.size[0] - 1) * (im.size[1] - 1)
            for i in range(random.randint(int(rnd_points_max / 10), int(rnd_points_max / 5))):
                drw.point([(random.randint(0, im.size[0] - 1), random.randint(0, im.size[1] - 1))],
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

            # Draw Lines
            for i in range(random.randint(int(im.size[1] / 18), int(im.size[1] / 16))):
                dw = random.randint(0, im.size[0] - 1)
                dv = random.randint(0, im.size[0] - 1)
                drw.line([(dw, dv),
                          (dw + random.randint(0, im.size[1] - 1), dv + random.randint(0, im.size[1] - 1))],
                         fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        answ = "".join(result_capt)

        # im.save("data\\"+"".join(result_capt)+".jpg");
        # im.convert("1").save("data\\"+"".join(result_capt)+"_bw.jpg");
        # im.show()
    return im, answ


# 二值化、去噪和调整
# 图像降噪
def image_noise_reduction(img):
    # 邻域降噪
    im_gray = img.convert("1")
    for i in range(len(denoise_sample_length)):
        im_gray = neighbourhood_noise_reduction(im_gray, denoise_sample_length[i], denoise_threshold[i])
    # 连通块降噪
    if not enable_nn_training:
        cfs_slice(im_gray)  # 对连通块划分
        for i in range(im_gray.size[0]):
            for j in range(im_gray.size[1]):
                if cfs_area[cfs_bin_arr[i][j]] <= cfs_remove_noise_threshold:
                    im_gray.putpixel((i, j), 1)  # 说明此处为噪点，将其删除
                else:
                    im_gray.putpixel((i, j), 0)  # 说明此处不为噪点，将其增强
    return im_gray


# 切割
def captcha_slice(img):
    im_gray = img;
    cfs_slice(im_gray)
    vert_project = vertical_projection(im_gray)
    global cfs_left, cfs_right
    cfs_left = [im_gray.size[0] for i in range(len(cfs_area))]
    cfs_right = [0 for i in range(len(cfs_area))]
    itop = [im_gray.size[1] for i in range(im_gray.size[0])]
    ibottom = [0 for i in range(im_gray.size[0])]
    im_gray_dup = im_gray.copy()
    for i in range(im_gray.size[0]):
        for j in range(im_gray.size[1]):
            cfs_left[cfs_bin_arr[i][j]] = min(cfs_left[cfs_bin_arr[i][j]], i)
            cfs_right[cfs_bin_arr[i][j]] = min(cfs_right[cfs_bin_arr[i][j]], i)
            if im_gray.getpixel((i, j)) == 0:
                itop[i] = min(itop[i], j)
                ibottom[i] = max(ibottom[i], j)
        for j in range(im_gray.size[1]):
            if itop[i] <= j <= ibottom[i]:
                im_gray_dup.putpixel((i, j), 0)
            else:
                im_gray_dup.putpixel((i, j), 1)
    return im_gray
    # im_gray.show();

# 基于竖直投影的切割
def image_slice_by_vertical_projection(img):
    vert_project = vertical_projection(img)
    start = 0 # 字符起始坐标
    start_flag = False # 字符起始标识
    im_list = [] # 切割图片结果
    char_width_list = [] # 字符宽度
    char_width_max = 0 # 最大字符宽
    # 计算最大字符宽度，处理空格
    for i in range(len(vert_project)):
        if start_flag == False and vert_project[i] != 0: # 字符起始
            start = i
            start_flag = True
            continue
        if start_flag == True and vert_project[i] == 0: # 字符终止
            start_flag = False
            char_width_list.append(i-start)
    start_flag = 0
    char_width_max = max(char_width_list)*1.2

    # 计算字符起点和终点，进行切割
    for i in range(len(vert_project)):
        if start_flag == False and vert_project[i] != 0: # 字符起始，将起始位置标记记录
            start = i
            start_flag = True
            continue
        elif start_flag == True and vert_project[i] == 0: # 字符终止，切割放入线性表
            start_flag = False
            im_crop = img.crop((start, 0, i, img.size[1]))
            im_list.append(im_crop)
            start = i
        elif start_flag == False and i-start > char_width_max : # 空白部分宽度大于最大字符宽，计为空格
            im_crop = img.crop((start, 0, i, img.size[1]))
            im_list.append(im_crop)
            start = i
    return im_list


# 图像缩放
def captcha_scale(img):
    v_left = img.size[0] # 左侧非空白区域终止横坐标
    v_right = 0 # 右侧空白区域起始横坐标
    v_top = 0 # 上册空白区域起始纵坐标
    v_bottom = img.size[1] # 下侧空白区域终止纵坐标
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if img.getpixel((i, j)) == 0:
                v_left = min(v_left, i)
                v_right = max(v_right, i)
                v_top = max(v_top, j)
                v_bottom = min(v_bottom, j)
    # 裁切空白并缩放图片
    im_dup = img.copy()
    im_dup = im_dup.crop((v_left, v_bottom, v_right, v_top))
    im_dup = im_dup.resize((letter_dict_width, letter_dict_height))
    return im_dup


# 数值化
def get_text_vector(char):
    vec = numpy.zeros(len(captcha_dict))
    for i in range(len(captcha_dict)):
        if char == captcha_dict[i]:
            vec[i] = 1
    return vec


def img_to_vector(img):
    vec = numpy.array(img)
    if len(vec.shape) > 2:
        vec2 = numpy.mean(vec, -1)
        vec = vec2.flatten()
    else:
        vec = vec.flatten()
    for i in range(vec.size):
        vec[i] = not vec[i]
    return vec


# 训练集
def gen_single_batch_img_visual():
    cp = captcha_gen()
    cp1 = image_noise_reduction(cp[0])
    return cp1,cp[1]


def gen_single_batch_img_visual_2():
    cp = captcha_gen()
    if not silent_mode: cp[0].show()
    cp1 = image_noise_reduction(cp[0])
    return cp1,cp[1]


def gen_single_batch_img():
    cp = captcha_gen()
    cp1 = image_noise_reduction(cp[0])
    cp2 = captcha_scale(cp1)
    return cp2, cp[1]


def gen_single_batch():
    # print("a")
    cp = captcha_gen()
    # print("b")
    cp1 = image_noise_reduction(cp[0])
    # print("c")
    cp2 = captcha_scale(cp1)
    # print("d")
    vc = img_to_vector(cp2)
    # cp2.show()
    # print("e")
    return vc, get_text_vector(cp[1])


def gen_single_batch_2():
    cp = captcha_gen()
    cp1 = image_noise_reduction(cp[0])
    cp2 = captcha_scale(cp1)
    vc = img_to_vector(cp2)
    # cp2.show()
    return vc, cp[1]


# gen_single_batch()


def get_training_batches(batch_size=128):
    bx = numpy.zeros([batch_size, letter_dict_width * letter_dict_height])
    by = numpy.zeros([batch_size, len(captcha_dict)])
    for i in range(batch_size):
        single_batch_data = gen_single_batch()
        bx[i, :] = single_batch_data[0]
        by[i, :] = single_batch_data[1]
        if i % 32 == 0 and i != 0:
            print("Generate Sample: " + str(int(i / batch_size * 100)) + "% Completed")
    print("Generate Sample: 100% Completed")
    return bx, by


# CNN
nn_x = tf.placeholder(tf.float32, [None, letter_dict_width * letter_dict_height])
nn_y = tf.placeholder(tf.float32, [None, len(captcha_dict)])
nn_kp = tf.placeholder(tf.float32)


# 网络建立
def build_neural_network(sample_width=3, fcl_neuron=2048):
    nn_v = tf.reshape(nn_x, shape=[-1, letter_dict_height, letter_dict_width, 1])
    # 第一卷积层
    w_conv1 = tf.Variable(0.01 * tf.random_normal([sample_width, sample_width, 1, 32]))
    b_conv1 = tf.Variable(0.1 * tf.random_normal([32]))
    cov1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(nn_v, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1))
    # 第一池化层
    cov1 = tf.nn.max_pool(cov1,strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1] , padding='SAME')
    cov1 = tf.nn.dropout(cov1, nn_kp)
    # 第二卷积层
    w_conv2 = tf.Variable(0.01 * tf.random_normal([sample_width, sample_width, 32, 64]))
    b_conv2 = tf.Variable(0.1 * tf.random_normal([64]))
    cov2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(cov1, w_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2))
    # 第二池化层
    cov2 = tf.nn.max_pool(cov2,  strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
    cov2 = tf.nn.dropout(cov2, nn_kp)
    # 第一全连接
    wf = tf.Variable(0.01 * tf.random_normal([(letter_dict_height // 4) * (letter_dict_width // 4) * 64, fcl_neuron]))
    bf = tf.Variable(0.1 * tf.random_normal([fcl_neuron]))
    final = tf.reshape(cov2, [-1, wf.get_shape().as_list()[0]])
    final = tf.nn.relu(tf.add(tf.matmul(final, wf), bf))
    final = tf.nn.dropout(final, nn_kp)
    # 输出层
    out_w = tf.Variable(0.01 * tf.random_normal([fcl_neuron, len(captcha_dict)]))
    out_b = tf.Variable(0.1 * tf.random_normal([len(captcha_dict)]))
    ret = tf.add(tf.matmul(final, out_w), out_b)
    return ret


def model_trainer():
    cnn_result = build_neural_network()
    loss_eval_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cnn_result, labels=nn_y))
    optimize_func = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_eval_func)
    forecast_res = tf.reshape(cnn_result, [-1, 1, len(captcha_dict)])
    forecast_p = tf.argmax(forecast_res, 2)
    real_p = tf.argmax(tf.reshape(nn_y, [-1, 1, len(captcha_dict)]), 2)
    accu_p = tf.equal(forecast_p, real_p)
    accuracy = tf.reduce_mean(tf.cast(accu_p, tf.float32))
    saver = tf.train.Saver(max_to_keep=1)
    step = 0
    start_time = time.perf_counter()
    acc_list = []
    loss_list = []
    stable_times = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while 1:
            bx, by = get_training_batches()
            _, loss_ = sess.run([optimize_func, loss_eval_func], feed_dict={nn_x: bx, nn_y: by, nn_kp: 0.70})
            print("[Trainer] Epoch Ends:" + str(step) + " Loss=" + str(loss_))
            loss_list.append(str(loss_))
            if step % 5 == 0 and step != 0:
                print("Now Evaluating...")
                bx, by = get_training_batches()
                acc = sess.run(accuracy, feed_dict={nn_x: bx, nn_y: by, nn_kp: 1})
                print("[Trainer] Training In Progress: Step=" + str(step) + " Accuracy=" + str(
                    acc) + " TimePerStep:" + str(
                    (time.perf_counter() - start_time) / step) + " sec" + " GoodMatches:" + str(stable_times))
                acc_list.append(str(acc))
                if acc > 0.968:
                    stable_times += 1

                if stable_times >= 70:
                    saver.save(sess, tensorflow_model_checkpoint + 'model.ckpt', global_step=step)
                    with open("C:\\Users\\Null\\Desktop\\capt_trainer.csv", 'w') as wf:
                        wf.write(",".join(acc_list))
                    with open("C:\\Users\\Null\\Desktop\\capt_trainer_loss.csv", 'w') as wf2:
                        wf2.write(",".join(loss_list))
                    break

            step += 1


def predict(img):
    output = build_neural_network()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(tensorflow_model_checkpoint)
        saver.restore(sess, ckpt.model_checkpoint_path)
        predict_w = tf.argmax(tf.reshape(output, [-1, len(captcha_dict)]), 1)
        text_list = sess.run(predict_w, feed_dict={nn_x: [img], nn_kp: 1})
        text = text_list[0].tolist()
    return text


def train():
    global nn_x, nn_y, nn_kp
    nn_x = tf.placeholder(tf.float32, [None, letter_dict_width * letter_dict_height])
    nn_y = tf.placeholder(tf.float32, [None, len(captcha_dict)])
    nn_kp = tf.placeholder(tf.float32)
    model_trainer()


def predict_test(test_case_cnt=1000):
    correct_test_case = 0
    half_correct_test_case = 0
    global nn_x, nn_y, nn_kp
    for i in range(test_case_cnt):
        tf.reset_default_graph()
        nn_x = tf.placeholder(tf.float32, [None, letter_dict_width * letter_dict_height])
        nn_y = tf.placeholder(tf.float32, [None, len(captcha_dict)])
        nn_kp = tf.placeholder(tf.float32)
        test_case = gen_single_batch_img()
        predict_answer = predict(img_to_vector(test_case[0]))
        predict_answer_formatted = captcha_dict[predict_answer]
        # print(i)
        if test_case[1] == predict_answer_formatted:
            print("【Correct】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
            correct_test_case += 1
        elif test_case[1].lower() == predict_answer_formatted.lower():
            print("【Partially Correct】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
            half_correct_test_case += 1
        else:
            print("【Wrong】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
    print("【Statistics】 Tested:" + str(test_case_cnt) + " Correct:" + str(
        correct_test_case) + " Partially Correct:" + str(half_correct_test_case))


# 暴力算法 / 其他算法测试
plain_algo_image_list = []
plain_algo_label_list = []


def plain_algo_generate_library(library_size=500):
    test_case_w = gen_single_batch_2()
    print(len(test_case_w[0]))
    for j in range(len(captcha_dict)):
        plain_algo_image_list.append([0 for k in range(len(test_case_w[0]))])
        plain_algo_label_list.append(captcha_dict[j])
    for i in range(library_size):
        test_case = gen_single_batch_2()
        r = 0
        for k in range(len(captcha_dict)):
            if captcha_dict[k] == test_case[1]:
                r = k
        for k in range(len(test_case[0])):
            plain_algo_image_list[r][k] += test_case[0][k]

        if i % 100 == 0:
            print("【PlainAlgo】 Library Creating: " + str(int(i / library_size * 1000) / 10) + "%")
    print("【PlainAlgo】 Library Created")


def plain_algo_predict(img):
    img_2 = img_to_vector(img)
    best_res = 0
    best_ans = ''
    for cj in range(len(plain_algo_image_list)):
        resemblance = 0
        for i in range(len(plain_algo_image_list[cj])):
            # print(str(plain_algo_image_list[cj]))
            if img_2[i] == 1:
                resemblance += plain_algo_image_list[cj][i]
        if resemblance > best_res:
            best_res = resemblance
            best_ans = plain_algo_label_list[cj]
    return best_ans


def plain_algo_predict_test(test_case_cnt=1000):
    plain_algo_generate_library()
    correct_test_case = 0
    half_correct_test_case = 0
    for i in range(test_case_cnt):
        test_case = gen_single_batch_img()
        predict_answer = plain_algo_predict(test_case[0])
        predict_answer_formatted = predict_answer
        if test_case[1] == predict_answer_formatted:
            print("【Correct】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
            correct_test_case += 1
        elif test_case[1].lower() == predict_answer_formatted.lower():
            print("【Partially Correct】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
            half_correct_test_case += 1
        else:
            print("【Wrong】 Answer:" + test_case[1] + " Predict:" + predict_answer_formatted)
    print("【Statistics】 Tested:" + str(test_case_cnt) + " Correct:" + str(
        correct_test_case) + " Partially Correct:" + str(half_correct_test_case))


# 分析
def interpret_image(img):
    global nn_x, nn_y, nn_kp
    img_list = image_slice_by_vertical_projection(img)
    answer = []
    for i in range(len(img_list)):
        img_list[i] = captcha_scale(img_list[i])
        tf.reset_default_graph()
        nn_x = tf.placeholder(tf.float32, [None, letter_dict_width * letter_dict_height])
        nn_y = tf.placeholder(tf.float32, [None, len(captcha_dict)])
        nn_kp = tf.placeholder(tf.float32)
        predict_answer = predict(img_to_vector(img_list[i]))
        answer.append(captcha_dict[predict_answer])
    return "".join(answer)

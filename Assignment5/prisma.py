import math
import cv2
import numpy as np

RGB_TO_LMS = [[0.3811, 0.5783, 0.0402],
              [0.1967, 0.7244, 0.0782],
              [0.0241, 0.1288, 0.8444]]
LMS_TO_RGB = [[4.4679, -3.5873, 0.1193],
              [-1.2186, 2.3809, -0.1624],
              [0.0497, -0.2439, 1.2045]]
LOGLMS_TO_LALPHABETA = [[0.57735, 0.57735, 0.57735],
                        [0.40825, 0.40825, -0.81649],
                        [0.70711, -0.70711, 0.00000]]
LALPHABETA_TO_LOGLMS = [[0.57735, 0.40825, 0.70711],
                        [0.57735, 0.40825, -0.70711],
                        [0.57735, -0.81649, 0.00000]]


def max(a, b):
    if a > b:
        return a
    return b


def min(a, b):
    if a < b:
        return a
    return b


def rgb_to_lalphabeta(rgb):
    # print rgb
    # cv2.imshow('RGB image before conversion', rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    result = np.copy(rgb).astype(np.float64)
    # result = result.astype(float)
    # print result
    # cv2.imshow('RGB image after copying', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for i in range(len(rgb)):
    #     # if i % 50 == 0:
    #     #     cv2.imshow('Intermediate images', result)
    #     #     cv2.waitKey(0)
    #     #     cv2.destroyAllWindows()
    #     for j in range(len(rgb[0])):
    #         rgb_triplet = rgb[i, j]
    #         # print rgb_triplet
    #         lms_triplet = np.matmul(RGB_TO_LMS, rgb_triplet)
    #         try:
    #             loglms_triplet = np.log10(lms_triplet)
    #         except FloatingPointError, e:
    #             # print lms_triplet, rgb_triplet
    #             rgb_triplet = [1, 1, 1]
    #             lms_triplet = np.matmul(RGB_TO_LMS, rgb_triplet)
    #             loglms_triplet = np.log10(lms_triplet)
    #         lalphabeta_triplet = np.matmul(LOGLMS_TO_LALPHABETA, loglms_triplet)
    #         result[i, j] = lalphabeta_triplet
    #         # result[i, j] = lms_triplet

    # print 'Trying another conversion technique'
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    '''
    [[0.3811, 0.5783, 0.0402],
     [0.1967, 0.7244, 0.0782],
     [0.0241, 0.1288, 0.8444]]
    '''

    l = 0.3811 * r + 0.5783 * g + 0.0402 * b
    m = 0.1967 * r + 0.7244 * g + 0.0782 * b
    s = 0.0241 * r + 0.1288 * g + 0.8444 * b
    l[l == 0] = 1
    m[m == 0] = 1
    s[s == 0] = 1
    lms = np.copy(rgb).astype(np.float64)
    lms[:, :, 0] = l
    lms[:, :, 1] = m
    lms[:, :, 2] = s
    log_lms = np.log10(lms)

    log_l = log_lms[:, :, 0]
    log_m = log_lms[:, :, 1]
    log_s = log_lms[:, :, 2]
    '''
    [[0.57735, 0.57735, 0.57735],
     [0.40825, 0.40825, -0.81649],
     [0.70711, -0.70711, 0.00000]]
    '''
    l_ = 0.57735 * log_l + 0.57735 * log_m + 0.57735 * log_s
    alpha = 0.40825 * log_l + 0.40825 * log_m - 0.81649 * log_s
    beta = 0.70711 * log_l - 0.70711 * log_m + 0.0 * log_s

    # lms = np.dot(RGB_TO_LMS, rgb)
    # log_lms = np.log10(lms)
    # lalphabeta = np.dot(LOGLMS_TO_LALPHABETA, log_lms)
    lalphabeta = np.copy(rgb).astype(np.float64)
    lalphabeta[:, :, 0] = l_
    lalphabeta[:, :, 1] = alpha
    lalphabeta[:, :, 2] = beta
    # print 'Result(unoptimised) mean: ' + str(result.mean())
    # print 'Result(optimised) mean: ' + str(lalphabeta.mean())
    # print 'Result(unoptimised) std: ' + str(result.std())
    # print 'Result(optimised) std: ' + str(lalphabeta.std())
    # difference = result - lalphabeta
    # print 'Difference mean: ' + str(difference.mean())
    # print 'Difference std: ' + str(difference.std())
    return lalphabeta


def lalphabeta_to_rgb(lalphabeta):
    # result = np.copy(lalphabeta)
    # result = result.astype(float)
    # for i in range(len(lalphabeta)):
    #     for j in range(len(lalphabeta[0])):
    #         lalphabeta_triplet = lalphabeta[i, j]
    #         # lms_triplet = lalphabeta[i, j]
    #         # print lalphabeta_triplet
    #         loglms_triplet = np.matmul(LALPHABETA_TO_LOGLMS, lalphabeta_triplet)
    #         lms_triplet = np.power(10.0, loglms_triplet)
    #         rgb_triplet = np.matmul(LMS_TO_RGB, lms_triplet)
    #         result[i, j] = rgb_triplet

    l_ = lalphabeta[:, :, 0]
    alpha = lalphabeta[:, :, 1]
    beta = lalphabeta[:, :, 2]
    '''
    [[0.57735, 0.40825, 0.70711],
     [0.57735, 0.40825,-0.70711],
     [0.57735,-0.81649, 0.00000]]
    '''

    log_l = 0.57735 * l_ + 0.40825 * alpha + 0.70711 * beta
    log_m = 0.57735 * l_ + 0.40825 * alpha - 0.70711 * beta
    log_s = 0.57735 * l_ - 0.81649 * alpha + 0.00000 * beta
    log_lms = np.copy(lalphabeta).astype(np.float64)
    log_lms[:, :, 0] = log_l
    log_lms[:, :, 1] = log_m
    log_lms[:, :, 2] = log_s
    # try:
    # np.seterr(all='print')
    lms = np.power(10.0, log_lms)
    # np.seterr(all='raise')
    # lms = np.power(10.0, log_lms)
    # except FloatingPointError:
    #     np.seterr(all='print')
    #     lms = np.power(10.0, log_lms)
    #     np.seterr(all='raise')

    l = lms[:, :, 0]
    m = lms[:, :, 1]
    s = lms[:, :, 2]
    '''
    [[ 4.4679,-3.5873, 0.1193],
     [-1.2186, 2.3809,-0.1624],
     [ 0.0497,-0.2439, 1.2045]]
    '''
    r = 4.4679 * l - 3.5873 * m + 0.1193 * s
    g = -1.2186 * l + 2.3809 * m - 0.1624 * s
    b = 0.0497 * l - 0.2439 * m + 1.2045 * s

    # lms = np.dot(RGB_TO_LMS, rgb)
    # log_lms = np.log10(lms)
    # lalphabeta = np.dot(LOGLMS_TO_LALPHABETA, log_lms)
    rgb = np.copy(lalphabeta).astype(np.float64)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def most_basic_correction(target, source):
    '''
    Simple effect of one image on another. Works like filters in different photo editing apps.
    Sample outputs included in "Most Basic Correction" folder

    :param target: np.ndarray
    :param source: np.ndarray
    :return: color corrected image
    '''
    result = np.copy(target).astype(np.float64)
    # print result

    # print 'Starting to correct image'
    target_l = target[:, :, 0]
    # print target_l
    target_alpha = target[:, :, 1]
    target_beta = target[:, :, 2]

    source_l = source[:, :, 0]
    source_alpha = source[:, :, 1]
    source_beta = source[:, :, 2]

    target_l_mean = np.mean(target_l)
    # print 'Mean:',
    # print target_l_mean
    target_alpha_mean = np.mean(target_alpha)
    target_beta_mean = np.mean(target_beta)

    source_l_mean = np.mean(source_l)
    source_alpha_mean = np.mean(source_alpha)
    source_beta_mean = np.mean(source_beta)

    target_l_sd = np.std(target_l)
    target_alpha_sd = np.std(target_alpha)
    target_beta_sd = np.std(target_beta)

    source_l_sd = np.std(source_l)
    source_alpha_sd = np.std(source_alpha)
    source_beta_sd = np.std(source_beta)

    print 'Source and target statistics computed'
    result_l = result[:, :, 0]
    # print result_l
    result_alpha = result[:, :, 1]
    result_beta = result[:, :, 2]

    result_l = (result_l - target_l_mean) * (source_l_sd / target_l_sd) + source_l_mean
    # result_l = (((result_l - target_l_mean)/target_l_sd)*source_l_sd)+source_l_mean
    result_alpha = (result_alpha - target_alpha_mean) * (source_alpha_sd / target_alpha_sd) + source_alpha_mean
    result_beta = (result_beta - target_beta_mean) * (source_beta_sd / target_beta_sd) + source_beta_mean
    # result_l = (((result_l - target_l_mean)/target_l_sd)*source_l_sd)+source_l_mean
    # result = np.subtract(result, np.array([target_l_mean, target_alpha_mean, target_beta_mean], dtype=float))
    # result = np.divide(result, np.array([target_l_sd, target_alpha_sd, target_beta_sd], dtype=float))
    # result = np.multiply(result, np.array([source_l_sd, source_alpha_sd, source_beta_sd], dtype=float))
    # result = np.add(result, np.array([source_l_mean, source_alpha_mean, source_beta_mean], dtype=float))
    result[:, :, 0] = result_l
    result[:, :, 1] = result_alpha
    result[:, :, 2] = result_beta
    # print 'Correction complete'
    # print result
    return result


# Helper functions for advanced_correction function
def get_neighbors(i, j, rows, columns):
    neighbors = list()
    starting_neighbour = (i-1, j-1)
    for i in range(3):
        for j in range(3):
            neighbor = (starting_neighbour[0] + i, starting_neighbour[1] + j)
            if neighbor[0] < 0 or neighbor[1] < 0:
                continue
            if neighbor[0] > rows-1 or neighbor[1] > columns-1:
                continue
            neighbors.append(neighbor)
    return neighbors


def get_centre(cluster_index, cluster_height, cluster_width):
    # centre_i = 0
    # centre_j = 0
    centre_i = float((2*cluster_index[0]+1)*cluster_height)/2.0
    centre_j = float((2*cluster_index[1]+1)*cluster_width)/2.0
    centre = (centre_i, centre_j)
    return centre


def get_distance(point1, point2):
    distance = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    distance = math.sqrt(distance)
    return distance


def distance_from_neighbors(index, neighbors, cluster_height, cluster_width):
    distances = list()
    for neighbor in neighbors:
        centre = get_centre(neighbor, cluster_height, cluster_width)
        distance = get_distance(index, centre)
        if distance == 0:
            distance = 1
        distances.append(distance)
    return distances


def contribution_of_neighbors(target_pixel, i, j, neighbors, target_mean, target_sd, source_mean, source_sd):
    values = list()
    for neighbor in neighbors:
        m = neighbor[0]
        n = neighbor[1]
        try:
            l = (target_pixel[0]-target_mean[i, j, 0])*(source_sd[m, n, 0]/target_sd[i, j, 0])+source_mean[m, n, 0]
            l = 0.65*l + 0.35*target_pixel[0]
        except FloatingPointError:
            l = (target_pixel[0]-target_mean[i, j, 0])*(source_sd[m, n, 0])+source_mean[m, n, 0]
            l = 0.65*l + 0.35*target_pixel[0]
        try:
            alpha = (target_pixel[1]-target_mean[i, j, 1])*(source_sd[m, n, 1]/target_sd[i, j, 1])+source_mean[m, n, 1]
            alpha = 0.65*alpha + 0.35*target_pixel[1]
        except FloatingPointError:
            alpha = (target_pixel[1]-target_mean[i, j, 1])*(source_sd[m, n, 1])+source_mean[m, n, 1]
            alpha = 0.65*alpha + 0.35*target_pixel[1]
        try:
            beta = (target_pixel[2]-target_mean[i, j, 2])*(source_sd[m, n, 2]/target_sd[i, j, 2])+source_mean[m, n, 2]
            beta = 0.65*beta + 0.35*target_pixel[2]
        except FloatingPointError:
            beta = (target_pixel[2]-target_mean[i, j, 2])*(source_sd[m, n, 2])+source_mean[m, n, 2]
            beta = 0.65*beta + 0.35*target_pixel[2]
        value = [l, alpha, beta]
        # try:
        #     l = (target_pixel[0]-source_mean[m, n, 0])*(target_sd[i, j, 0]/source_sd[m, n, 0])+target_mean[i, j, 0]
        # except FloatingPointError:
        #     l = (target_pixel[0]-source_mean[m, n, 0])*(target_sd[i, j, 0])+target_mean[i, j, 0]
        # try:
        #     alpha = (target_pixel[1]-source_mean[m, n, 1])*(target_sd[i, j, 1]/source_sd[m, n, 1])+target_mean[i, j, 1]
        # except FloatingPointError:
        #     alpha = (target_pixel[1]-source_mean[m, n, 1])*(target_sd[i, j, 1])+target_mean[m, n, 1]
        # try:
        #     beta = (target_pixel[2]-source_mean[m, n, 2])*(target_sd[i, j, 2]/source_sd[m, n, 2])+target_mean[i, j, 2]
        # except FloatingPointError:
        #     beta = (target_pixel[2]-source_mean[m, n, 2])*(target_sd[i, j, 2])+target_mean[m, n, 2]
        # value = [l, alpha, beta]
        values.append(value)
    return values


def weighted_average(values, inverse_weights):
    weights = np.power(inverse_weights, -1.0)
    # print weights
    values = np.array(values)
    l = values[:, 0]
    alpha = values[:, 1]
    beta = values[:, 2]
    denominator = np.sum(weights, dtype=np.float64)
    result_l = np.multiply(l, weights)
    result_l = np.sum(result_l, dtype=np.float64)
    result_l = result_l/denominator
    result_alpha = np.multiply(alpha, weights)
    result_alpha = np.sum(result_alpha, dtype=np.float64)
    result_alpha = result_alpha/denominator
    result_beta = np.multiply(beta, weights)
    result_beta = np.sum(result_beta, dtype=np.float64)
    result_beta = result_beta/denominator
    result = list()
    # print 'L:',
    # print l
    # print 'Alpha:',
    # print alpha
    # print 'Beta:',
    # print beta
    # print 'L_numerator:',
    # print
    # print 'Weights:',
    # print weights
    # print 'Denominator:',
    # print denominator
    # print 'Result_l: ' + str(result_l)
    # print result
    result.append(result_l)
    result.append(result_alpha)
    result.append(result_beta)
    return result


# def normalise_distances(neighbors, distances, source_cluster_sd):
    # normalised_distances = list()
    # for i in range(len(neighbors)):
    #     neighbor = neighbors[i]
    #     distance = distances[i]
    #     source_sd = source_cluster_sd[neighbor[0], neighbor[1]]
    # return distances


def advanced_correction(target, source):
    '''
    Completely overlaps effect of source image on target image. Generates prisma app like effect.
    Does this by breaking the whole images into small fixed sized clusters, and mapping each cluster from source
    to target image.
    Sample outputs included in "Advanced Correction" folder

    :param target: np.ndarray
    :param source: np.ndarray
    :return: color corrected image
    '''
    target_height = len(target)
    target_width = len(target[0])

    source_height = len(source)
    source_width = len(source[0])

    target_cluster_height = 2  # Value to be set experimentally
    target_cluster_width = 2  # Value to be set experimentally

    source_cluster_height = 0  # Value to be set by formula
    source_cluster_width = 0  # Value to be set by formula

    number_of_cluster_rows = int(math.ceil(target_height / target_cluster_height))
    number_of_cluster_columns = int(math.ceil(target_width / target_cluster_width))

    source_cluster_height = float(source_height) / float(number_of_cluster_rows)
    source_cluster_width = float(source_width) / float(number_of_cluster_columns)

    target_cluster_mean = np.zeros(shape=(number_of_cluster_rows, number_of_cluster_columns, 3))
    target_cluster_sd = np.zeros(shape=(number_of_cluster_rows, number_of_cluster_columns, 3))
    source_cluster_mean = np.zeros(shape=(number_of_cluster_rows, number_of_cluster_columns, 3))
    source_cluster_sd = np.zeros(shape=(number_of_cluster_rows, number_of_cluster_columns, 3))

    print 'Computing cluster statistics'
    for i in range(number_of_cluster_rows):
        for j in range(number_of_cluster_columns):
            target_cluster = target[int(i * target_cluster_height): int(min((i + 1) * target_cluster_height,
                                                                            target_height - 1)),
                             int(j * target_cluster_width): int(min((j + 1) * target_cluster_width, target_width - 1))]
            source_cluster = source[int(i * source_cluster_height): int(min((i + 1) * source_cluster_height,
                                                                            source_height - 1)),
                             int(j * source_cluster_width): int(min((j + 1) * source_cluster_width, source_width - 1))]
            target_cluster_mean[i, j] = target_cluster.mean(dtype=np.float64, axis=(0, 1))
            target_cluster_sd[i, j] = target_cluster.std(dtype=np.float64, axis=(0, 1))
            source_cluster_mean[i, j] = source_cluster.mean(dtype=np.float64, axis=(0, 1))
            source_cluster_sd[i, j] = source_cluster.std(dtype=np.float64, axis=(0, 1))

    result = np.zeros(shape=target.shape)
    total_clusters = number_of_cluster_columns*number_of_cluster_rows
    print 'Computing pixels'
    for i in range(number_of_cluster_rows):
        for j in range(number_of_cluster_columns):
            neighbors = get_neighbors(i, j, number_of_cluster_rows, number_of_cluster_columns)
            target_cluster = target[int(i * target_cluster_height): int(min((i + 1) * target_cluster_height,
                                                                            target_height - 1)),
                             int(j * target_cluster_width): int(min((j + 1) * target_cluster_width, target_width - 1))]
            cluster_number = i*number_of_cluster_columns + j
            percent = float(cluster_number*100)/float(total_clusters)
            if float(percent/5) - int(percent/5) < 0.001:
                print "{0:.2f}% complete".format(percent)
            for m in range(len(target_cluster)):
                for n in range(len(target_cluster[0])):
                    target_pixel_index = (i*target_cluster_height + m, j*target_cluster_width + n)
                    target_pixel = target_cluster[m, n]
                    distances = distance_from_neighbors(target_pixel_index, neighbors, target_cluster_height,
                                                        target_cluster_width)
                    values = contribution_of_neighbors(target_pixel, i, j, neighbors, target_cluster_mean,
                                                       target_cluster_sd, source_cluster_mean, source_cluster_sd)
                    # if len(values) == 9:
                    #     print 'Yes'
                    # inverse_weights = normalise_distances(neighbors, distances, source_cluster_sd)
                    pixel_value = weighted_average(values, distances)
                    result[target_pixel_index[0], target_pixel_index[1]] = pixel_value

    return result


def main():
    # target = 'Flowerbed.jpg'
    # target = 'Palash_1.JPG'
    # target = 'Snowball_1.jpg'
    # target = 'target1.jpg'
    # target = 'target2.jpg'
    target = 'target3.jpg'
    # source = 'Painting_flowers.jpg'
    # source = 'PaintingFlowerGarden.jpg'
    # source = 'PaintingTreeBarkFall.jpg'
    # source = 'source1.jpg'
    # source = 'source2.jpg'
    # source = 'source3.jpg'
    # source = 'source4.jpg'
    # source = 'source5.jpg'
    source = 'source6.png'
    input_image = cv2.imread(target)
    source_image = cv2.imread(source)
    cv2.imshow('Original Image', input_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imshow('Source Image', source_image)
    print 'Press any key to start computation'
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    input_image = input_image[:, :, (2, 1, 0)]
    source_image = source_image[:, :, (2, 1, 0)]
    # cv2.imshow('Original Image in RGB', input_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
    # print input_image
    # print len(input_image)
    # print len(input_image[0])
    # b = np.copy(input_image)
    # g = np.copy(input_image)
    # r = np.copy(input_image)
    # b[:, :, 1] = 0
    # b[:, :, 2] = 0
    #
    # g[:, :, 0] = 0
    # g[:, :, 2] = 0
    #
    # r[:, :, 0] = 0
    # r[:, :, 1] = 0

    # print b
    # cv2.imshow('Red image', r)
    # cv2.waitKey(0)
    # cv2.imshow('Green image', g)
    # cv2.waitKey(0)
    # cv2.imshow('Blue image', b)
    # cv2.waitKey(0)

    # input_red = input_image[:, :, 2]
    # input_green = input_image[:, :, 1]
    # input_blue = input_image[:, :, 0]
    # print input_red
    print 'Target image being converted to l,alpha,beta'
    input_lalphabeta = rgb_to_lalphabeta(input_image)
    print 'Source image being converted to l,alpha,beta'
    source_lalphabeta = rgb_to_lalphabeta(source_image)
    # cv2.imshow('LAB Image', input_lalphabeta)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print 'Conversion complete'
    # print 'Basic color correction started'
    # corrected_lalphabeta = most_basic_correction(input_lalphabeta, source_lalphabeta)
    print 'Advanced color correction started'
    corrected_lalphabeta = advanced_correction(input_lalphabeta, source_lalphabeta)

    print 'Target image corrected'
    print 'Corrected image being converted back to rgb'
    corrected_rgb = lalphabeta_to_rgb(corrected_lalphabeta)
    corrected_rgb = (corrected_rgb / 255.0)
    # print corrected_rgb
    # cv2.imshow('Restored RGB Image', corrected_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print 'Conversion complete'
    corrected_rgb = corrected_rgb[:, :, (2, 1, 0)]
    # print corrected_rgb
    cv2.imshow('Corrected Image', corrected_rgb)
    cv2.imwrite(str(target.split('.')[0] + '+' + source.split('.')[0] + '.jpg'), corrected_rgb * 255.0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


np.seterr(all='raise')
main()

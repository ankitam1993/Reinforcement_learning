import sys

def check_wall_hit(new_i , new_j , i , j ):

    if (new_i == 2 and new_j == 2) or (new_i == 3 and new_j == 2):
        return i , j
    else:
        return new_i , new_j

def vi(prev_matrix,reward_matrix):

    if vi.count == 100 :
        sys.exit()

    matrix = [[0 for x in range(5)] for y in range(5)]
    matrix[2][2] = matrix[3][2] = None
    sum = 8

    while(sum>-1):

        for i in range(4,-1,-1):
            for j in range(4,-1,-1):

                # going diagonal wise
                if i + j == sum:

                    #print 'calculating i , j :' , i , j
                    # by passing thwhe goal state or wall states - no values need to be computed on these
                    if (i ==4 and j == 4) or (i == 2 and j == 2) or (i ==3 and j == 2):
                         continue

                    # Taking left action : i , j - 1 , at each action check 4 possibilities ( action succeeds - which is left ,
                    # -90, +90 , at same stage

                    left_succeed_i = i
                    left_succeed_j = j - 1 if j-1 >= 0 else j

                    left_succeed_i ,left_succeed_j = check_wall_hit(left_succeed_i,left_succeed_j, i, j)

                    #print 'left_succeed_:' , left_succeed_i , left_succeed_j, reward_matrix[left_succeed_i][left_succeed_j], prev_matrix[left_succeed_i][left_succeed_j]

                    left_pos_90_i = i - 1 if i-1 >= 0 else i
                    left_pos_90_j = j

                    left_pos_90_i, left_pos_90_j = check_wall_hit(left_pos_90_i, left_pos_90_j, i, j)

                    #print 'left_pos_:', left_pos_90_i, left_pos_90_j

                    left_neg_90_i = i + 1 if i+1 <= 4 else i
                    left_neg_90_j = j

                    left_neg_90_i, left_neg_90_j = check_wall_hit(left_neg_90_i, left_neg_90_j, i, j)

                    #print 'left_neg_:', left_neg_90_i, left_neg_90_j , reward_matrix[left_neg_90_i][left_neg_90_j] , prev_matrix[left_neg_90_i][left_neg_90_j]

                    left_stay_i = i
                    left_stay_j = j


                    moving_left_score = 0.8*(reward_matrix[left_succeed_i][left_succeed_j] + prev_matrix[left_succeed_i][left_succeed_j]) +\
                                        0.05*(reward_matrix[left_pos_90_i][left_pos_90_j] + prev_matrix[left_pos_90_i][left_pos_90_j]) +\
                                        0.05*(reward_matrix[left_neg_90_i][left_neg_90_j] + prev_matrix[left_neg_90_i][left_neg_90_j]) +\
                                        0.1*(reward_matrix[left_stay_i][left_stay_j] + prev_matrix[left_stay_i][left_stay_j])


                    # Taking right action : i , j + 1

                    right_succeed_i = i
                    right_succeed_j = j + 1 if j + 1 <= 4 else j

                    right_succeed_i, right_succeed_j = check_wall_hit(right_succeed_i, right_succeed_j, i, j)

                    right_pos_90_i = i - 1 if i - 1 >= 0 else i
                    right_pos_90_j = j

                    right_pos_90_i, right_pos_90_j = check_wall_hit(right_pos_90_i, right_pos_90_j, i, j)

                    right_neg_90_i = i + 1 if i + 1 <= 4 else i
                    right_neg_90_j = j

                    right_neg_90_i, right_neg_90_j = check_wall_hit(right_neg_90_i, right_neg_90_j, i, j)

                    right_stay_i = i
                    right_stay_j = j

                    moving_right_score = 0.8 * (reward_matrix[right_succeed_i][right_succeed_j] + prev_matrix[right_succeed_i][right_succeed_j]) + \
                                        0.05 * (reward_matrix[right_pos_90_i][right_pos_90_j] + prev_matrix[right_pos_90_i][right_pos_90_j]) + \
                                        0.05 * (reward_matrix[right_neg_90_i][right_neg_90_j] + prev_matrix[right_neg_90_i][right_neg_90_j]) + \
                                        0.1 * (reward_matrix[right_stay_i][right_stay_j] + prev_matrix[right_stay_i][right_stay_j])

                    # Taking up action : i-1 , j

                    up_succeed_i = i - 1 if i - 1 >= 0 else i
                    up_succeed_j = j

                    up_succeed_i, up_succeed_j = check_wall_hit(up_succeed_i, up_succeed_j, i, j)

                    up_pos_90_i = i
                    up_pos_90_j = j + 1 if j + 1 <= 4 else j

                    up_pos_90_i, up_pos_90_j = check_wall_hit(up_pos_90_i, up_pos_90_j, i, j)

                    up_neg_90_i = i
                    up_neg_90_j = j - 1 if j-1 >= 0 else j

                    up_neg_90_i, up_neg_90_j = check_wall_hit(up_neg_90_i, up_neg_90_j, i, j)

                    up_stay_i = i
                    up_stay_j = j

                    moving_up_score = 0.8 * (reward_matrix[up_succeed_i][up_succeed_j] + prev_matrix[up_succeed_i][up_succeed_j]) + \
                                      0.05 * (reward_matrix[up_pos_90_i][up_pos_90_j] + prev_matrix[up_pos_90_i][up_pos_90_j]) + \
                                      0.05 * (reward_matrix[up_neg_90_i][up_neg_90_j] + prev_matrix[up_neg_90_i][up_neg_90_j]) + \
                                      0.1 * (reward_matrix[up_stay_i][up_stay_j] + prev_matrix[up_stay_i][up_stay_j])

                    # Taking down action : i+1 , j


                    down_succeed_i = i + 1 if i + 1 <= 4 else i
                    down_succeed_j = j

                    down_succeed_i, down_succeed_j = check_wall_hit(down_succeed_i, down_succeed_j, i, j)

                    down_pos_90_i = i
                    down_pos_90_j = j + 1 if j + 1 <= 4 else j

                    down_pos_90_i, down_pos_90_j = check_wall_hit(down_pos_90_i, down_pos_90_j, i, j)

                    down_neg_90_i = i
                    down_neg_90_j = j - 1 if j - 1 >= 0 else j

                    down_neg_90_i, down_neg_90_j = check_wall_hit(down_neg_90_i, down_neg_90_j, i, j)

                    down_stay_i = i
                    down_stay_j = j

                    moving_down_score = 0.8 * (reward_matrix[down_succeed_i][down_succeed_j] + prev_matrix[down_succeed_i][down_succeed_j]) + \
                                        0.05 * (reward_matrix[down_pos_90_i][down_pos_90_j] + prev_matrix[down_pos_90_i][down_pos_90_j]) + \
                                        0.05 * (reward_matrix[down_neg_90_i][down_neg_90_j] + prev_matrix[down_neg_90_i][down_neg_90_j]) + \
                                        0.1 * (reward_matrix[down_stay_i][down_stay_j] + prev_matrix[down_stay_i][down_stay_j])

                    # At the end take maximum of all  and update the matrix

                    matrix[i][j] = round(max(moving_left_score,moving_right_score,moving_up_score,moving_down_score),3)
                    #print matrix[i][j]
        sum -= 1

    vi.count +=1
    print 'matrix no {0} : {1} ' .format(vi.count, matrix)
    vi(matrix,reward_matrix)

reward_matrix = [[0 for x in range(5)] for y in range(5)]

reward_matrix[4][4] = 10
reward_matrix[4][2] = -10

prev_matrix = [[0 for x in range(5)] for y in range(5)]
prev_matrix[2][2] = prev_matrix[3][2] = None

#print prev_matrix
vi.count = 0
vi(prev_matrix,reward_matrix)

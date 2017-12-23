down_succeed_i = i - 1 if i - 1 >= 0 else i
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

moving_down_score = 0.8 * (
    reward_matrix[down_succeed_i][down_succeed_j] + prev_matrix[down_succeed_i][down_succeed_j]) + \
                  0.05 * (reward_matrix[down_pos_90_i][down_pos_90_j] + prev_matrix[down_pos_90_i][down_pos_90_j]) + \
                  0.05 * (reward_matrix[down_neg_90_i][down_neg_90_j] + prev_matrix[down_neg_90_i][down_neg_90_j]) + \
                  0.1 * (reward_matrix[down_stay_i][down_stay_j] + prev_matrix[down_stay_i][down_stay_j])
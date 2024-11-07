task_mem_requirements = [10, 20, 15, 25, 30, 5, 12, 40, 35, 22, 18, 28] 

def computation_cost(cur_seg):
    return sum(task_mem_requirements[i - 1] for i in cur_seg)



def max_segment_mem_req(cur_seg):
    return sum(task_mem_requirements[i - 1] for i in cur_seg)


def optimal_seg_finder(cur_seg, index, n, M, min_cost, optimal_seg, cur_seg_result):
    print(max_segment_mem_req(cur_seg))
    if max_segment_mem_req(cur_seg) <= M:
        cost = computation_cost(cur_seg)
        if cost < min_cost[0]:
            min_cost[0] = cost
            optimal_seg[:] = cur_seg[:]

    i = index
    while i <= n:
        optimal_seg_finder(cur_seg.copy() + [i], i + 1, n, M, min_cost, optimal_seg, cur_seg_result)
        i += 1


def rank_finder(tk, mk, M):
    consumed = 0
    ranks = [0] * tk
    current_rank = 1
    p = tk - 1

    while p >= 0:
        if consumed + mk[p] > M:
            consumed = mk[p]
            current_rank += 1
        else:
            consumed += mk[p]
        ranks[p] = current_rank
        p -= 1

    return ranks



n = 12  
M = 100  
min_cost = [float('inf')]
optimal_seg = []
cur_seg = []

optimal_seg_finder(cur_seg, 1, n, M, min_cost, optimal_seg, [])

print("Optimal Segmentation:", optimal_seg)
print("Minimum Cost:", min_cost[0])

tk = 4  
mk = [100,80,20,90]  
ranks = rank_finder(tk, mk, M)

print("Segment Ranks:", ranks)

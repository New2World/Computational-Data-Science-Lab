#pragma once

#define QUE_LEN 1000
#define QUE_ST(x) (x * QUE_LEN)
#define QUE_ED(x) (x * QUE_LEN + QUE_LEN)

// judge if queue overflow
__device__ bool que_isFull(int que_h, int que_t){
    return que_t == que_h;
}

// judge if queue is empty
__device__ bool que_isEmpty(int que_h, int que_t, int index){
    return (que_t == que_h + 1) || (que_t == que_h + 1 - QUE_LEN);
}

// clear all elements in queue and reset queue
__device__ void que_init(int& que_h, int& que_t, int index){
    que_h = QUE_ST(index);
    que_t = que_h + 1;
}

// enqueue
__device__ bool que_enque(int* queue, int que_h, int& que_t, int val, int index){
    if(que_isFull(que_h, que_t))
        return false;
    int tail = que_t - 1;
    if(tail < QUE_ST(index))
        tail += QUE_LEN;
    queue[tail] = val;
    que_t++;
    if(que_t >= QUE_ED(index))
        que_t -= QUE_LEN;
    return true;
}

// dequeue
__device__ int que_deque(int* queue, int& que_h, int que_t, int index){
    int val = -1;
    if(que_isEmpty(que_h, que_t, index))
        return val;
    val = queue[que_h];
    que_h++;
    if(que_h >= QUE_ED(index))
        que_h -= QUE_LEN;
    return val;
}